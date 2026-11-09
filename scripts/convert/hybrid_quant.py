#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hybrid_quant.py —— 分类头 FP16 + 主干 INT8 混合量化

背景（为什么需要它）：
  YOLOv8 的 cv3（分类分支）权重存在 ±13 的离群值，而均值只有 0.1 左右。
  INT8 只有 256 级，为容下离群值步长被迫拉大，正常权重被压成 0/±1，
  分类头精度崩塌 -> mAP 掉 9~11 点。

  w16a16i（全 16 位）在 RKNN 2.3.2 + RK3588 上已实测崩溃：
    E RKNN: Meet unsupported tensor dtype in Sub, /model.22/Constant_12_output_0
    Illegal instruction (core dumped)
  => int16 只有 API 枚举，后端没实现。只剩 FP16 / INT8 两条路。

  所以走官方标准做法：只把 cv3 分类头标成 float16，其余保持 w8a8。

流程（RKNN 官方三步）：
  step1: hybrid_quantization_step1(dataset) -> 生成 *.quantization.cfg / *.data / *.model
  修改 : 在 cfg 的 custom_quantize_layers 里把 cv3 层写成 float16（本脚本自动做）
  step2: hybrid_quantization_step2(...)     -> 重新编译出 rknn

用法（WSL 的 rknn-env 里）：
  python hybrid_quant.py \\
    --onnx models/yolov8n_2out_bs8.onnx \\
    --dataset /mnt/d/01_STUDY/rk3588/calib/calib200 --num 200 \\
    --img /mnt/d/01_STUDY/rk3588/calib/calib200/000000057027.jpg \\
    --pattern model.22/cv3 \\
    --box-div 8 --out models/yolov8n_2out_hybrid.rknn

  # 只想看看会改哪些层（不编译）：
  python hybrid_quant.py ... --dry-run
"""

import os
import re
import sys
import glob
import argparse


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def collect_images(d, num):
    return [os.path.join(d, n) for n in sorted(os.listdir(d))
            if os.path.isfile(os.path.join(d, n)) and n.lower().endswith(IMG_EXTS)][:num]


def letterbox(img, size=640, color=(114, 114, 114)):
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    left = (size - nw) // 2
    out = np.full((size, size, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out


def list_candidate_layers(onnx_path, patterns):
    """从 ONNX 图里找出要保 FP16 的层名（cfg 的 key 用 ONNX node name）"""
    import onnx
    m = onnx.load(onnx_path)
    names = []
    for n in m.graph.node:
        nm = n.name or ''
        if not nm:
            continue
        if any(p in nm for p in patterns):
            names.append(nm)
    return sorted(set(names))


def health(outs, box_div=1.0):
    import numpy as np
    print("\n[health] 输出 %d 个张量:" % len(outs))
    arrs = []
    for k, o in enumerate(outs):
        a = np.array(o)
        print("  [%d] shape=%s dtype=%s" % (k, a.shape, a.dtype))
        arrs.append(a)

    boxes = scores = None
    for a in arrs:
        if a.ndim != 3:
            continue
        c = a.shape[1]
        if c == 4 and boxes is None:
            boxes = a[0]
        elif c in (80, 84, 85) and scores is None:
            scores = a[0]
    if boxes is None or scores is None:
        print("  !! 没同时找到 4 通道和 80 通道输出")
        return False

    raw_max = float(boxes.max())
    if box_div != 1.0:
        boxes = boxes * box_div
    cmax = scores.max(axis=0)
    print("  boxes : mean=%.2f max=%.2f (原始 %.2f x %.0f)"
          % (boxes.mean(), boxes.max(), raw_max, box_div))
    print("  scores: max=%.4f mean=%.4f min=%.4f"
          % (cmax.max(), cmax.mean(), cmax.min()))
    ok = cmax.max() > 1e-6 and boxes.max() > 100
    print("  判定: %s" % ("✅ 正常" if ok else "❌ 异常"))
    return ok


def find_step1_outputs(onnx_path):
    """step1 的产物默认落在 CWD，也可能落在 onnx 同目录，两处都找"""
    base = os.path.splitext(os.path.basename(onnx_path))[0]
    dirs = [os.getcwd(), os.path.dirname(os.path.abspath(onnx_path))]
    res = {}
    for ext in ('.quantization.cfg', '.data', '.model'):
        for d in dirs:
            p = os.path.join(d, base + ext)
            if os.path.isfile(p):
                res[ext] = p
                break
    if not res:
        print("[step1] 两个目录都没找到产物，实际存在的同类文件：")
        for d in dirs:
            for p in sorted(glob.glob(os.path.join(d, '*quantization*'))):
                print("   ", p)
    return res


def inject_custom_layers(cfg_path, layers):
    """把 custom_quantize_layers 填成 {层名: float16}

    cfg 是 YAML，两种形态都要兼容：
      custom_quantize_layers: {}          <- 空
      custom_quantize_layers:             <- 已有子项
        'xxx': 'float16'
    """
    text = open(cfg_path, encoding='utf-8').read()

    def block(indent):
        s = "%scustom_quantize_layers:\n" % indent
        for n in layers:
            s += "%s  '%s': 'float16'\n" % (indent, n)
        return s

    pat_inline = re.compile(r'(?m)^([ \t]*)custom_quantize_layers:[ \t]*\{\}[ \t]*$')
    pat_block = re.compile(r'(?m)^([ \t]*)custom_quantize_layers:[ \t]*$\n((?:[ \t]+.*\n?)*)')

    if pat_inline.search(text):
        text = pat_inline.sub(lambda m: block(m.group(1)).rstrip('\n'), text, count=1)
        how = 'inline({}) -> block'
    elif pat_block.search(text):
        text = pat_block.sub(lambda m: block(m.group(1)), text, count=1)
        how = 'block -> block(overwritten)'
    else:
        text = text.rstrip('\n') + '\n' + block('')
        how = 'appended at EOF'

    open(cfg_path, 'w', encoding='utf-8').write(text)
    print("[cfg] custom_quantize_layers 注入方式: %s，共 %d 层" % (how, len(layers)))
    return how


def main():
    import numpy as np
    import cv2
    from rknn.api import RKNN

    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True, help='已拆好的 2out onnx')
    ap.add_argument('--dataset', required=True, help='校准目录（绝对路径）')
    ap.add_argument('--num', type=int, default=200)
    ap.add_argument('--img', required=True, help='体检单图')
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--pattern', action='append', default=None,
                    help='要保 FP16 的层名关键字，可多次给。默认 model.22/cv3')
    ap.add_argument('--box-div', type=float, default=8.0)
    ap.add_argument('--core', default='012')
    ap.add_argument('--dry-run', action='store_true',
                    help='只跑到 step1 并打印要改的层，不编译')
    ap.add_argument('--skip-step1', action='store_true',
                    help='复用已存在的 .quantization.cfg/.data/.model，'
                         '跳过耗时的 step1 量化（改完 cfg 想重编译时用）')
    ap.add_argument('--out', default='models/yolov8n_2out_hybrid.rknn')
    args = ap.parse_args()

    patterns = args.pattern or ['model.22/cv3']

    for f in (args.onnx, args.img):
        if not os.path.isfile(f):
            print("!! 不存在:", f); sys.exit(1)
    if not os.path.isdir(args.dataset):
        print("!! 校准目录不存在:", args.dataset); sys.exit(1)

    # ---------- 0. 先列出要改的层 ----------
    print("[0/3] 扫描 ONNX 里匹配 %s 的层 ..." % patterns)
    layers = list_candidate_layers(args.onnx, patterns)
    if not layers:
        print("!! 一层都没匹配到，检查 --pattern。图里的节点名样例：")
        import onnx
        m = onnx.load(args.onnx)
        for n in m.graph.node[:15]:
            print("   ", n.op_type, n.name)
        sys.exit(1)
    print("  匹配到 %d 层：" % len(layers))
    for n in layers:
        print("    %s" % n)

    # ---------- 1. 校准集 ----------
    files = collect_images(args.dataset, args.num)
    if not files:
        print("!! 校准目录没图"); sys.exit(1)
    txt = os.path.expanduser('~/calib_list_hybrid.txt')
    with open(txt, 'w') as f:
        f.write('\n'.join(os.path.abspath(p) for p in files) + '\n')
    print("\n校准集: %d 张 -> %s" % (len(files), txt))

    # ---------- 2. step1 ----------
    rknn = None
    if args.skip_step1:
        print("\n[1/3] --skip-step1：复用已有 step1 产物，跳过量化")
    else:
        print("\n[1/3] hybrid_quantization_step1 ...")
        rknn = RKNN(verbose=False)
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                    target_platform='rk3588')
        if rknn.load_onnx(model=args.onnx) != 0:
            print("!! load_onnx 失败"); rknn.release(); sys.exit(1)
        if rknn.hybrid_quantization_step1(dataset=txt, rknn_batch_size=1) != 0:
            print("!! step1 失败"); rknn.release(); sys.exit(1)

    outs = find_step1_outputs(args.onnx)
    print("[step1] 产出文件:")
    for k, v in outs.items():
        print("   %-20s %s (%.2f MB)" % (k, v, os.path.getsize(v) / 1048576.0))

    cfg = outs.get('.quantization.cfg')
    if cfg is None:
        print("!! 没找到 .quantization.cfg，CWD 与 onnx 目录里的同类文件：")
        for d in (os.getcwd(), os.path.dirname(os.path.abspath(args.onnx))):
            for p in sorted(glob.glob(os.path.join(d, '*quantization*'))):
                print("   ", p)
        if rknn is not None:
            rknn.release()
        sys.exit(1)
    missing = [k for k in ('.data', '.model') if k not in outs]
    if missing:
        print("!! 缺少 step1 产物: %s（重跑 step1，别加 --skip-step1）" % missing)
        if rknn is not None:
            rknn.release()
        sys.exit(1)

    # ---------- 3. 改 cfg ----------
    print("\n[2/3] 修改 %s ..." % cfg)
    before = open(cfg, encoding='utf-8').read()
    print("  修改前 custom_quantize_layers 附近内容:")
    m = re.search(r'(?m)^[ \t]*custom_quantize_layers:.*$', before)
    print("   ", m.group(0).strip() if m else "（没找到该字段！）")

    inject_custom_layers(cfg, layers)

    after = open(cfg, encoding='utf-8').read()
    m2 = re.search(r'(?ms)^[ \t]*custom_quantize_layers:.*?(?=^\S|\Z)', after)
    print("  修改后片段（前 8 行）:")
    for ln in (m2.group(0).splitlines()[:8] if m2 else []):
        print("   ", ln)

    if args.dry_run:
        print("\n--dry-run：已停在 cfg 修改。检查上面 cfg 内容无误后，"
              "去掉 --dry-run 重跑即可完成 step2。")
        if rknn is not None:
            rknn.release()
        sys.exit(0)

    # ---------- 4. step2 ----------
    print("\n[3/3] hybrid_quantization_step2 编译 ...")
    rknn2 = RKNN(verbose=False)

    import inspect
    sig = inspect.signature(rknn2.hybrid_quantization_step2)
    print("[step2] 实际签名: hybrid_quantization_step2%s" % sig)

    # 不同版本的参数名不一致（model_script / model_script_path / quantize_cfg ...），
    # 这里按参数名语义自动匹配，都匹配不到就用位置参数
    pnames = [p for p in sig.parameters if p not in ('self',)]
    kw = {}
    for p in pnames:
        low = p.lower()
        if 'model' in low and 'input' in low:
            kw[p] = outs['.model']
        elif 'data' in low and 'input' in low:
            kw[p] = outs['.data']
        elif 'script' in low or 'cfg' in low or 'config' in low:
            kw[p] = cfg
        elif 'dataset' in low:
            kw[p] = None
        elif 'batch' in low:
            kw[p] = 1
    # 位置参数（有默认值且没匹配上的，留给默认）
    print("[step2] 传参: %s" % {k: os.path.basename(str(v)) if v else v
                               for k, v in kw.items()})

    try:
        ret = rknn2.hybrid_quantization_step2(**kw)
    except TypeError as e:
        print("[step2] 关键字调用失败(%s)，改试纯位置参数" % e)
        ret = rknn2.hybrid_quantization_step2(outs['.model'], outs['.data'], cfg, 1)

    if ret != 0:
        print("!! step2 失败"); rknn2.release(); sys.exit(1)

    # 混合模型在模拟器上可能跑不起来，体检失败不阻断导出
    print("[step2] init_runtime + 体检 ...")
    ok = True
    try:
        if rknn2.init_runtime(core_mask=None) != 0:
            raise RuntimeError('init_runtime 返回非 0')
        img = cv2.imread(args.img)
        if img is None:
            raise RuntimeError('读不到体检图 %s' % args.img)
        inp = cv2.cvtColor(letterbox(img, args.size),
                           cv2.COLOR_BGR2RGB)[None].astype(np.uint8)
        houts = rknn2.inference(inputs=[inp])
        ok = health(houts, args.box_div)
    except Exception as e:
        print("[warn] 模拟器体检失败（混合模型常见，不影响导出）: %s" % e)
        print("[warn] 导出后必须在板端跑一次 eval 确认精度")

    if ok:
        if rknn2.export_rknn(args.out) == 0:
            print("\n✅ 导出成功: %s (%.2f MB)"
                  % (args.out, os.path.getsize(args.out) / 1048576.0))
            print("   对比参考: INT8 4.74 MB / 28.68 ms，FP16 约 9.5 MB / 46.75 ms")
            print("   混合量化的健康体积: 5.5~6.5 MB（若仍 4.74 -> float16 没生效）")
        else:
            print("\n!! 导出失败")
    else:
        print("\n❌ 体检不通过，先别用")
    rknn2.release()
    if rknn is not None:
        rknn.release()


if __name__ == '__main__':
    main()
