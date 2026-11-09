# -*- coding: utf-8 -*-
"""
混合量化：把「输出张量」（可选：连同分类分支 cv3）单独保持 float32，其余全部 INT8。

根因：RKNN 把 [1,84,8400] 这个「box + cls 拼接」的输出张量整体做了 int8 量化，
     量程被 box 分支（最大 641）主导，步长约 2.5，
     导致 0~1 区间的类别分数被全部抹平成 0。

用法：
    mkdir -p ~/hyb2 && cd ~/hyb2
    python ~/hybrid_fix.py --onnx ~/yolov8n.onnx --dataset ~/calib \
        --img ~/calib/000000000009.jpg --mode output \
        --out ~/rknn_out/yolov8n_hybrid.rknn

    --mode output  只把输出张量设为 float32（首选，改动最小）
    --mode both    输出张量 + 所有 cv3 层都设为 float32（上一招无效时用）
    --mode cv3     只改 cv3 层（一般不单独用）
"""

import os
import sys
import re
import glob
import argparse
import numpy as np
import cv2
from rknn.api import RKNN

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
HDR = re.compile(r'^(\s{4})(\S+):\s*$')   # 4 空格缩进的块头，如 "    /model.0/conv/Conv_output_0:"


def collect_images(d, num):
    return [os.path.join(d, n) for n in sorted(os.listdir(d))
            if os.path.isfile(os.path.join(d, n)) and n.lower().endswith(IMG_EXTS)][:num]


def letterbox(img, size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nw, nh = int(round(w * s)), int(round(h * s))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    left = (size - nw) // 2
    return cv2.copyMakeBorder(im, top, size - nh - top, left, size - nw - left,
                              cv2.BORDER_CONSTANT, value=color)


def block_range(lines, i):
    """给块头行号 i，返回该块内容(不含头)的 [起, 止)"""
    j = i + 1
    while j < len(lines):
        if HDR.match(lines[j]) or (lines[j].strip() and not lines[j].startswith('        ')):
            break
        j += 1
    return i + 1, j


def set_float32(lines, i):
    """把第 i 行这个块改成 float32：dtype=float32，scale/zero_point 置空"""
    a, b = block_range(lines, i)
    body = lines[a:b]
    out = []
    k = 0
    while k < len(body):
        ln = body[k]
        s = ln.strip()
        if s.startswith('dtype:'):
            out.append('        dtype: float32\n')
        elif s.startswith('scale:') or s.startswith('zero_point:'):
            key = s.split(':')[0]
            out.append('        %s: []\n' % key)
            k += 1
            # 吃掉原本多行列表项
            while k < len(body) and body[k].strip().startswith('-'):
                k += 1
            continue
        else:
            out.append(ln)
        k += 1
    return lines[:a] + out + lines[b:]


def pick_output_name(lines, onnx_path):
    """挑出「输出张量」在 cfg 里的名字"""
    headers = [(i, HDR.match(l).group(2)) for i, l in enumerate(lines) if HDR.match(l)]
    # 1) 优先用 ONNX 图输出名精确匹配
    names = []
    try:
        import onnx
        m = onnx.load(onnx_path)
        names = [o.name for o in m.graph.output]
    except Exception as e:
        print("[warn] 读不了 ONNX 输出名（不影响）:", e)
    for n in names:
        for i, h in headers:
            if h == n:
                print("[pick] 按 ONNX 输出名匹配 ->", h)
                return i, h
    # 2) 退而求其次：最靠后的 Concat
    cand = [(i, h) for i, h in headers if 'Concat' in h or 'concat' in h]
    if cand:
        i, h = cand[-1]
        print("[pick] 按 Concat 关键字 ->", h)
        return i, h
    # 3) 再退：最后一个块
    i, h = headers[-1]
    print("[pick] 兜底：取最后一个块 ->", h)
    return i, h


def report(name, outs):
    o = np.array(outs[0])
    print("  输出 shape=%s dtype=%s" % (o.shape, o.dtype))
    if o.ndim == 3:
        x = o[0] if o.shape[1] in (84, 85) and o.shape[2] > 100 else o[0].T
    else:
        x = None
    if x is None:
        print("  ！！ 认不出布局"); return False
    box = x[:4]
    cls = x[4:84]
    cmax = cls.max(axis=0)
    zero_rows = int(np.sum(~x.any(axis=1)))
    uniq = int(np.unique(np.round(box.reshape(-1), 4)).size)
    print("  框回归 : mean=%.2f max=%.2f   框 unique=%d/%d"
          % (box.mean(), box.max(), uniq, box.reshape(-1).size))
    print("  类别分 : max=%.4f mean=%.4f   全 0 行=%d/%d"
          % (cmax.max(), cmax.mean(), zero_rows, x.shape[0]))
    ok = cmax.max() > 1e-6
    print("  判定: %s" % ("✅ 正常" if ok else "❌ 类别分仍为 0"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--dataset', default='.')
    ap.add_argument('--img', required=True)
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--num', type=int, default=200)
    ap.add_argument('--mode', default='concat',
                    choices=['concat', 'output', 'cv3', 'both', 'all'],
                    help='concat=输出张量+所有 Concat 层设 float32（首选）；'
                         'both=再叠加 cv3 层；all=两者全开')
    ap.add_argument('--out', default='hybrid.rknn')
    args = ap.parse_args()

    for f in (args.onnx, args.img):
        if not os.path.isfile(f):
            print("!! 文件不存在:", f); sys.exit(1)

    files = collect_images(args.dataset, args.num)
    if not files:
        print("!! 校准目录没图:", args.dataset); sys.exit(1)
    txt = os.path.abspath('calib_list.txt')
    open(txt, 'w').write('\n'.join(files) + '\n')
    print("校准集: %d 张" % len(files))

    img = cv2.imread(args.img)
    inp = cv2.cvtColor(letterbox(img, args.size), cv2.COLOR_BGR2RGB)[None].astype(np.uint8)

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform='rk3588')
    if rknn.load_onnx(model=args.onnx) != 0:
        print("!! load_onnx 失败"); sys.exit(1)

    print("[1/4] hybrid_quantization_step1 ...")
    if rknn.hybrid_quantization_step1(dataset=txt) != 0:
        print("!! step1 失败"); rknn.release(); sys.exit(1)

    cfgs = glob.glob('*.quantization.cfg')
    if not cfgs:
        print("!! 没找到 cfg"); rknn.release(); sys.exit(1)
    cfg = cfgs[0]
    lines = open(cfg).readlines()
    print("  cfg: %s (%d 行)" % (cfg, len(lines)))

    print("[2/4] 改写 cfg，mode=%s ..." % args.mode)
    headers = [(i, HDR.match(l).group(2)) for i, l in enumerate(lines) if HDR.match(l)]

    # 先把候选列出来，方便核对
    concat_like = [(i, h) for i, h in headers
                   if 'concat' in h.lower() or h == 'output0' or 'VConcat' in h]
    cv3_like = [(i, h) for i, h in headers if '/model.22/cv3' in h]
    print("  找到 Concat/输出类张量 %d 个，cv3 类 %d 个" % (len(concat_like), len(cv3_like)))
    for i, h in concat_like:
        print("      concat-wise: %s" % h)

    need_cv3 = args.mode in ('cv3', 'both', 'all')
    need_cat = args.mode in ('concat', 'both', 'all')

    # 从后往前改，避免行号漂移
    if need_cv3:
        for i, _ in reversed(cv3_like):
            lines = set_float32(lines, i)
        print("  已把 %d 个 cv3 张量改为 float32" % len(cv3_like))
    if need_cat:
        for i, _ in reversed(concat_like):
            lines = set_float32(lines, i)
        print("  已把 %d 个 Concat/输出张量改为 float32" % len(concat_like))
    if args.mode == 'output':
        i, h = pick_output_name(lines, args.onnx)
        lines = set_float32(lines, i)
        print("  已把输出张量改为 float32: %s (仅 output0，通常不够)" % h)

    newcfg = cfg.replace('.quantization.cfg', '.hybrid.cfg')
    open(newcfg, 'w').writelines(lines)
    print("  新 cfg: %s" % newcfg)

    print("[3/4] hybrid_quantization_step2 ...")
    models = glob.glob('*.model')
    datas = glob.glob('*.data')
    if models and datas:
        print("  step1 产物: %s, %s" % (models[0], datas[0]))
    attempts = []
    if models and datas:
        attempts.append(dict(model_input=models[0], data_input=datas[0],
                             model_quantization_cfg=newcfg))
    attempts.append(dict(model_quantization_cfg=newcfg))
    attempts.append(dict(dataset=txt, model_quantization_cfg=newcfg))
    ret = -1
    for kw in attempts:
        try:
            ret = rknn.hybrid_quantization_step2(**kw)
            print("  step2 采用参数:", sorted(kw.keys()))
            if ret == 0:
                break
        except Exception as e:
            print("  参数组合 %s 不合适: %r" % (sorted(kw.keys()), e))
    if ret != 0:
        print("!! step2 失败 ret=%s" % ret)
        print("   当前目录:", sorted(os.listdir('.'))[:20])
        rknn.release(); sys.exit(1)

    print("[4/4] init_runtime + 验证 ...")
    if rknn.init_runtime() != 0:
        print("!! init_runtime 失败"); rknn.release(); sys.exit(1)
    outs = rknn.inference(inputs=[inp])
    ok = report('hybrid', outs)

    if ok:
        if rknn.export_rknn(args.out) == 0:
            print("\n✅ 导出成功: %s (%.2f MB)" % (args.out, os.path.getsize(args.out) / 1048576.0))
        else:
            print("\n!! 导出失败")
    else:
        print("\n❌ 类别分仍为 0。下一步：把 --mode 换成 both 再跑一次")

    rknn.release()


if __name__ == '__main__':
    main()
