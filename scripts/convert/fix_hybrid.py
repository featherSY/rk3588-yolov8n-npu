#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fix_hybrid.py —— 从 cfg 的 quantize_parameters 提取真实张量名，重建 custom_quantize_layers 并跑 step2

为什么需要它：
  第一版 hybrid_quant.py 直接拿 ONNX 的 node name 写进 custom_quantize_layers，
  step2 报：
      E hybrid_quantization_step2: Invalid operands name '/model.22/cv3.0/cv3.0.0/act/Mul'
  原因：RKNN 认的是「输出张量名」，形如
      /model.0/conv/Conv_output_0
  而不是节点名 /model.0/conv/Conv。少一个 _output_0 就查无此人。

  另外 step1 会把图优化一遍（177 层 -> 154 层），act/Mul、act/Sigmoid
  这类 SiLU 内部节点可能被融合掉，逐层硬写更容易踩空。
  => 唯一可靠来源：cfg 自己的 quantize_parameters 小节。

用法：
  cd /mnt/d/01_STUDY/rk3588
  source ~/rknn-env/bin/activate

  # 1) 先看看能匹配到哪些张量名（不改文件、不编译）
  python scripts/fix_hybrid.py --cfg yolov8n_2out_bs8.quantization.cfg --dry-run

  # 2) 确认后真正重建 + 编译 + 导出
  python scripts/fix_hybrid.py --cfg yolov8n_2out_bs8.quantization.cfg \
      --pattern cv3 --out models/yolov8n_2out_hybrid.rknn
"""

import os
import re
import sys
import inspect
import argparse


def parse_quant_keys(lines):
    """抓 quantize_parameters: 之后所有 4 空格缩进的 key（即张量名）"""
    keys = []
    in_block = False
    for ln in lines:
        if re.match(r'^quantize_parameters:\s*$', ln):
            in_block = True
            continue
        if not in_block:
            continue
        if re.match(r'^\S', ln):          # 回到顶层，块结束
            in_block = False
            continue
        m = re.match(r'^ {4}(\S.*?):\s*$', ln)
        if m:
            keys.append(m.group(1))
    return keys


def is_valid_operand(name):
    """custom_quantize_layers 只接受「可量化的激活张量」。

    下面这些在 cfg 里也以张量名出现，但塞进去会被 RKNN 拒：
      - 权重常量：含 'Constant'
      - 网络输入：含 'images'
      - 量化后的变体：结尾 '_int8'
      - 常量操作数：形如 Div_1_2mul_i1 / BoxScaleDiv_2mul_i1（二元 op 的常数输入）
    """
    n = name.lower()
    if 'constant' in n:
        return False
    if 'images' in n:
        return False
    if n.endswith('_int8'):
        return False
    if re.search(r'_\d*(mul|add)_i\d+$', n):
        return False
    return True


def rebuild_cfg(path, keys, dtype='float16'):
    """把 custom_quantize_layers 整个换成 keys，其余原样保留"""
    lines = open(path, encoding='utf-8').read().splitlines()
    out, i, done = [], 0, False
    while i < len(lines):
        ln = lines[i]
        if not done and re.match(r'^custom_quantize_layers:\s*(\{\})?\s*$', ln):
            out.append('custom_quantize_layers:')
            for k in keys:
                out.append("  '%s': '%s'" % (k, dtype))
            i += 1
            while i < len(lines) and re.match(r'^ {2}\S', lines[i]):
                i += 1                    # 丢掉旧的子项
            done = True
            continue
        out.append(ln)
        i += 1
    if not done:
        out = ['custom_quantize_layers:'] + \
              ["  '%s': '%s'" % (k, dtype) for k in keys] + out
    open(path, 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    return done


def run_step2(model_path, data_path, cfg_path, out_path):
    from rknn.api import RKNN
    r = RKNN(verbose=False)
    sig = inspect.signature(r.hybrid_quantization_step2)
    print("[step2] 签名: hybrid_quantization_step2%s" % sig)

    kw = {}
    for p in sig.parameters:
        if p == 'self':
            continue
        low = p.lower()
        if 'model' in low and 'input' in low:
            kw[p] = model_path
        elif 'data' in low and 'input' in low:
            kw[p] = data_path
        elif 'cfg' in low or 'script' in low or 'config' in low:
            kw[p] = cfg_path
    print("[step2] 传参: %s" % {k: os.path.basename(str(v)) for k, v in kw.items()})

    ret = r.hybrid_quantization_step2(**kw)
    print("[step2] 返回值: %s" % ret)
    if ret != 0:
        print("!! step2 失败，cfg 里的层名仍不被接受")
        r.release()
        return False

    if r.export_rknn(out_path) != 0:
        print("!! export 失败")
        r.release()
        return False
    size = os.path.getsize(out_path) / 1048576.0
    print("\n✅ 导出成功: %s (%.2f MB)" % (out_path, size))
    print("   判据: 4.74 MB = 全 INT8 / 5.25 MB ≈ 仅 cv3 FP16 / "
          "6.5~7.5 MB ≈ model.22 head FP16 / 7.5~9 MB ≈ head+输出 FP16")
    r.release()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='*.quantization.cfg')
    ap.add_argument('--pattern', default='cv3',
                    help='只挑名字里含该子串的张量（默认 cv3）。'
                         '支持逗号分隔多条件，如 model.22,yolo_out')
    ap.add_argument('--outputs', action='store_true',
                    help='同时把非 /model 的输出张量（yolo_out_0/1 等）也设成 FP16，'
                         '拿掉输出量化地板——这是救回 mAP 的关键')
    ap.add_argument('--dtype', default='float16', choices=['float16', 'w8a8'])
    ap.add_argument('--model', default=None, help='.model，默认按 cfg 同名推')
    ap.add_argument('--data', default=None, help='.data，默认按 cfg 同名推')
    ap.add_argument('--out', default='models/yolov8n_2out_hybrid.rknn')
    ap.add_argument('--dry-run', action='store_true',
                    help='只打印匹配到的张量名，不改文件不编译')
    args = ap.parse_args()

    if not os.path.isfile(args.cfg):
        print("!! cfg 不存在:", args.cfg); sys.exit(1)

    base = args.cfg[:-len('.quantization.cfg')]
    model_path = args.model or (base + '.model')
    data_path = args.data or (base + '.data')
    for p in (model_path, data_path):
        if not os.path.isfile(p):
            print("!! 缺少 step1 产物:", p); sys.exit(1)

    lines = open(args.cfg, encoding='utf-8').read().splitlines()
    all_keys = parse_quant_keys(lines)
    print("quantize_parameters 里共 %d 个张量" % len(all_keys))

    patterns = [p.strip() for p in args.pattern.split(',') if p.strip()]
    if args.outputs and args.pattern == 'cv3':
        patterns = []   # --outputs 单独用时，不夹带默认 cv3，只改输出张量
    hit = [k for k in all_keys if any(p in k for p in patterns)]
    # 去掉非法 operand：常量 / 输入 / int8 变体 / 常量操作数
    dropped = [k for k in hit if not is_valid_operand(k)]
    hit = [k for k in hit if is_valid_operand(k)]
    print("匹配 %s 的张量: %d 个（剔除 %d 个非法 operand）"
          % (patterns, len(hit), len(dropped)))
    for k in hit:
        print("  ", k)
    if dropped:
        print("  剔除(常量/输入/int8/操作数):")
        for k in dropped:
            print("    -", k)

    if args.outputs:
        outs = [k for k in all_keys if not k.startswith('/model')]
        added = [o for o in outs if o not in hit and is_valid_operand(o)]
        hit = hit + added
        print("\n[--outputs] 额外加入 %d 个输出张量（已剔除非法）:" % len(added))
        for o in added:
            print("  +", o)
        if not added:
            print("  （未找到合法的输出张量，请检查 cfg）")

    if not hit:
        print("\n!! 没匹配到。含 model.22 的张量如下（挑关键字重试）：")
        cand = [k for k in all_keys if 'model.22' in k]
        print("   model.22 相关共 %d 个，前 40 个:" % len(cand))
        for k in cand[:40]:
            print("  ", k)
        sys.exit(1)

    # 顺便验证：这些名字是否真在 quantize_parameters 里（一定是，因为就从那儿来的）
    print("\n对比：第一版用的节点名（错的） vs 现在用的张量名（对的）")
    print("   错: /model.22/cv3.0/cv3.0.0/conv/Conv")
    print("   对: %s" % hit[0])

    if args.dry_run:
        print("\n--dry-run：未修改文件。去掉 --dry-run 执行重建+编译。")
        print("   想拿掉输出量化地板时加：--outputs")
        return

    ok = rebuild_cfg(args.cfg, hit, args.dtype)
    print("\n[cfg] custom_quantize_layers 已重建（%s）" % ("替换原块" if ok else "插到文件头"))
    head = open(args.cfg, encoding='utf-8').read().splitlines()[:6]
    for ln in head:
        print("   ", ln)

    run_step2(model_path, data_path, args.cfg, args.out)


if __name__ == '__main__':
    main()
