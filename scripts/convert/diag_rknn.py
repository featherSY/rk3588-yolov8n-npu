# -*- coding: utf-8 -*-
"""
在 PC（x86 rknn-toolkit2 模拟器）上「转一个配置、立刻验证一遍」，自动扫多个量化配置。

注意：RKNN 模拟器不支持 load_rknn 直接推理，必须在同一进程里
      load_onnx -> build -> init_runtime -> inference，所以每个配置都重转一遍。

用法（重点看三种量化算法的差异）：
    source ~/rknn-env/bin/activate
    python diag_rknn.py --onnx ~/yolov8n.onnx --dataset ~/calib \
                        --img ~/calib/000000000009.jpg --size 640 \
                        --dtypes fp16 int8 \
                        --algos normal mmse kl_divergence --save-dir ~/rknn_out

w8a16 不支持 RK3588（只有 RK3566/3568 支持），已移除。

输出末尾是一张表，看「类别max」和「判定」两列就知道哪个配置能用：

    配置              类别max      框mean  anchor>0.25  判定
    fp16             0.8643      198.14         1234  ✅ 正常
    int8/normal      0.0000      200.15            0  ❌ 分类分支恒 0
    int8/mmse        0.8521      199.02         1100  ✅ 正常
"""

import os
import sys
import argparse
import numpy as np
import cv2
from rknn.api import RKNN

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# 需要走 build(do_quantization=True) 的配置
# 注意 w8a16 只有 RK3566/3568 支持，RK3588 不支持，所以这里没有
QUANT_DTYPES = ('int8', 'w16a16i', 'w16a16i_dfp', 'w4a16')


def collect_images(d):
    return [os.path.join(d, n) for n in sorted(os.listdir(d))
            if os.path.isfile(os.path.join(d, n)) and n.lower().endswith(IMG_EXTS)]


def letterbox(img, size=640, color=(114, 114, 114)):
    """YOLOv8 标准 letterbox：等比缩放 + 灰边填充"""
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nw, nh = int(round(w * s)), int(round(h * s))
    img2 = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    left = (size - nw) // 2
    img2 = cv2.copyMakeBorder(img2, top, size - nh - top, left, size - nw - left,
                              cv2.BORDER_CONSTANT, value=color)
    return img2


def unfold(o):
    """把 output 整理成 [84, 8400]：行 0~3 框回归，行 4~83 类别 logits"""
    a = np.array(o)
    if a.ndim != 3:
        return None, a.shape
    if a.shape[1] in (84, 85) and a.shape[2] > 100:
        return a[0], a.shape
    if a.shape[2] in (84, 85) and a.shape[1] > 100:
        return a[0].T, a.shape
    return None, a.shape


def run_one(cfg_name, args, inp, dataset_txt):
    """跑一个完整配置：config -> load_onnx -> build -> inference，返回统计 dict"""
    dtype = cfg_name.split('/')[0]
    algo = cfg_name.split('/')[1] if '/' in cfg_name else args.algos[0]
    do_quant = dtype in ('int8', 'w16a16i', 'w16a16i_dfp', 'w4a16')

    print("\n" + "-" * 60)
    print(" 配置: %s" % cfg_name)
    print("-" * 60)

    rknn = RKNN(verbose=False)
    try:
        cfg = dict(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                   target_platform='rk3588', optimization_level=args.opt)
        if do_quant:
            cfg['quantized_dtype'] = {'int8': 'w8a8'}.get(dtype, dtype)
            cfg['quantized_algorithm'] = algo
        print("  config:", cfg)

        if rknn.config(**cfg) != 0:
            print("  !! config 失败")
            return None
        if rknn.load_onnx(model=args.onnx) != 0:
            print("  !! load_onnx 失败")
            return None
        if rknn.build(do_quantization=do_quant,
                      dataset=dataset_txt if do_quant else None,
                      rknn_batch_size=1) != 0:
            print("  !! build 失败")
            return None

        # build 完必须 init_runtime 才能推理；不传 target = PC 模拟器
        if rknn.init_runtime() != 0:
            print("  !! init_runtime 失败")
            return None

        outs = rknn.inference(inputs=[inp])
        if not outs:
            print("  !! inference 无输出")
            return None

        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            p = os.path.join(args.save_dir, 'yolov8n_%s.rknn' % cfg_name.replace('/', '_'))
            if rknn.export_rknn(p) == 0:
                print("  已导出: %s (%.2f MB)" % (p, os.path.getsize(p) / 1048576.0))
    except Exception as e:
        print("  !! 异常: %r" % (e,))
        return None
    finally:
        pass

    x, shape = unfold(outs[0])
    rknn.release()

    if x is None:
        print("  !! 认不出输出布局 shape=%s" % (shape,))
        return None

    box = x[:4]
    cls = x[4:84]
    cmax = cls.max(axis=0)
    prob = 1.0 / (1.0 + np.exp(-cmax))

    r = dict(
        cfg=cfg_name,
        cls_max=float(cmax.max()),
        cls_mean=float(cmax.mean()),
        box_mean=float(box.mean()),
        box_max=float(box.max()),
        n025=int((prob > 0.25).sum()),
        total=int(prob.size),
    )
    print("  输出 shape=%s dtype=%s" % (shape, np.array(outs[0]).dtype))
    print("  框回归 : mean=%.2f max=%.2f" % (r['box_mean'], r['box_max']))
    print("  类别分 : max=%.4f mean=%.4f" % (r['cls_max'], r['cls_mean']))

    # —— 决定性诊断：区分「权重被压成 0」和「输出张量被量化」——
    zero_rows = np.where(~x.any(axis=1))[0]
    print("  全 0 行: %d / %d" % (len(zero_rows), x.shape[0]))
    if len(zero_rows):
        rows = [int(i) for i in zero_rows]
        print("     行索引: %s%s   [4~83 = 分类分支]"
              % (', '.join(str(i) for i in rows[:8]),
                 ' ... %d' % rows[-1] if len(rows) > 8 else ''))
    b = x[:4].reshape(-1)
    uniq = int(np.unique(np.round(b, 4)).size)
    r['box_uniq'] = uniq
    print("  框区域 unique 值: %d / %d" % (uniq, b.size))
    if uniq < b.size * 0.1:
        print("     -> 框值被量化成粗网格，可能是输出张量 int8 量化导致")
    else:
        print("     -> 框值连续，输出没被粗量化；分类行恒 0 = 分类卷积权重被量化压没")

    print("  sigmoid>0.25 的 anchor: %d / %d" % (r['n025'], r['total']))

    if r['cls_max'] < 1e-6:
        r['verdict'] = '❌ 分类分支恒 0'
    elif r['cls_max'] < 0.5 or r['n025'] == 0:
        r['verdict'] = '⚠️  分数偏低,会漏检'
    else:
        r['verdict'] = '✅ 正常'
    print("  判定: %s" % r['verdict'])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--dataset', default='.', help='校准集目录（int8/w8a16 需要）')
    ap.add_argument('--img', required=True, help='测试图片')
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--dtypes', nargs='+', default=['fp16', 'int8'],
                    choices=['fp32', 'fp16', 'int8', 'w16a16i', 'w16a16i_dfp', 'w4a16'],
                    help='注意：w8a16 仅 RK3566/3568 支持，RK3588 不支持')
    ap.add_argument('--algos', nargs='+', default=['normal'],
                    choices=['normal', 'mmse', 'kl_divergence'])
    ap.add_argument('--opt', type=int, default=3)
    ap.add_argument('--num', type=int, default=200, help='最多用多少张校准图')
    ap.add_argument('--save-dir', default='', help='把每个配置导出的 rknn 存到这个目录')
    args = ap.parse_args()

    for f, tag in ((args.onnx, 'ONNX'), (args.img, '测试图')):
        if not os.path.isfile(f):
            print("!! %s 不存在: %s" % (tag, f))
            sys.exit(1)

    img = cv2.imread(args.img)
    if img is None:
        print("!! 图片读不出来:", args.img)
        sys.exit(1)
    print("测试图: %s  %dx%d" % (args.img, img.shape[1], img.shape[0]))

    lb = letterbox(img, args.size)
    inp = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)[None].astype(np.uint8)  # NHWC 1x640x640x3

    txt = None
    if set(args.dtypes) & set(QUANT_DTYPES):
        files = collect_images(args.dataset)[:args.num]
        if not files:
            print("!! 校准目录没图:", args.dataset)
            sys.exit(1)
        print("校准集: %d 张" % len(files))
        txt = os.path.expanduser('~/calib_list.txt')
        with open(txt, 'w') as f:
            f.write('\n'.join(files) + '\n')

    combos = []
    for d in args.dtypes:
        if d in QUANT_DTYPES:
            for a in args.algos:
                combos.append('%s/%s' % (d, a))
        else:
            combos.append(d)

    print("待测配置: %s" % ', '.join(combos))
    print("（每个配置都要重新 build，量化配置较慢，耐心等）")

    results = []
    for c in combos:
        r = run_one(c, args, inp, txt)
        if r:
            results.append(r)

    print("\n" + "=" * 76)
    print(" 汇总")
    print("=" * 76)
    print("%-16s %10s %10s %12s  %s"
          % ("配置", "类别max", "框mean", "anchor>0.25", "判定"))
    print("-" * 76)
    for r in results:
        print("%-16s %10.4f %10.2f %12d  %s"
              % (r['cfg'], r['cls_max'], r['box_mean'], r['n025'], r['verdict']))
    print("-" * 76)
    ok = [r for r in results if r['verdict'].startswith('✅')]
    fast = [r for r in ok if r['cfg'].startswith('int8')]
    if fast:
        print(" ✅ 且最快的: %s  -> 优先传板子" % fast[0]['cfg'])
    elif ok:
        print(" 可用的: %s" % ', '.join(x['cfg'] for x in ok))
    else:
        print(" 全灭。下一步: --algos mmse kl_divergence，或 --opt 2 再跑一轮")
    print("=" * 76)


if __name__ == '__main__':
    main()
