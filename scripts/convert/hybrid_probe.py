# -*- coding: utf-8 -*-
"""
混合量化「探针」：只跑到 hybrid_quantization_step1，把 RKNN 自动生成的
量化配置文件打印出来。

目的：拿到 2.3.2 版本的 cfg 格式，下一步才能写 step2 去精准地把
      model.22.cv3.* （分类分支）单独保持 fp16，其余保持 int8。

用法：
    mkdir -p ~/hyb && cd ~/hyb
    python ~/hybrid_probe.py --onnx ~/yolov8n.onnx --dataset ~/calib

跑完把 ~/hyb 目录下那个 *.quantization.cfg 的内容贴出来即可。
"""

import os
import sys
import glob
import argparse
from rknn.api import RKNN

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def collect_images(d, num):
    return [os.path.join(d, n) for n in sorted(os.listdir(d))
            if os.path.isfile(os.path.join(d, n)) and n.lower().endswith(IMG_EXTS)][:num]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--dataset', default='.')
    ap.add_argument('--num', type=int, default=200)
    ap.add_argument('--maxlines', type=int, default=120, help='cfg 最多打印多少行')
    args = ap.parse_args()

    if not os.path.isfile(args.onnx):
        print("!! ONNX 不存在:", args.onnx)
        sys.exit(1)

    files = collect_images(args.dataset, args.num)
    if not files:
        print("!! 校准目录没图:", args.dataset)
        sys.exit(1)
    txt = os.path.abspath('calib_list.txt')
    with open(txt, 'w') as f:
        f.write('\n'.join(files) + '\n')
    print("校准集: %d 张 -> %s" % (len(files), txt))

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform='rk3588')
    if rknn.load_onnx(model=args.onnx) != 0:
        print("!! load_onnx 失败")
        sys.exit(1)

    print("[1/1] hybrid_quantization_step1 ...（会跑一遍量化，约 1 分钟）")
    ret = rknn.hybrid_quantization_step1(dataset=txt)
    if ret != 0:
        print("!! step1 失败 ret=%s" % ret)
        rknn.release()
        sys.exit(1)

    cfgs = glob.glob('*.quantization.cfg')
    if not cfgs:
        print("!! 没找到生成的 .quantization.cfg，当前目录内容:")
        for x in sorted(os.listdir('.')):
            print("   ", x)
        rknn.release()
        sys.exit(1)

    for c in cfgs:
        print("\n" + "=" * 66)
        print(" 配置文件: %s   (%d 字节)" % (os.path.abspath(c), os.path.getsize(c)))
        print("=" * 66)
        with open(c) as f:
            lines = f.readlines()
        print("总行数: %d" % len(lines))
        print("-" * 66)
        for i, ln in enumerate(lines[:args.maxlines]):
            print("%4d| %s" % (i + 1, ln.rstrip('\n')))
        if len(lines) > args.maxlines:
            print("... 还有 %d 行未显示" % (len(lines) - args.maxlines))
        print("-" * 66)
        # 顺便看看分类分支的层名在里面长什么样
        hits = [(i + 1, ln.rstrip()) for i, ln in enumerate(lines) if 'cv3' in ln]
        print("含 cv3 的行: %d 条" % len(hits))
        for i, ln in hits[:20]:
            print("   %5d| %s" % (i, ln))

    print("\n把上面输出的 cfg 内容（尤其含 cv3 的片段）发给我。")
    rknn.release()


if __name__ == '__main__':
    main()
