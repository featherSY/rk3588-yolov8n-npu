#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P0 口径对齐 · x86 侧补测（与板端 video_demo.py 同口径）

背景：报告 §3「相对 x86 FP32 (26.5 ms)」三行全是「—」，因为
    x86 26.5 ms = 端到端，板端 33.30 ms = 纯前向 rknn.inference()
两者不可直接比较。板端 video_demo.py 已同时打印这两个口径，
本脚本在 x86 ONNXRuntime 上按完全相同的分段测一遍，让对比列能填实。

分段（与 video_demo.py 主循环一一对应）：
    t0 -> t1  前处理：letterbox + BGR2RGB + NCHW float32 /255
    t1 -> t2  纯前向：sess.run()
    t2 -> t3  后处理：解包 + conf 过滤 + NMS + 坐标映射
    t0 -> t3  端到端：含全部

用法（WSL，零占位符）：
    source ~/rknn-env/bin/activate
    python3 /mnt/d/01_STUDY/rk3588/scripts/bench_alignment.py \
        --model /home/feather/projects/rk3588-yolo/models/yolov8n.onnx \
        --img-dir /mnt/d/01_STUDY/rk3588/calib/calib200 \
        --warmup 10 --repeat 200 --threads 1

依赖：与 eval_map.py 放同一目录（复用它的 letterbox / postprocess / map_back）。
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from eval_map import letterbox, postprocess, map_back
except ImportError:
    print('!! 同目录找不到 eval_map.py；本脚本需与 eval_map.py 放在一起')
    sys.exit(1)

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def report(arr):
    """与 video_demo.py 的 report() 完全一致：平均 / P50 / P95 / FPS"""
    if not arr:
        return None
    a = np.asarray(arr)
    return (float(a.mean()), float(np.percentile(a, 50)),
            float(np.percentile(a, 95)), float(1000.0 / a.mean()))


def main():
    ap = argparse.ArgumentParser(description='P0 口径对齐：x86 ORT 分段延迟补测')
    ap.add_argument('--model', required=True, help='yolov8n.onnx (FP32)')
    ap.add_argument('--img-dir', required=True, help='图片目录，按顺序循环取帧')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--repeat', type=int, default=200)
    ap.add_argument('--threads', type=int, default=1,
                    help='固定 ORT intra_op 线程数，避免 CPU 频率/调度波动')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--nms', type=float, default=0.45)
    ap.add_argument('--size', type=int, default=640)
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print('!! 当前环境没有 onnxruntime；换一个环境试：')
        print('   source ~/venv-ai/bin/activate   或   source ~/rknn-env/bin/activate')
        sys.exit(1)

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.model, sess_options=so,
                                providers=['CPUExecutionProvider'])
    iname = sess.get_inputs()[0].name

    files = sorted(p for p in glob.glob(os.path.join(args.img_dir, '*'))
                   if p.lower().endswith(IMG_EXTS))
    if not files:
        print('!! 图片目录为空或路径不存在:', args.img_dir)
        sys.exit(1)

    print('=' * 60)
    print(' 模型 : %s (%.2f MB)' % (args.model, os.path.getsize(args.model) / 1e6))
    print(' ORT  : %s | intra_op=%d | inter_op=1 | CPUExecutionProvider'
          % (ort.__version__, args.threads))
    print(' 图片 : %d 张 | warmup %d + 测量 %d 帧 | conf %.2f nms %.2f'
          % (len(files), args.warmup, args.repeat, args.conf, args.nms))
    print('=' * 60)

    def one(path):
        img = cv2.imread(path)
        if img is None:
            return None
        t0 = time.perf_counter()
        padded, scale, pad = letterbox(img, args.size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        x = np.ascontiguousarray(rgb, dtype=np.float32)[None] / 255.0
        t1 = time.perf_counter()
        out = sess.run(None, {iname: x})[0][0]          # [84, 8400]
        t2 = time.perf_counter()
        boxes, scores = out[:4].T, out[4:84].T
        det = postprocess(boxes, scores, args.conf, args.nms)
        det = map_back(det, scale, pad, img.shape[1], img.shape[0])
        t3 = time.perf_counter()
        return ((t1 - t0) * 1e3, (t2 - t1) * 1e3,
                (t3 - t2) * 1e3, (t3 - t0) * 1e3, len(det))

    for _ in range(args.warmup):
        one(files[0])
    print('   预热 %d 帧完成' % args.warmup)

    rec = []
    for i in range(args.repeat):
        r = one(files[i % len(files)])
        if r:
            rec.append(r)
        if (i + 1) % 50 == 0:
            print('   %4d 帧 | 前向 %6.2f ms | 端到端 %6.2f ms'
                  % (i + 1, np.mean([q[1] for q in rec]),
                     np.mean([q[3] for q in rec])))

    if not rec:
        print('!! 没有有效帧，检查图片是否可读')
        sys.exit(1)

    print('-' * 60)
    names = ('前处理 letterbox+归一化', '纯前向 sess.run()',
             '后处理 解包+NMS+映射', '端到端 含全部')
    for k, name in enumerate(names):
        r = report([q[k] for q in rec])
        print('   %-26s 平均 %6.2f ms | P50 %6.2f | P95 %6.2f | %5.1f FPS'
              % ((name,) + r))
    print('-' * 60)
    print('   每图平均检出框数: %.1f' % np.mean([q[4] for q in rec]))
    print('   对比用法：')
    print('     · 「纯前向」与板端 33.30 ms / video_demo 纯前向对齐')
    print('     · 「端到端」与板端 video_demo 端到端对齐，替换报告 26.5 ms 基线')
    print('   注：x86 输出为单张量 [1,84,8400]，板端为 2out；解包开销差异在报告注明')


if __name__ == '__main__':
    main()
