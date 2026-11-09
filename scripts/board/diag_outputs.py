#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板端输出体检：用 1 张图判断「输出 FP16 是否真的去掉了量化台阶」。

用法：
  sudo LD_LIBRARY_PATH=/home/zsy/rk3588-yolo/lib python3 diag_outputs.py \
      --model models/yolov8n_out_fp16.rknn \
      --img /path/to/val2017/000000000009.jpg

  多个模型可一次 A/B 对比：
  --model models/yolov8n_2out_int8.rknn models/yolov8n_out_fp16.rknn

判读（看 out 里标 <-- scores 那行）：
  分数 uniq 值   INT8 ≈ 147（1/255 台阶）；FP16 生效应 > 1000，通常上万
  台阶吻合度     越接近 1.0 越说明值仍贴在 1/255 网格上
  boxes max      ≈ 79  -> 模型内插了 Div(8)，评估必须带 --box-scale 8
                 ≈ 640 -> 无需缩放，别带 --box-scale
"""

import os
import sys
import argparse
import numpy as np
import cv2


def letterbox(img, size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nw, nh = int(round(w * s)), int(round(h * s))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    return cv2.copyMakeBorder(im, top, size - nh - top, left, size - nw - left,
                              cv2.BORDER_CONSTANT, value=color)


def stats(a):
    f = a.astype(np.float64).ravel()
    uniq = int(np.unique(np.round(f, 6)).size)
    # 与 1/255 台阶的吻合度：dist 越小越贴网格
    d = np.abs(f * 255 - np.round(f * 255))
    stair = float((d < 0.02).mean())
    return float(f.min()), float(f.max()), float(f.mean()), uniq, stair


def run(model, img_rgb):
    from rknnlite.api import RKNNLite
    r = RKNNLite(verbose=False)
    if r.load_rknn(model) != 0:
        print("  !! load_rknn 失败: %s" % model)
        return
    if r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        print("  !! init_runtime 失败（NPU 打不开）")
        return
    outs = r.inference(inputs=[np.expand_dims(img_rgb, 0)])
    if not isinstance(outs, (list, tuple)):
        outs = [outs]

    print("  模型: %s (%.2f MB)" % (model, os.path.getsize(model) / 1048576.0))
    print("  输出个数: %d" % len(outs))
    for i, o in enumerate(outs):
        a = np.asarray(o)
        if a.ndim < 2:
            print("  out[%d] shape=%s dtype=%s (跳过)" % (i, a.shape, a.dtype))
            continue
        lo, hi, mean, uniq, stair = stats(a)
        c = a.shape[1] if a.ndim == 3 else -1
        tag = '  <-- boxes' if c == 4 else ('  <-- scores' if c == 80 else '')
        print("  out[%d] shape=%-18s dtype=%-8s min=%9.4f max=%9.4f "
              "mean=%9.4f uniq=%7d 台阶吻合=%.3f%s"
              % (i, str(a.shape), a.dtype, lo, hi, mean, uniq, stair, tag))
        if c == 4:
            print("       -> %s" % (
                'boxes 被压缩过（≈79），评估必须带 --box-scale 8'
                if hi < 120 else 'boxes 是原量程（≈640），别带 --box-scale'))
        if c == 80:
            print("       -> %s" % (
                'X 仍是 1/255 量化台阶，输出 FP16 没生效'
                if uniq < 300 else 'OK 分数已连续，输出 FP16 生效'))
    r.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', nargs='+', required=True, help='一个或多个 .rknn')
    ap.add_argument('--img', required=True, help='任意一张 val2017 图')
    args = ap.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        print("!! 图片读不出来:", args.img)
        sys.exit(1)
    rgb = cv2.cvtColor(letterbox(img), cv2.COLOR_BGR2RGB)

    for m in args.model:
        print("=" * 86)
        run(m, rgb)
    print("=" * 86)


if __name__ == '__main__':
    main()
