#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""convert_w16.py —— 把已拆好的 2out ONNX 转成 w16a16（全 16 位）rknn

为什么单独一个脚本：
  split_outputs.py 里 quantized_dtype hard-code 成 'w8a8'，
  这里只做「换精度重转」，不重复拆图逻辑，避免动到已验证的代码。

用法（在 WSL 的 rknn-env 里跑）：
  python convert_w16.py \
    --onnx models/yolov8n_2out_bs8.onnx \
    --dataset /mnt/d/01_STUDY/rk3588/calib/calib200 --num 200 \
    --img /mnt/d/01_STUDY/rk3588/calib/calib200/000000000139.jpg \
    --dtype w16a16i --out models/yolov8n_2out_w16.rknn

注意：
  - w16a16 不需要每层做 KL/MMSE 搜索，校准集只为确定动态范围，
    所以 build 比 w8a8 快很多，但模型体积约 2x。
  - 评估时不需要 --box-scale（w16 精度足够，但我们这版 ONNX 里
    仍然带着 Div(8)，所以评估端照样要 --box-scale 8）。
"""

import os
import sys
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


def health(outs, box_div=1.0):
    """体检：按通道数挑 boxes / scores"""
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

    if box_div != 1.0:
        boxes = boxes * box_div
    cmax = scores.max(axis=0)
    print("  boxes : mean=%.2f max=%.2f (已乘回 %.0f)" % (boxes.mean(), boxes.max(), box_div))
    print("  scores: max=%.4f mean=%.4f min=%.4f"
          % (cmax.max(), cmax.mean(), cmax.min()))
    ok = cmax.max() > 1e-6 and boxes.max() > 100   # 乘回后应接近 640
    print("  判定: %s" % ("✅ 正常" if ok else "❌ 异常"))
    return ok


def main():
    import numpy as np
    import cv2
    from rknn.api import RKNN

    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True, help='已拆好的 2out onnx')
    ap.add_argument('--dataset', required=True, help='校准图目录（必须是绝对路径）')
    ap.add_argument('--num', type=int, default=200)
    ap.add_argument('--img', required=True, help='体检用单图')
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--dtype', default='w16a16i',
                    choices=['w16a16i', 'w16a16i_dfp', 'w8a16', 'w8a8'],
                    help='w16a16i=权重+激活都int16(方案C); '
                         'w8a16=权重int8激活int16; w8a8=全int8')
    ap.add_argument('--box-div', type=float, default=8.0,
                    help='ONNX 里 boxes 被除的常数，体检时乘回来')
    ap.add_argument('--core', default='012')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    for f in (args.onnx, args.img):
        if not os.path.isfile(f):
            print("!! 不存在:", f); sys.exit(1)
    if not os.path.isdir(args.dataset):
        print("!! 校准目录不存在:", args.dataset); sys.exit(1)
    if not os.path.isabs(args.dataset):
        print("!! 警告: dataset 不是绝对路径，rknn 会按 $HOME 解析，可能读不到图")

    files = collect_images(args.dataset, args.num)
    if not files:
        print("!! 校准目录没图"); sys.exit(1)
    txt = os.path.expanduser('~/calib_list_w16.txt')
    with open(txt, 'w') as f:
        f.write('\n'.join(os.path.abspath(p) for p in files) + '\n')   # 写绝对路径
    print("校准集: %d 张  -> %s" % (len(files), txt))

    print("[1/2] 转 %s ..." % args.dtype)
    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform='rk3588',
                quantized_dtype=args.dtype,
                quantized_algorithm='normal')
    if rknn.load_onnx(model=args.onnx) != 0:
        print("!! load_onnx 失败"); rknn.release(); sys.exit(1)
    if rknn.build(do_quantization=True, dataset=txt, rknn_batch_size=1) != 0:
        print("!! build 失败"); rknn.release(); sys.exit(1)

    # 先导出：build 成功就有模型了，体检失败不该丢掉成果
    if rknn.export_rknn(args.out) == 0:
        print("\n✅ 导出成功: %s (%.2f MB)"
              % (args.out, os.path.getsize(args.out) / 1048576.0))
    else:
        print("\n!! 导出失败"); rknn.release(); sys.exit(1)

    print("\n[2/2] 模拟器体检（失败不影响上面的导出）...")
    try:
        # 模拟器不接受 core_mask，连 None 都不行，必须完全不传
        if rknn.init_runtime() != 0:
            raise RuntimeError('init_runtime 返回非 0')
        img = cv2.imread(args.img)
        if img is None:
            raise RuntimeError('读不到体检图: %s' % args.img)
        inp = cv2.cvtColor(letterbox(img, args.size),
                           cv2.COLOR_BGR2RGB)[None].astype(np.uint8)
        outs = rknn.inference(inputs=[inp])
        health(outs, args.box_div)
    except Exception as e:
        print("[warn] 体检跳过: %s" % e)
        print("[warn] 模型已导出，精度去板端 eval 验证即可")
    rknn.release()


if __name__ == '__main__':
    main()
