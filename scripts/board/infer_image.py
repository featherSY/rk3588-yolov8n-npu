# -*- coding: utf-8 -*-
"""
RK3588 板端 YOLOv8 推理脚本 —— Stage 2

用法：
    python3 infer_image.py --model models/yolov8n_fp16.rknn --img test.jpg
    python3 infer_image.py --model models/yolov8n_fp16.rknn --img test.jpg --repeat 100

说明：
  - 兼容两种模型输出：
      fp16     单输出 [1,84,8400]（4 框 + 80 类）
      int8 拆输出  [1,84,8400] + [1,80,8400](scores) + [1,4,8400](boxes)
    按通道数自动挑，不依赖输出顺序
  - 类别分在 ONNX 里已经过 sigmoid，值域 [0,1]，此处不再套 sigmoid
  - 预处理用 letterbox（保持长宽比补灰边），后处理把坐标映射回原图
"""

import os
import sys
import time
import argparse

import numpy as np
import cv2

from rknnlite.api import RKNNLite

# 板端结果图默认存到 <项目根>/results/（本脚本在 scripts/ 下，取上一级）。
# PC 侧评测结果一直用显式的 results/xxx.json，不在这里管。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'results'))

# COCO 80 类
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
]


def letterbox(img, size=640, pad_color=(114, 114, 114)):
    """保持长宽比缩放并补边，返回 (pad后的图, 缩放比例, (左pad, 上pad))"""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=pad_color)
    return out, scale, (left, top)


def infer_once(rknn, img_u8, layout='auto'):
    """
    img_u8: (640,640,3) uint8 RGB
    RKNN 对 NCHW 的 ONNX 模型：通常接受 NHWC 输入（内部自动转置），
    但不同版本行为不一，这里两种都试。
    """
    cands = [
        (np.expand_dims(img_u8, 0), 'NHWC (1,640,640,3)'),
        (np.transpose(img_u8, (2, 0, 1))[None], 'NCHW (1,3,640,640)'),
    ]
    if layout == 'nhwc':
        cands = cands[:1]
    elif layout == 'nchw':
        cands = cands[1:]

    last_err = None
    for arr, tag in cands:
        try:
            out = rknn.inference(inputs=[arr])
            return out, tag
        except Exception as e:
            last_err = e
    raise RuntimeError('推理失败: %s' % last_err)


def apply_box_scale(boxes, spec):
    """boxes 分支若插过 Div(8) 压缩量程，必须乘回 8 才是 640 图上的像素坐标。

    eval_map.py 用 --box-scale 8 显式还原，所以 mAP 正常；本脚本以前没做，
    导致 0~80 的坐标被 letterbox 的 top pad 减成负数、clamp 到 0，
    所有框变成贴顶的一条线（y1=y2=0），结果图看不到框。

    spec='auto'：按量程自动判断（max<120 视为压缩过）。
    """
    if boxes is None:
        return boxes, 1.0, 'n/a'
    hi = float(boxes.max())
    if spec == 'auto':
        scale = 8.0 if hi < 120 else 1.0
    else:
        try:
            scale = float(spec)
        except ValueError:
            print('!! --box-scale 只能是数字或 auto')
            sys.exit(1)
    note = '%.0f (boxes max=%.1f, %s)' % (
        scale, hi, '自动判定' if spec == 'auto' else '手动指定')
    return boxes * scale, scale, note


def unpack_outputs(out):
    """
    把模型输出整理成 (boxes [8400,4], scores [8400,80])，两类模型都兼容：
      - 单输出 [1,84,8400]：前 4 行是框，后 80 行是类别分（fp16 模型）
      - 拆开的双输出：另有 [1,4,8400] 和 [1,80,8400] 两个独立张量，
        优先用它们 —— 量程各自独立，INT8 下精度才有保障

    注意：输出顺序和 ONNX 里声明的顺序不一定一致，必须按通道数挑。
    """
    arrs = list(out) if isinstance(out, (list, tuple)) else [out]
    arrs = [np.asarray(a) for a in arrs]
    boxes = scores = None
    for a in arrs:
        if a.ndim != 3:
            continue
        c = a.shape[1]
        if c == 4 and boxes is None:
            boxes = a[0].T
        elif c == 80 and scores is None:
            scores = a[0].T
    if boxes is None or scores is None:      # 回退：单输出合并张量
        for a in arrs:
            if a.ndim == 3 and a.shape[1] in (84, 85):
                x = a[0]
                boxes = x[:4].T
                scores = x[4:84].T
                break
    return boxes, scores


def postprocess(boxes_cxcywh, cls_scores, conf_thres=0.25, nms_thres=0.45):
    """
    boxes_cxcywh: [8400,4] cx,cy,w,h（640 letterbox 图上的像素）
    cls_scores  : [8400,80] 已 sigmoid，值域 [0,1]
    返回 [x1,y1,x2,y2,score,class_id] 列表
    """
    cls_ids = np.argmax(cls_scores, axis=1)
    confs = np.max(cls_scores, axis=1)

    mask = confs > conf_thres
    if not np.any(mask):
        return []
    boxes_cxcywh = boxes_cxcywh[mask]
    cls_ids = cls_ids[mask]
    confs = confs[mask]

    cx, cy, bw, bh = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    boxes_xyxy = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()  # cv2 NMSBoxes 要 xywh
    keep = cv2.dnn.NMSBoxes(boxes_xyxy, confs.tolist(), conf_thres, nms_thres)

    out = []
    if len(keep) > 0:
        keep = np.array(keep).flatten()
        for i in keep:
            out.append([x1[i], y1[i], x2[i], y2[i], float(confs[i]), int(cls_ids[i])])
    return out


def map_back(det, scale, pad, orig_w, orig_h):
    """把 640 letterbox 上的坐标映射回原图"""
    left, top = pad
    res = []
    for d in det:
        x1, y1, x2, y2, s, c = d
        x1 = (x1 - left) / scale
        y1 = (y1 - top) / scale
        x2 = (x2 - left) / scale
        y2 = (y2 - top) / scale
        x1 = max(0, min(orig_w, x1)); x2 = max(0, min(orig_w, x2))
        y1 = max(0, min(orig_h, y1)); y2 = max(0, min(orig_h, y2))
        res.append([x1, y1, x2, y2, s, c])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--img', required=True)
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--nms', type=float, default=0.45)
    ap.add_argument('--repeat', type=int, default=1, help='重复推理次数，用于计时')
    ap.add_argument('--warmup', type=int, default=10, help='预热帧数，不计入统计')
    ap.add_argument('--verbose', action='store_true',
                    help='打开 RKNN 详细日志（会严重拖慢推理，仅调试用）')
    ap.add_argument('--out', default=os.path.join(RESULT_DIR, 'result.jpg'),
                    help='结果图，默认落到 results/ 目录')
    ap.add_argument('--core', default='0', choices=['0', '1', '2', '012'])
    ap.add_argument('--layout', default='auto', choices=['auto', 'nhwc', 'nchw'],
                    help='输入布局；模型是 NCHW，但 RKNN 也接受 NHWC（内部会转置）')
    ap.add_argument('--box-scale', default='auto',
                    help='boxes 量程还原系数：auto=按 max<120 自动乘 8；'
                         '也可显式给数字（如 8 / 1）。评估脚本用的 --box-scale 8 与此处等价')
    args = ap.parse_args()

    core_map = {
        '0': RKNNLite.NPU_CORE_0,
        '1': RKNNLite.NPU_CORE_1,
        '2': RKNNLite.NPU_CORE_2,
        '012': RKNNLite.NPU_CORE_0_1_2,
    }

    if not os.path.isfile(args.img):
        print('!! 图片不存在:', args.img); sys.exit(1)

    orig = cv2.imread(args.img)
    oh, ow = orig.shape[:2]
    padded, scale, pad = letterbox(orig, args.size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    print('=' * 52)
    print(' 模型 :', args.model)
    print(' 图片 :', args.img, '%dx%d' % (ow, oh))
    print('=' * 52)

    rknn = RKNNLite(verbose=args.verbose)
    print('[1/3] load_rknn ...')
    if rknn.load_rknn(args.model) != 0:
        print('!! load_rknn 失败'); rknn.release(); sys.exit(1)

    print('[2/3] init_runtime（上 NPU）...')
    if rknn.init_runtime(core_mask=core_map[args.core]) != 0:
        print('!! init_runtime 失败'); rknn.release(); sys.exit(1)
    print('   NPU 初始化成功')

    print('[3/3] inference ...')
    out, layout = infer_once(rknn, rgb, layout=args.layout)
    print('   输入布局:', layout)

    # 输出数值体检：量化出问题时靠这个一眼看出来
    try:
        arrs = list(out) if isinstance(out, (list, tuple)) else [out]
        arrs = [np.asarray(a) for a in arrs]
        print('   输出张量数: %d' % len(arrs))
        for k, a in enumerate(arrs):
            print('     [%d] shape=%-16s min=%.4f max=%.4f mean=%.4f'
                  % (k, str(a.shape), float(a.min()), float(a.max()), float(a.mean())))
        b, s = unpack_outputs(out)
        b, _, bs_note = apply_box_scale(b, args.box_scale)
        print('   box-scale: %s' % bs_note)
        if s is not None:
            print('   类别分(scores): max=%.4f mean=%.4f  >0.25 的 anchor=%d/%d'
                  % (float(s.max()), float(s.mean()), int((s.max(axis=1) > 0.25).sum()), s.shape[0]))
        if b is not None:
            print('   框参数(boxes) : min=%.3f max=%.3f mean=%.3f  (已还原为 640 图像素)'
                  % (float(b.min()), float(b.max()), float(b.mean())))
    except Exception as e:
        print('   输出统计失败:', e)

    lay_key = layout.split()[0].lower()

    # 预热：让 NPU 频率爬上去、页表稳定，不计入统计
    for _ in range(args.warmup):
        infer_once(rknn, rgb, layout=lay_key)

    times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        out, _ = infer_once(rknn, rgb, layout=lay_key)
        times.append((time.perf_counter() - t0) * 1000.0)

    if args.repeat > 1:
        arr = np.array(times)
        print('   推理 %d 次（预热 %d）: 平均 %.2f ms | P50 %.2f | P95 %.2f | 最快 %.2f'
              % (len(arr), args.warmup, arr.mean(), np.percentile(arr, 50),
                 np.percentile(arr, 95), arr.min()))
        print('   等效帧率: %.1f FPS' % (1000.0 / arr.mean()))

    boxes_o, scores_o = unpack_outputs(out)
    if boxes_o is None or scores_o is None:
        print('!! 认不出输出张量'); rknn.release(); sys.exit(1)
    boxes_o, _, _ = apply_box_scale(boxes_o, args.box_scale)   # 关键：乘回 Div(8)
    det = postprocess(boxes_o, scores_o, args.conf, args.nms)
    det = map_back(det, scale, pad, ow, oh)

    print('-' * 52)
    print(' 检测到 %d 个目标:' % len(det))
    for d in det:
        x1, y1, x2, y2, s, c = d
        name = CLASSES[c] if c < len(CLASSES) else str(c)
        print('   %-14s %.2f  [%.0f,%.0f,%.0f,%.0f]' % (name, s, x1, y1, x2, y2))
    print('-' * 52)

    for d in det:
        x1, y1, x2, y2, s, c = d
        cv2.rectangle(orig, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = '%s %.2f' % (CLASSES[c] if c < len(CLASSES) else str(c), s)
        cv2.putText(orig, label, (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, orig)
    print(' 结果图已保存:', args.out)

    rknn.release()


if __name__ == '__main__':
    main()
