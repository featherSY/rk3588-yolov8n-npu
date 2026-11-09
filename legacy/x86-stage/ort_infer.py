# -*- coding: utf-8 -*-
"""
第三课：手撸 ONNXRuntime 推理流水线（零 ultralytics 内部依赖版）

依赖只有三样：onnxruntime（推理引擎）+ opencv（读图/缩放）+ numpy（数组运算）
后处理（NMS 去重 + 坐标映射）全部自己实现 —— 真实部署本来就该自己写。

运行：python ort_infer.py
"""
import statistics
import time

import cv2
import numpy as np
import onnxruntime as ort

ONNX = "yolov8n.onnx"
IMG = "test.jpg"
IMGSZ = 640
CONF_THRES, IOU_THRES = 0.25, 0.7
WARMUP, RUNS = 10, 100

# COCO 80 类名称（0~79），方便把类别号翻译成人话
COCO_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
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
              'hair drier', 'toothbrush']

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
INPUT_NAME = sess.get_inputs()[0].name


def letterbox(im, new_shape=640, color=(114, 114, 114)):
    """等比缩放 + 居中 padding，返回 (图, 缩放比例, (左右padding, 上下padding))"""
    h, w = im.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape - nh) // 2
    bottom = new_shape - nh - top
    left = (new_shape - nw) // 2
    right = new_shape - nw - left
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def preprocess(path):
    """原图 -> (1,3,640,640) float32 输入"""
    im0 = cv2.imread(path)                      # OpenCV: BGR、HWC
    im, r, pad = letterbox(im0, IMGSZ)
    im = im[:, :, ::-1].transpose(2, 0, 1)      # BGR->RGB，HWC -> CHW
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
    return im[None], im0, r, pad


def xywh2xyxy(x):
    """(中心x, 中心y, 宽, 高) -> (左上x, 左上y, 右下x, 右下y)"""
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def nms(boxes, scores, iou_thres):
    """
    非极大值抑制（手写实现，面试常考）
    boxes: (N,4) xyxy   scores: (N,)
    返回保留下来的框索引
    """
    order = scores.argsort()[::-1]      # 置信度降序
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        # 与剩余框批量算 IoU（向量化，比 Python 循环快得多）
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou <= iou_thres]  # 只保留重叠不严重的
    return keep


def postprocess(pred, r, pad):
    """(1,84,8400) 原始输出 -> 原图坐标系下的 [(类别, 置信度, xyxy)]"""
    p = pred[0].T                        # (84,8400) -> (8400,84)
    boxes_xywh = p[:, :4]                # 前 4 个是框（xywh，相对 640 输入）
    cls_scores = p[:, 4:]                # 后 80 个是各类别分数

    cls_ids = cls_scores.argmax(axis=1)  # 每个框取分数最高的类别
    confs = cls_scores.max(axis=1)

    mask = confs > CONF_THRES            # 置信度阈值过滤
    boxes_xywh, confs, cls_ids = boxes_xywh[mask], confs[mask], cls_ids[mask]
    if len(boxes_xywh) == 0:
        return []

    boxes = xywh2xyxy(boxes_xywh)

    padw, padh = pad
    out = []
    for c in np.unique(cls_ids):         # 逐类别做 NMS
        idx = np.where(cls_ids == c)[0]
        b, s = boxes[idx], confs[idx]
        for k in nms(b, s, IOU_THRES):
            # 减 padding、除缩放比例 -> 映射回原图像素坐标系
            x1 = (b[k, 0] - padw) / r
            y1 = (b[k, 1] - padh) / r
            x2 = (b[k, 2] - padw) / r
            y2 = (b[k, 3] - padh) / r
            out.append((int(c), round(float(s[k]), 3),
                        [round(float(x1), 1), round(float(y1), 1),
                         round(float(x2), 1), round(float(y2), 1)]))
    out.sort(key=lambda t: -t[1])        # 按置信度排序输出
    return out


# ↓↓↓ 只有直接运行本文件时才执行；被 import 时不会跑（Python 的 __main__ 惯用法）↓↓↓
if __name__ == "__main__":
    x, im0, r, pad = preprocess(IMG)
    print(f"原图: {im0.shape[1]}x{im0.shape[0]} (宽x高)   网络输入: {tuple(x.shape)}")

    def run():
        t0 = time.perf_counter()
        out = sess.run(None, {INPUT_NAME: x})[0]     # 纯网络前向
        t1 = time.perf_counter()
        boxes = postprocess(out, r, pad)             # NMS + 坐标映射
        t2 = time.perf_counter()
        return (t1 - t0) * 1000, (t2 - t1) * 1000, boxes

    for _ in range(WARMUP):
        run()

    fwd, post, total = [], [], []
    for _ in range(RUNS):
        f, p, boxes = run()
        fwd.append(f)
        post.append(p)
        total.append(f + p)

    print("\n" + "=" * 62)
    print("手撸 ORT 流水线 —— 性能拆解（100 帧统计）")
    print("=" * 62)
    print(f"  网络前向 (ORT)     : {statistics.mean(fwd):6.1f} ms")
    print(f"  后处理 (NMS+映射)  : {statistics.mean(post):6.1f} ms")
    print(f"  " + "-" * 42)
    print(f"  合计               : {statistics.mean(total):6.1f} ms  "
          f"({1000 / statistics.mean(total):.1f} FPS)")
    print(f"\n  横评参照：ultralytics 包装 ONNX 130.3 ms / PyTorch 端到端 31.7 ms")
    print("=" * 62)

    print(f"\n检测结果（{len(boxes)} 个框）：")
    for cls, conf, xyxy in boxes:
        print(f"  {COCO_NAMES[cls]:<12s} conf={conf:.3f}  xyxy={xyxy}")
