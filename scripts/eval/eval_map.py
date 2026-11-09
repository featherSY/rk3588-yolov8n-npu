#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8n 在 COCO val2017 上的 mAP 评估

同一个脚本跑两种后端，保证预处理/后处理口径完全一致：
    ort   : x86 ONNX Runtime（FP32 基线）
    rknn  : RK3588 板端 rknnlite（INT8 / FP16）

用法：
    # x86 FP32 基线
    python3 eval_map.py --backend ort --model yolov8n.onnx \
        --img-dir /path/to/val2017 --ann /path/to/instances_val2017.json \
        --limit 500 --out preds_ort.json

    # RK3588 INT8（板端）
    sudo LD_LIBRARY_PATH=/home/zsy/rk3588-yolo/lib python3 eval_map.py \
        --backend rknn --model models/yolov8n_2out_int8.rknn \
        --img-dir /path/to/val2017 --ann /path/to/instances_val2017.json \
        --limit 500 --out preds_int8.json

要点：
  - 两边必须跑同一批图（默认按文件名排序取前 N 张），否则 mAP 不可比
  - 类别分在 ONNX 里已经过 sigmoid，此处不再套 sigmoid
  - COCO 80 类的 category_id 不是 0~79，需要 COCO_IDS 映射
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import cv2

# COCO 80 类在官方标注里的 category_id（顺序对应 YOLO 的 0~79）
COCO_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


# ---------------------------------------------------------------- 预处理

def letterbox(img, size=640, pad_color=(114, 114, 114)):
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


# ---------------------------------------------------------------- 后处理

def sigmoid(x):
    """数值稳定的 sigmoid，用于 logits -> 概率（float64 计算避免溢出）"""
    x = np.asarray(x, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))).astype(np.float32)


def postprocess(boxes_cxcywh, cls_scores, conf_thres=0.25, nms_thres=0.45):
    """返回 [x1,y1,x2,y2,score,cls_id]，坐标在 640 letterbox 图上"""
    cls_ids = np.argmax(cls_scores, axis=1)
    confs = cls_scores.max(axis=1)

    mask = confs > conf_thres
    if not np.any(mask):
        return []
    boxes_cxcywh = boxes_cxcywh[mask]
    cls_ids = cls_ids[mask]
    confs = confs[mask]

    cx, cy, bw, bh = (boxes_cxcywh[:, 0], boxes_cxcywh[:, 1],
                      boxes_cxcywh[:, 2], boxes_cxcywh[:, 3])
    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy + bh / 2

    xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    keep = cv2.dnn.NMSBoxes(xywh, confs.tolist(), conf_thres, nms_thres)
    out = []
    if len(keep) > 0:
        for i in np.array(keep).flatten():
            out.append([float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                        float(confs[i]), int(cls_ids[i])])
    return out


def map_back(det, scale, pad, ow, oh):
    """640 letterbox 坐标 -> 原图坐标"""
    left, top = pad
    res = []
    for x1, y1, x2, y2, s, c in det:
        x1 = (x1 - left) / scale
        y1 = (y1 - top) / scale
        x2 = (x2 - left) / scale
        y2 = (y2 - top) / scale
        x1 = max(0.0, min(ow, x1)); x2 = max(0.0, min(ow, x2))
        y1 = max(0.0, min(oh, y1)); y2 = max(0.0, min(oh, y2))
        res.append([x1, y1, x2, y2, s, c])
    return res


# ---------------------------------------------------------------- 后端

class OrtDetector:
    """x86 ONNX Runtime。输入 NCHW float32 /255"""
    def __init__(self, onnx):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            onnx, providers=['CPUExecutionProvider'])
        self.name = self.sess.get_inputs()[0].name

    def __call__(self, img_bgr):
        padded, scale, pad = letterbox(img_bgr)
        rgb = padded[:, :, ::-1].transpose(2, 0, 1)
        x = np.ascontiguousarray(rgb, dtype=np.float32)[None] / 255.0
        out = self.sess.run(None, {self.name: x})[0][0]   # [84,8400]
        boxes = out[:4].T
        scores = out[4:84].T
        return postprocess(boxes, scores)


class RknnDetector:
    """RK3588 板端 rknnlite。输入 NHWC uint8（mean=0/std=255 在模型里）"""
    def __init__(self, rknn_path, core='012', box_scale=1.0, logits=False):
        from rknnlite.api import RKNNLite
        core_map = {'0': RKNNLite.NPU_CORE_0, '1': RKNNLite.NPU_CORE_1,
                    '2': RKNNLite.NPU_CORE_2, '012': RKNNLite.NPU_CORE_0_1_2}
        self.box_scale = box_scale
        self.logits = logits
        self.r = RKNNLite(verbose=False)
        if self.r.load_rknn(rknn_path) != 0:
            raise RuntimeError('load_rknn 失败: %s' % rknn_path)
        if self.r.init_runtime(core_mask=core_map[core]) != 0:
            raise RuntimeError('init_runtime 失败（NPU 打不开）')

    def __call__(self, img_bgr):
        padded, _, _ = letterbox(img_bgr)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        out = self.r.inference(inputs=[np.expand_dims(rgb, 0)])

        arrs = [np.asarray(a) for a in (out if isinstance(out, (list, tuple)) else [out])]
        boxes = scores = None
        for a in arrs:
            if a.ndim != 3:
                continue
            c = a.shape[1]
            if c == 4 and boxes is None:
                boxes = a[0].T
            elif c == 80 and scores is None:
                scores = a[0].T
        if boxes is None or scores is None:          # 回退：单输出
            for a in arrs:
                if a.ndim == 3 and a.shape[1] in (84, 85):
                    x = a[0]
                    boxes, scores = x[:4].T, x[4:84].T
                    break
        if boxes is not None and self.box_scale != 1.0:
            boxes = boxes * self.box_scale
        if self.logits and scores is not None:
            scores = sigmoid(scores)
        return postprocess(boxes, scores)

    def release(self):
        try:
            self.r.release()
        except Exception:
            pass


# ---------------------------------------------------------------- mAP

def iou_matrix(b1, b2):
    """b1:[N,4] b2:[M,4] xyxy -> [N,M]"""
    if len(b1) == 0 or len(b2) == 0:
        return np.zeros((len(b1), len(b2)))
    a1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    a2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    ix1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    iy1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    ix2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    iy2 = np.minimum(b1[:, None, 3], b2[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = a1[:, None] + a2[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def compute_ap(recall, precision):
    """COCO 的 101-point 插值"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.searchsorted(mrec, np.linspace(0, 1, 101), side='left')
    return float(mpre[idx].mean())


def evaluate(preds, gts, ious=np.arange(0.5, 1.0, 0.05), max_dets=100):
    """
    preds: {img_id: [[x1,y1,x2,y2,score,cls]...]}
    gts  : {img_id: [(cls, [x1,y1,x2,y2])...]}
    """
    aps_50, aps_all = [], []
    per_cls = {}

    for cls in range(80):
        # 该类在本批图里有没有 GT？没有就不参与平均（与 pycocotools 一致：
        # summarize 只平均 AP>-1 的类别）。否则 80 类里大部分为 0，会把结果稀释成 1/80
        n_gt_cls = sum(1 for v in gts.values() for g in v if g[0] == cls)
        if n_gt_cls == 0:
            continue

        # 收集该类的预测。max_dets 按「每图」计（跨类别），与 COCO 一致；
        # 若按类别各取 100，一张图上 3 个类就能塞 300 个框，AP 会虚高
        P = []
        for img_id, dets in preds.items():
            d = sorted(dets, key=lambda z: -z[4])[:max_dets]
            for x in d:
                if x[5] == cls:
                    P.append((img_id, x[:4], x[4]))
        if not P:                       # 有 GT 但一个都没检出来 -> AP = 0
            aps_50.append(0.0); aps_all.append(0.0); per_cls[cls] = 0.0
            continue

        P.sort(key=lambda z: -z[2])
        p_boxes = np.array([p[1] for p in P])
        p_imgs = [p[0] for p in P]

        # 每图预计算 IoU（一次算好，10 个阈值复用）
        # row_of[i] = (img_id, 该预测在 IoU 矩阵里的行号)
        iou_cache, gt_size, row_of = {}, {}, {}
        for img_id in set(p_imgs):
            gb = np.array([g[1] for g in gts.get(img_id, []) if g[0] == cls]).reshape(-1, 4)
            gt_size[img_id] = len(gb)
            rows = [i for i, im in enumerate(p_imgs) if im == img_id]
            for r, i in enumerate(rows):
                row_of[i] = (img_id, r)
            iou_cache[img_id] = iou_matrix(p_boxes[rows], gb) if len(gb) else None

        # 必须是「全部图」里该类 GT 的总数。若只统计出现过预测的图，
        # 那些「有真值但一个没检出」的图会被漏掉，recall 分母偏小 -> AP 虚高
        n_gt = n_gt_cls
        if n_gt == 0:
            aps_50.append(0.0); aps_all.append(0.0); per_cls[cls] = 0.0
            continue

        ap_list = []
        for th in ious:
            tp = np.zeros(len(P))
            fp = np.zeros(len(P))
            matched = {i: np.zeros(g, dtype=bool) for i, g in gt_size.items()}
            for i in range(len(P)):
                img_id, r = row_of[i]
                iou = iou_cache[img_id]
                if iou is None or iou.shape[1] == 0:
                    fp[i] = 1
                    continue
                best_j = int(np.argmax(iou[r]))
                if float(iou[r][best_j]) >= th and not matched[img_id][best_j]:
                    tp[i] = 1
                    matched[img_id][best_j] = True
                else:
                    fp[i] = 1

            tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
            rec = tp_c / max(n_gt, 1)
            prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
            ap_list.append(compute_ap(rec, prec))

        aps_all.append(float(np.mean(ap_list)))
        aps_50.append(ap_list[0])
        per_cls[cls] = float(np.mean(ap_list))

    return float(np.mean(aps_50)), float(np.mean(aps_all)), per_cls


# ---------------------------------------------------------------- main

def load_gts(ann_path, img_ids):
    with open(ann_path) as f:
        coco = json.load(f)
    id2file = {im['id']: im['file_name'] for im in coco['images']}
    gts = {}
    for a in coco['annotations']:
        iid = a['image_id']
        if iid not in img_ids:
            continue
        if a.get('iscrowd', 0):
            continue
        x, y, w, h = a['bbox']
        cls = COCO_IDS.index(a['category_id']) if a['category_id'] in COCO_IDS else -1
        if cls < 0:
            continue
        gts.setdefault(iid, []).append((cls, [x, y, x + w, y + h]))
    return gts, id2file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', required=True, choices=['ort', 'rknn'])
    ap.add_argument('--model', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--ann', required=True)
    ap.add_argument('--limit', type=int, default=0, help='只用前 N 张图（0=全部）')
    ap.add_argument('--conf', type=float, default=0.001,
                    help='COCO 标准评估用 0.001；若想测实际部署配置可设 0.25')
    ap.add_argument('--nms', type=float, default=0.7,
                    help='COCO 标准评估用 0.7；实际部署常用 0.45')
    ap.add_argument('--max-dets', type=int, default=300,
                    help='每张图最多保留多少检测，COCO 标准为 300')
    ap.add_argument('--core', default='012', choices=['0', '1', '2', '012'])
    ap.add_argument('--box-scale', type=float, default=1.0,
                    help='若转模型时用了 --box-scale N 压缩 boxes 量程，'
                         '这里必须用同样的 N 把 boxes 乘回来')
    ap.add_argument('--logits', action='store_true',
                    help='模型输出的是 sigmoid 之前的 logits（split_logits.py 转的），'
                         '这里补一次 sigmoid 再算分')
    ap.add_argument('--out', default='preds.json')
    args = ap.parse_args()

    # 图列表：按文件名排序，保证两个后端跑同一批
    files = sorted(f for f in os.listdir(args.img_dir)
                   if f.lower().endswith(IMG_EXTS))
    if args.limit:
        files = files[:args.limit]
    print('图片数:', len(files))

    with open(args.ann) as f:
        coco = json.load(f)
    name2id = {im['file_name']: im['id'] for im in coco['images']}
    gts, _ = load_gts(args.ann, set(name2id[f] for f in files if f in name2id))

    det = (OrtDetector(args.model) if args.backend == 'ort'
           else RknnDetector(args.model, args.core))

    preds = {}
    t0 = time.time()
    for k, fn in enumerate(files):
        img = cv2.imread(os.path.join(args.img_dir, fn))
        if img is None:
            continue
        oh, ow = img.shape[:2]
        padded, scale, pad = letterbox(img)
        if args.backend == 'rknn':
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            out = det.r.inference(inputs=[np.expand_dims(rgb, 0)])
            arrs = [np.asarray(a) for a in (out if isinstance(out, (list, tuple)) else [out])]
            boxes = scores = None
            for a in arrs:
                if a.ndim != 3:
                    continue
                c = a.shape[1]
                if c == 4 and boxes is None:
                    boxes = a[0].T
                elif c == 80 and scores is None:
                    scores = a[0].T
            if boxes is None or scores is None:
                for a in arrs:
                    if a.ndim == 3 and a.shape[1] in (84, 85):
                        x = a[0]; boxes, scores = x[:4].T, x[4:84].T; break
            if boxes is not None and args.box_scale != 1.0:
                boxes = boxes * args.box_scale      # 还原转换时压缩的量程
            if args.logits and scores is not None:
                scores = sigmoid(scores)            # logits -> 概率
            d = postprocess(boxes, scores, args.conf, args.nms)
        else:
            rgb = padded[:, :, ::-1].transpose(2, 0, 1)
            x = np.ascontiguousarray(rgb, dtype=np.float32)[None] / 255.0
            o = det.sess.run(None, {det.name: x})[0][0]
            d = postprocess(o[:4].T, o[4:84].T, args.conf, args.nms)

        preds[name2id.get(fn, fn)] = map_back(d, scale, pad, ow, oh)

        if (k + 1) % 100 == 0:
            el = time.time() - t0
            print('  %d/%d  用时 %.1fs  预计剩余 %.1fs'
                  % (k + 1, len(files), el, el / (k + 1) * (len(files) - k - 1)),
                  flush=True)

    el = time.time() - t0
    print('\n推理完成: %d 张，用时 %.1f s，平均 %.1f ms/张（含预处理+后处理）'
          % (len(files), el, el * 1000 / max(len(files), 1)))

    m50, mall, per_cls = evaluate(preds, gts, max_dets=args.max_dets)
    print('=' * 52)
    print(' mAP@0.5      : %.4f' % m50)
    print(' mAP@0.5:0.95 : %.4f' % mall)
    print('=' * 52)

    with open(args.out, 'w') as f:
        json.dump({'backend': args.backend, 'model': args.model,
                   'num_images': len(files),
                   'mAP50': m50, 'mAP50_95': mall,
                   'per_class_AP': {str(k): v for k, v in per_cls.items()},
                   'preds': {str(k): v for k, v in preds.items()}}, f)
    print(' 预测已保存:', args.out)

    if args.backend == 'rknn':
        det.release()


if __name__ == '__main__':
    main()
