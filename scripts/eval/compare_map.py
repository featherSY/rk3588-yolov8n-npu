#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比两次 mAP 评估的结果，输出掉点分析。

用法：
    python3 compare_map.py results/preds_ort_fp32_500.json results/preds_rknn_int8_500.json
    python3 compare_map.py <基线.json> <对比.json> --md report.md

会输出：
  - 总览：mAP@0.5 / mAP@0.5:0.95 的绝对值与掉点
  - 掉点最多的 Top-5 类别（INT8 到底损失在哪几类上）
  - 掉点最少的 Top-3 类别（对照组）
"""

import json
import argparse

# COCO 80 类名称，顺序对应类别索引 0~79
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


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('baseline', help='基线 json（通常是 FP32 / x86）')
    ap.add_argument('target', help='对比 json（通常是 INT8 / RK3588）')
    ap.add_argument('--md', help='额外把表格写成 markdown 文件')
    args = ap.parse_args()

    a, b = load(args.baseline), load(args.target)

    print('=' * 62)
    print(' 基线 : %-24s (%s)' % (a['model'], a['backend']))
    print(' 对比 : %-24s (%s)' % (b['model'], b['backend']))
    print(' 图片数: 基线 %d / 对比 %d' % (a['num_images'], b['num_images']))
    if a['num_images'] != b['num_images'] or \
       a.get('img_dir') != b.get('img_dir'):
        print(' !! 两边图片数或图集不同，mAP 不可直接比较')
    print('=' * 62)

    rows = []
    for key, name in (('mAP50', 'mAP@0.5'), ('mAP50_95', 'mAP@0.5:0.95')):
        va, vb = a[key], b[key]
        rows.append((name, va, vb, vb - va))
        print(' %-14s  %.4f  ->  %.4f   %+.4f' % (name, va, vb, vb - va))
    print('=' * 62)

    # ---- 诊断：mAP@0.5 涨、mAP@0.5:0.95 跌，八成是检测框数量/分数分布变了
    pa_det = a.get('preds', {})
    pb_det = b.get('preds', {})
    if pa_det and pb_det:
        na = [len(v) for v in pa_det.values()]
        nb = [len(v) for v in pb_det.values()]
        print('\n 诊断：检测框数量')
        print('   每图平均框数   基线 %.1f  ->  对比 %.1f'
              % (sum(na) / len(na), sum(nb) / len(nb)))
        print('   框总数         %d  ->  %d' % (sum(na), sum(nb)))

        sa = sorted(x[4] for v in pa_det.values() for x in v)
        sb = sorted(x[4] for v in pb_det.values() for x in v)
        if sa and sb:
            def q(lst, p):
                return lst[int(len(lst) * p)]
            print('\n 诊断：置信度分位数')
            print('   %-8s %8s %8s' % ('分位', '基线', '对比'))
            for p, name in ((0.1, 'P10'), (0.5, 'P50'), (0.9, 'P90')):
                print('   %-8s %8.4f %8.4f' % (name, q(sa, p), q(sb, p)))

            # 框面积：INT8 框系统性变大/变小会直接影响 IoU
            def areas(det):
                out = []
                for v in det.values():
                    for x in v:
                        out.append(max(x[2] - x[0], 0) * max(x[3] - x[1], 0))
                return sorted(out)
            aa, ab = areas(pa_det), areas(pb_det)
            if aa and ab:
                print('\n 诊断：框面积中位数（原图像素²）')
                print('   基线 %.0f  ->  对比 %.0f   (%+.1f%%)'
                      % (q(aa, 0.5), q(ab, 0.5),
                         (q(ab, 0.5) / max(q(aa, 0.5), 1) - 1) * 100))

    # 类别级掉点
    pa = {int(k): v for k, v in a.get('per_class_AP', {}).items()}
    pb = {int(k): v for k, v in b.get('per_class_AP', {}).items()}
    common = sorted(set(pa) & set(pb))
    if not common:
        print(' 没有可对比的类别数据')
        return

    diffs = [(cls, pa[cls], pb[cls], pb[cls] - pa[cls]) for cls in common]
    diffs.sort(key=lambda z: z[3])

    print('\n 掉点最多的 5 个类别：')
    print('   %-16s %8s %8s %8s' % ('类别', '基线', '对比', '掉点'))
    for cls, va, vb, d in diffs[:5]:
        print('   %-16s %8.4f %8.4f %+8.4f' % (CLASSES[cls], va, vb, d))

    print('\n 掉点最少（甚至反超）的 3 个类别：')
    print('   %-16s %8s %8s %8s' % ('类别', '基线', '对比', '掉点'))
    for cls, va, vb, d in diffs[-3:]:
        print('   %-16s %8.4f %8.4f %+8.4f' % (CLASSES[cls], va, vb, d))

    mean_drop = sum(d for _, _, _, d in diffs) / len(diffs)
    print('\n 80 类平均掉点: %+.4f' % mean_drop)
    print('=' * 62)

    if args.md:
        with open(args.md, 'w') as f:
            f.write('| 指标 | 基线 (%s) | 对比 (%s) | 掉点 |\n'
                    % (a['backend'], b['backend']))
            f.write('|---|---|---|---|\n')
            for name, va, vb, d in rows:
                f.write('| %s | %.4f | %.4f | %+.4f |\n' % (name, va, vb, d))
            f.write('\n### 掉点最多的 5 个类别\n\n')
            f.write('| 类别 | 基线 | 对比 | 掉点 |\n|---|---|---|---|\n')
            for cls, va, vb, d in diffs[:5]:
                f.write('| %s | %.4f | %.4f | %+.4f |\n' % (CLASSES[cls], va, vb, d))
        print(' markdown 已保存:', args.md)


if __name__ == '__main__':
    main()
