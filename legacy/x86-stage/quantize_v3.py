# -*- coding: utf-8 -*-
"""
第三课（v3）：选择性量化 —— 检测头保持 FP32

根因回顾：
    输出张量 (1,84,8400) 里，框坐标值域 0~636、类别分数值域 0~1。
    静态量化的激活是 per-tensor（整个张量共用一把尺子）：
        scale = 636/255 ≈ 2.49
        类别分数 0.946 -> round(0.946/2.49) = 0     ← 被抹平成 0
    所以框坐标正常、类别分数全零。

解法（工业界标准）：
    量化骨干网络 + 颈部，【检测头 head 保持 FP32】。
    检测头输出直接决定框坐标和类别分数，对量化最敏感；
    而它的计算量占比小，不量化损失不大。

需要先装：pip install onnx -i https://pypi.tuna.tsinghua.edu.cn/simple
运行：python quantize_v3.py
"""
import glob
import os
import statistics
import time

import onnx
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)

from ort_infer import IMG, postprocess, preprocess

FP32 = "yolov8n.onnx"
WARMUP, RUNS = 10, 100


class YOLOCalibReader(CalibrationDataReader):
    def __init__(self, paths, input_name):
        self.name = input_name
        self.data = [preprocess(p)[0] for p in paths]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        d = {self.name: self.data[self.i]}
        self.i += 1
        return d

    def rewind(self):
        self.i = 0


def probe(model_path, tag):
    if not os.path.exists(model_path):
        print(f"  {tag:<18s} 文件不存在")
        return None
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x, im0, r, pad = preprocess(IMG)
    for _ in range(WARMUP):
        sess.run(None, {name: x})
    lat = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        out = sess.run(None, {name: x})[0]
        lat.append((time.perf_counter() - t0) * 1000)
    cls_max = float(out[0, 4:, :].max())
    boxes = postprocess(out, r, pad)
    size = os.path.getsize(model_path) / 1024 / 1024
    mean = statistics.mean(lat)
    print(f"  {tag:<18s} {size:5.2f} MB | {mean:6.1f} ms | {1000 / mean:5.1f} FPS | "
          f"{len(boxes):2d} 框 | 类别分数最高={cls_max:.3f}")
    return {"size": size, "lat": mean, "boxes": boxes, "cls_max": cls_max}


# ============ 1) 看看模型图长什么样，找出检测头 ============
print("=" * 74)
print("[1] 分析 ONNX 图结构，定位检测头节点")
print("=" * 74)
model = onnx.load(FP32)
nodes = list(model.graph.node)
print(f"  总节点数: {len(nodes)}")
print("\n  最后 12 个节点：")
for n in nodes[-12:]:
    print(f"    {n.op_type:<14s} {n.name}")

# YOLOv8 的检测头：分支名通常含 cv2（框回归）/ cv3（分类）
head_by_name = [n.name for n in nodes
                if n.op_type == "Conv" and ("cv2" in n.name or "cv3" in n.name)]
convs = [n.name for n in nodes if n.op_type == "Conv"]
last6 = convs[-6:]

print(f"\n  按名字匹配到的检测头 Conv (cv2/cv3): {len(head_by_name)} 个")
for n in head_by_name[:8]:
    print(f"    {n}")
print(f"\n  备选（最后 6 个 Conv）:")
for n in last6:
    print(f"    {n}")

# ============ 2) 用不同排除策略重新量化 ============
print("\n" + "=" * 74)
print("[2] 选择性量化：排除检测头，其余照常量化")
print("=" * 74)

calib_paths = sorted(glob.glob("*.jpg")) or [IMG]
tmp = ort.InferenceSession(FP32, providers=["CPUExecutionProvider"])
in_name = tmp.get_inputs()[0].name
reader = YOLOCalibReader(calib_paths, in_name)

CONFIGS = [
    ("D_排除头", "yolov8n_int8_D.onnx", head_by_name, False),
    ("E_排除头+perCh", "yolov8n_int8_E.onnx", head_by_name, True),
    ("F_排除末6Conv", "yolov8n_int8_F.onnx", last6, True),
]

for tag, out_path, exclude, per_ch in CONFIGS:
    if not exclude:
        print(f"\n  跳过 {tag}：没找到可排除的节点")
        continue
    print(f"\n  量化 {tag}（排除 {len(exclude)} 个节点, per_channel={per_ch}）...")
    try:
        reader.rewind()
        quantize_static(
            FP32, out_path, reader,
            quant_format=QuantFormat.QOperator,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_ch,
            reduce_range=False,
            nodes_to_exclude=exclude,
        )
        print(f"    ✅ 生成 {out_path}")
    except Exception as e:
        print(f"    ❌ 失败: {str(e)[:150]}")

# ============ 3) 对比 ============
print("\n" + "=" * 74)
print("[3] 全部方案横评")
print("=" * 74)
fp = probe(FP32, "FP32 基准")
results = {}
for tag, out_path, _, _ in CONFIGS:
    if os.path.exists(out_path):
        results[tag] = probe(out_path, tag)
if os.path.exists("yolov8n_int8_B.onnx"):
    results["B_只量化Conv"] = probe("yolov8n_int8_B.onnx", "B_只量化Conv")

print("\n[4] 结论")
print("  " + "-" * 70)
best = None
for tag, m in results.items():
    if not m or not fp:
        continue
    ok = "✅" if m["boxes"] else "❌"
    speed = fp["lat"] / m["lat"]
    print(f"  {tag:<16s} 加速 {speed:4.2f}x | 体积 {m['size'] / fp['size'] * 100:5.1f}% | "
          f"{len(m['boxes'])} 框 | 最高分 {m['cls_max']:.3f} | {ok}")
    if m["boxes"] and speed > 1.0:
        if best is None or speed > fp["lat"] / results[best]["lat"]:
            best = tag

print()
if best:
    print(f"  🏆 最佳方案：{best}")
    print("     把它作为 README 里的 INT8 数据，并注明「检测头保持 FP32」这一关键决策。")
else:
    print("  ⚠️ 没有既加速又保精度的方案。这也是真实结论：")
    print("     x86 CPU 的 INT8 卷积收益有限，真正的加速要靠 NPU（RK3588 阶段）。")
print("=" * 74)
