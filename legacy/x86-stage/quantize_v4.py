# -*- coding: utf-8 -*-
"""
第三课（v4 · 终版）：按模块路径排除检测头 + 严谨性能测量

前两版踩的坑（都是真实教训）：
  坑1: 按 "cv2"/"cv3" 匹配节点 —— 这两个名字在骨干的 C2f 里也有，
       误伤 37 个骨干节点，导致体积只降到 68.6%。
  坑2: 只排除 Conv —— 输出通路上的 Slice/Sub/Add/Div/Concat/Mul
       是真正的"抹平"执行者，它们的激活量化没被排除，照样 0 框。
  坑3: 单次测量波动巨大 —— 同一个模型两次测出 28.4ms 和 18.9ms（差 50%），
       所以必须多轮取中位数。

本版做法：
  G 方案 = 排除整个 /model.22/ 模块（真正的 Detect head），其余全部量化。
  并对所有候选模型做 3 轮 × 100 帧测量，取中位数，控制波动。

运行前建议：关掉 Chrome / Steam 等后台，插电源，测量时别动电脑。
运行：python quantize_v4.py
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
OUT_G = "yolov8n_int8_G.onnx"
WARMUP, RUNS, ROUNDS = 10, 100, 3


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


def probe_strict(model_path, tag):
    """严谨测量：3 轮 × 100 帧，取各轮中位数后再取中位数"""
    if not os.path.exists(model_path):
        print(f"  {tag:<18s} 文件不存在")
        return None
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x, im0, r, pad = preprocess(IMG)

    round_medians = []
    for _ in range(ROUNDS):
        for _ in range(WARMUP):
            sess.run(None, {name: x})
        lat = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = sess.run(None, {name: x})[0]
            lat.append((time.perf_counter() - t0) * 1000)
        round_medians.append(statistics.median(lat))

    mean_of_med = statistics.median(round_medians)
    cls_max = float(out[0, 4:, :].max())
    boxes = postprocess(out, r, pad)
    size = os.path.getsize(model_path) / 1024 / 1024
    spread = max(round_medians) - min(round_medians)
    print(f"  {tag:<18s} {size:5.2f} MB | {mean_of_med:6.1f} ms | "
          f"{1000 / mean_of_med:5.1f} FPS | {len(boxes):2d} 框 | "
          f"最高分 {cls_max:.3f} | 轮间波动 {spread:.1f} ms")
    return {"size": size, "lat": mean_of_med, "boxes": boxes,
            "cls_max": cls_max, "spread": spread}


# ============ 1) 按模块路径定位检测头 ============
print("=" * 78)
print("[1] 按模块路径定位检测头（model.22 = Detect head）")
print("=" * 78)
model = onnx.load(FP32)
nodes = list(model.graph.node)

head_nodes = [n.name for n in nodes if n.name.startswith("/model.22/")]
backbone = [n.name for n in nodes if not n.name.startswith("/model.22/")]
print(f"  总节点 {len(nodes)}  →  检测头 {len(head_nodes)} 个 / 骨干+颈部 {len(backbone)} 个")
print("\n  检测头里被排除的关键节点（节选）：")
for n in head_nodes[-8:]:
    print(f"    {n}")

# ============ 2) 生成 G 方案 ============
print("\n" + "=" * 78)
print("[2] 量化 G 方案：排除整个 /model.22/ 检测头")
print("=" * 78)
calib_paths = sorted(glob.glob("*.jpg")) or [IMG]
tmp = ort.InferenceSession(FP32, providers=["CPUExecutionProvider"])
in_name = tmp.get_inputs()[0].name
reader = YOLOCalibReader(calib_paths, in_name)

try:
    reader.rewind()
    quantize_static(
        FP32, OUT_G, reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        nodes_to_exclude=head_nodes,
    )
    print(f"  ✅ 生成 {OUT_G}")
except Exception as e:
    print(f"  ❌ 失败: {str(e)[:150]}")

# ============ 3) 严谨横评 ============
print("\n" + "=" * 78)
print(f"[3] 严谨横评（{ROUNDS} 轮 × {RUNS} 帧，取中位数）")
print("=" * 78)
fp = probe_strict(FP32, "FP32 基准")
results = {}
for tag, path in (("B_只量化Conv", "yolov8n_int8_B.onnx"),
                  ("G_排除检测头", OUT_G),
                  ("F_排除末6Conv", "yolov8n_int8_F.onnx")):
    if os.path.exists(path):
        results[tag] = probe_strict(path, tag)

print("\n[4] 最终结论")
print("  " + "-" * 74)
best = None
for tag, m in results.items():
    if not m or not fp:
        continue
    speed = fp["lat"] / m["lat"]
    ok = "✅ 可用" if m["boxes"] else "❌ 精度崩"
    print(f"  {tag:<16s} 加速 {speed:4.2f}x | 体积 {m['size'] / fp['size'] * 100:5.1f}% | "
          f"{len(m['boxes'])} 框 | 最高分 {m['cls_max']:.3f} | {ok}")
    if m["boxes"] and speed > 1.0:
        if best is None or speed > fp["lat"] / results[best]["lat"]:
            best = tag

print()
if best:
    b = results[best]
    print(f"  🏆 最终方案：{best}")
    print(f"     延迟 {fp['lat']:.1f} ms → {b['lat']:.1f} ms（{fp['lat'] / b['lat']:.2f}x）")
    print(f"     体积 {fp['size']:.2f} MB → {b['size']:.2f} MB（{b['size'] / fp['size'] * 100:.1f}%）")
    print(f"     检测框 {len(b['boxes'])} 个，最高置信度 {b['cls_max']:.3f}（FP32 为 0.946）")
    print("\n  → 这组数据可以进 README 了，注明量化策略与测量方法。")
else:
    print("  ⚠️ 仍未拿到「又快又准」的方案。真实结论同样有价值：")
    print("     x86 CPU 上 INT8 卷积收益有限，真正的加速要等 NPU（RK3588 阶段）。")
print("=" * 78)
