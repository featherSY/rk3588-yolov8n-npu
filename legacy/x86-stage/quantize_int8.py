# -*- coding: utf-8 -*-
"""
第三课：INT8 量化（PTQ）+ FP32 / INT8 全维度对比

做两件量化：
  1) 动态量化 Dynamic  —— 只量化权重，不需要校准数据
  2) 静态量化 Static PTQ —— 权重+激活都量化，需要校准集（工业标准做法）

然后用【同一条手撸流水线】跑三个模型对比：体积 / 延迟 / 检测结果。
复用 ort_infer.py 里的预处理和后处理，保证口径完全一致。

运行：python quantize_int8.py
"""
import glob
import os
import statistics
import time

import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_dynamic,
                                      quantize_static)

from ort_infer import IMG, postprocess, preprocess

FP32 = "yolov8n.onnx"
DYN = "yolov8n_int8_dyn.onnx"
STATIC = "yolov8n_int8_static.onnx"
WARMUP, RUNS = 10, 100


class YOLOCalibReader(CalibrationDataReader):
    """校准数据读取器：把图片预处理后逐个喂给量化器统计激活范围"""

    def __init__(self, paths, input_name):
        self.name = input_name
        self.data = [preprocess(p)[0] for p in paths]   # 每张图 -> (1,3,640,640)
        self.i = 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        d = {self.name: self.data[self.i]}
        self.i += 1
        return d

    def rewind(self):
        self.i = 0


def bench(model_path, tag):
    """用同一条流水线测一个模型：体积 + 网络前向延迟 + 检测结果"""
    if not os.path.exists(model_path):
        print(f"  {tag:<18s} 文件不存在，跳过")
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

    boxes = postprocess(out, r, pad)
    size = os.path.getsize(model_path) / 1024 / 1024
    mean = statistics.mean(lat)
    print(f"  {tag:<18s} 体积 {size:5.2f} MB | 前向 {mean:6.1f} ms | "
          f"{1000 / mean:5.1f} FPS | 检出 {len(boxes)} 框")
    return {"size": size, "lat": mean, "boxes": boxes}


print("=" * 70)
print("第三课：INT8 量化（PTQ）")
print("=" * 70)

# ---------- 校准集：目录下所有 jpg ----------
calib_paths = sorted(glob.glob("*.jpg"))
print(f"\n校准集：找到 {len(calib_paths)} 张 jpg")
if len(calib_paths) < 10:
    print("  ⚠️ 校准集偏少（工业界建议 100~300 张有代表性的图），")
    print("     当前结果仅供流程验证；校准集越接近真实场景，量化精度越好。")

# ---------- 1) 动态量化 ----------
print("\n[1] 动态量化（只量化权重，无需校准数据）...")
try:
    quantize_dynamic(FP32, DYN, weight_type=QuantType.QInt8)
    print(f"    ✅ 已生成 {DYN}")
except Exception as e:
    print(f"    ❌ 失败：{e}")

# ---------- 2) 静态量化 PTQ ----------
print("\n[2] 静态量化 PTQ（权重+激活，需校准集）...")
try:
    tmp = ort.InferenceSession(FP32, providers=["CPUExecutionProvider"])
    in_name = tmp.get_inputs()[0].name
    reader = YOLOCalibReader(calib_paths or [IMG], in_name)
    quantize_static(
        FP32, STATIC, reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,     # CPU 较新(支持 VNNI)设 False；老 CPU 溢出时改 True
    )
    print(f"    ✅ 已生成 {STATIC}")
except Exception as e:
    print(f"    ❌ 失败：{e}")
    print("       静态量化对算子支持更挑剔，失败不影响动态量化的结论。")

# ---------- 3) 三模型对比 ----------
print("\n[3] 三模型对比（同一条流水线，100 帧统计）")
print("  " + "-" * 66)
fp = bench(FP32, "FP32 (原始)")
dy = bench(DYN, "INT8 动态")
st = bench(STATIC, "INT8 静态PTQ")

print("\n[4] 量化收益")
print("  " + "-" * 66)
for tag, m in (("INT8 动态", dy), ("INT8 静态PTQ", st)):
    if m and fp:
        print(f"  {tag:<14s} 体积 → {m['size'] / fp['size'] * 100:5.1f}%   "
              f"延迟 → {m['lat'] / fp['lat'] * 100:5.1f}%   "
              f"加速 {fp['lat'] / m['lat']:.2f}x")

print("\n[5] 精度变化（检测到的目标）")
print("  " + "-" * 66)
for tag, m in (("FP32", fp), ("INT8 动态", dy), ("INT8 静态PTQ", st)):
    if m:
        s = ", ".join(f"{c:.3f}" for c in sorted([b[1] for b in m["boxes"]], reverse=True))
        print(f"  {tag:<14s} 共 {len(m['boxes'])} 个框，置信度: {s}")

print("""
  📖 怎么读：
    · 体积应降到 ~25%（4 字节 -> 1 字节），这是 INT8 最稳的收益
    · 延迟：动态量化一般小幅提速；静态 PTQ 提速更明显（激活也走整数运算）
    · 精度：注意看【低置信度目标有没有消失】——量化精度损失最先体现在它们身上
""")
print("=" * 70)
