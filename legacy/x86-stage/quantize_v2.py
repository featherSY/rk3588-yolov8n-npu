# -*- coding: utf-8 -*-
"""
第三课（修正版）：诊断 + 修复静态 PTQ 精度崩坏

上一轮结果：
  FP32        12.26 MB  27.4 ms  3 框   ✅
  INT8 动态    3.34 MB  56.7 ms  4 框   ❌ 慢了 2 倍
  INT8 静态    3.27 MB  20.8 ms  0 框   ❌ 快了但精度崩坏

本脚本做两件事：
  [1] 诊断：打印各模型原始输出的数值统计，看 INT8 静态是"分数被压低"还是"数值完全崩坏"
  [2] 修复：换 3 种量化配置重新生成，挑出精度可用的那一个
       A) reduce_range=True        —— 规避老 CPU 上 INT8 累加溢出（最常见修复）
       B) 只量化 Conv 层           —— 输出头的 Sigmoid/Mul 对量化极敏感，跳过它们
       C) QDQ 格式                 —— 兼容性更好的量化表示

运行：python quantize_v2.py
"""
import glob
import os
import statistics
import time

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
    """跑一次，输出：数值统计 + 延迟 + 检出框数（诊断崩坏原因的关键）"""
    if not os.path.exists(model_path):
        print(f"  {tag:<16s} 文件不存在")
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

    cls_scores = out[0, 4:, :]              # 80 个类别的分数
    boxes = postprocess(out, r, pad)
    size = os.path.getsize(model_path) / 1024 / 1024
    mean = statistics.mean(lat)

    print(f"  {tag:<16s} {size:5.2f} MB | {mean:6.1f} ms | {1000 / mean:5.1f} FPS | "
          f"{len(boxes):2d} 框")
    print(f"  {'':<16s} 原始输出 min={out.min():8.3f} max={out.max():8.3f} "
          f"mean={out.mean():7.3f} | 类别分数最高={cls_scores.max():.3f} (阈值0.25)")
    return {"size": size, "lat": mean, "boxes": boxes, "cls_max": float(cls_scores.max())}


# ================= 诊断阶段 =================
print("=" * 72)
print("[1] 诊断：为什么静态 PTQ 一个框都检不出？")
print("=" * 72)
print("  看「类别分数最高」这一项：")
print("    · 明显低于 0.25 → 分数被整体压低，量化误差累积")
print("    · 数值 min/max 与 FP32 差几个数量级 → 数值彻底崩坏（多半是溢出）")
print()
fp = probe(FP32, "FP32 基准")
if os.path.exists("yolov8n_int8_static.onnx"):
    probe("yolov8n_int8_static.onnx", "INT8静态(旧)")
if os.path.exists("yolov8n_int8_dyn.onnx"):
    probe("yolov8n_int8_dyn.onnx", "INT8动态")

# ================= 修复阶段 =================
print("\n" + "=" * 72)
print("[2] 修复：换配置重新量化")
print("=" * 72)

calib_paths = sorted(glob.glob("*.jpg")) or [IMG]
tmp = ort.InferenceSession(FP32, providers=["CPUExecutionProvider"])
in_name = tmp.get_inputs()[0].name
reader = YOLOCalibReader(calib_paths, in_name)

CONFIGS = [
    ("A_reduce_range", "yolov8n_int8_A.onnx",
     dict(quant_format=QuantFormat.QOperator, activation_type=QuantType.QUInt8,
          weight_type=QuantType.QInt8, reduce_range=True)),
    ("B_只量化Conv", "yolov8n_int8_B.onnx",
     dict(quant_format=QuantFormat.QOperator, activation_type=QuantType.QUInt8,
          weight_type=QuantType.QInt8, reduce_range=True,
          op_types_to_quantize=["Conv"])),
    ("C_QDQ格式", "yolov8n_int8_C.onnx",
     dict(quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
          weight_type=QuantType.QInt8, reduce_range=True)),
]

results = {}
for tag, out_path, kwargs in CONFIGS:
    print(f"\n  尝试 {tag} ...")
    try:
        reader.rewind()
        quantize_static(FP32, out_path, reader, **kwargs)
        print(f"    ✅ 生成 {out_path}")
    except Exception as e:
        print(f"    ❌ 失败: {str(e)[:120]}")
        continue

print("\n" + "=" * 72)
print("[3] 修复后的三配置对比")
print("=" * 72)
for tag, out_path, _ in CONFIGS:
    if os.path.exists(out_path):
        results[tag] = probe(out_path, tag)

print("\n[4] 结论")
print("  " + "-" * 68)
if fp:
    for tag, m in results.items():
        if m:
            verdict = "✅ 精度可用" if m["boxes"] else "❌ 仍崩坏"
            print(f"  {tag:<16s} 加速 {fp['lat'] / m['lat']:.2f}x | 体积 {m['size'] / fp['size'] * 100:.1f}% "
                  f"| {len(m['boxes'])} 框 | {verdict}")
print("""
  📖 如果三种都崩坏，说明这台机器的 INT8 路径与该模型不兼容，
     那也是真实结论——简历上写"实测 INT8 在 x86 CPU 上精度不达标，
     需改为在 NPU 端部署"同样是有价值的一手经验。
""")
print("=" * 72)
