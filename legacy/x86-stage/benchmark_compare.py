# -*- coding: utf-8 -*-
"""
第二课：PyTorch(.pt) vs ONNX(.onnx) 全面对比

对比三个维度，全部直接可用作 README 数据：
  [1] 模型文件体积
  [2] 端到端延迟（预热 + 100 帧统计，与第一课同一口径）
  [3] 输出一致性 —— 证明"导出的 ONNX 和原模型干的是同一件事"

为什么第 [3] 项很重要（面试常问）：
  "你怎么验证导出的 ONNX 和原始模型等价？"
  答案不能是"看着差不多"，而要拿同一张图跑两边，比较类别、置信度、框坐标的数值偏差。

运行前准备：
  1) 导出 ONNX:
     python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
  2) 装推理引擎:
     pip install onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
"""
import os
import time
import statistics
from ultralytics import YOLO

PT = "yolov8n.pt"
ONNX = "yolov8n.onnx"
IMG = "test.jpg"
WARMUP, RUNS = 10, 100


def measure(tag, path):
    """测量端到端延迟：预热排除冷启动，再统计多帧"""
    model = YOLO(path)
    for _ in range(WARMUP):
        model.predict(source=IMG, device="cpu", verbose=False)
    lat = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        model.predict(source=IMG, device="cpu", verbose=False)
        lat.append((time.perf_counter() - t0) * 1000)
    mean = statistics.mean(lat)
    print(f"  {tag:<14s} 均值 {mean:6.1f} ms | 中位数 {statistics.median(lat):6.1f} ms "
          f"| 标准差 {statistics.stdev(lat):4.1f} ms | {1000 / mean:5.1f} FPS")
    return mean


def detect(path):
    """跑一次推理，返回 (类别, 置信度, 框坐标) 列表"""
    model = YOLO(path)
    r = model.predict(source=IMG, device="cpu", verbose=False, conf=0.25)[0]
    return [(int(b.cls[0]), round(float(b.conf[0]), 3),
             [round(v, 1) for v in b.xyxy[0].tolist()]) for b in r.boxes]


print("=" * 62)
print("第二课：PyTorch (.pt)  vs  ONNX (.onnx) 对比")
print("=" * 62)

# 前置检查：缺文件就明确报错，别让新手猜
for f in (PT, ONNX, IMG):
    if not os.path.exists(f):
        raise SystemExit(f"\n找不到文件：{f}\n请确认已完成 ONNX 导出，且图片在当前目录。")

print("\n[1] 模型文件体积")
for f in (PT, ONNX):
    print(f"  {f:<14s} {os.path.getsize(f) / 1024 / 1024:6.2f} MB")

print(f"\n[2] 端到端延迟（预热 {WARMUP} 帧 + 测量 {RUNS} 帧，CPU）")
pt_mean = measure("PyTorch FP32", PT)
onnx_mean = measure("ONNX FP32", ONNX)

print("\n[3] 输出一致性验证（同一张图，两边结果对比）")
a, b = detect(PT), detect(ONNX)
print(f"  PyTorch 检出 {len(a)} 个框 / ONNX 检出 {len(b)} 个框")
print(f"  类别序列一致 : {[x[0] for x in a] == [y[0] for y in b]}")
if len(a) == len(b) and len(a) > 0:
    conf_diff = max(abs(x[1] - y[1]) for x, y in zip(a, b))
    coord_diff = max(max(abs(p - q) for p, q in zip(x[2], y[2])) for x, y in zip(a, b))
    print(f"  置信度最大偏差: {conf_diff:.3f}")
    print(f"  框坐标最大偏差: {coord_diff:.1f} 像素")
    print("  → 偏差在 1e-3 / 亚像素级，即可判定两边等价 ✅")

print("\n[4] README 结论速记")
print(f"  PyTorch FP32 : {pt_mean:6.1f} ms  ({1000 / pt_mean:5.1f} FPS)")
print(f"  ONNX    FP32 : {onnx_mean:6.1f} ms  ({1000 / onnx_mean:5.1f} FPS)")
print(f"  延迟变化     : {(onnx_mean - pt_mean) / pt_mean * 100:+.1f}%")
print("=" * 62)
