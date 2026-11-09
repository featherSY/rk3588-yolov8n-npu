# -*- coding: utf-8 -*-
"""
第二课排错：诊断 ONNX 为什么「又慢 + 和 PyTorch 结果不一致」

两项诊断：
  [1] 打印 ONNX 的输入/输出结构 —— 判断 NMS 有没有被塞进计算图里
  [2] 裸测 ORT 纯前向耗时 —— 区分"慢在 ONNX 图本身"还是"慢在 ultralytics 包装层"

运行：python inspect_onnx.py
"""
import time

import numpy as np
import onnxruntime as ort

ONNX = "yolov8n.onnx"

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])

print("=" * 64)
print("[1] ONNX 模型结构")
print("=" * 64)
for i in sess.get_inputs():
    print(f"  输入   {i.name:<16s} shape={i.shape}   {i.type}")
for o in sess.get_outputs():
    print(f"  输出   {o.name:<16s} shape={o.shape}   {o.type}")

print("""
  📖 怎么读：
    · 输出 (1, 84, 8400)  → 原始输出，NMS 由外部代码做   ✅ 正常
    · 输出 (1, 300, 6)    → NMS 被塞进了 ONNX 图里       ⚠️ 又慢又容易结果不一致
    注解：84 = 4 个框坐标 + 80 个类别；8400 = 三个尺度的预测框总数
""")

print("=" * 64)
print("[2] 裸 ORT 前向耗时（不含任何前后处理，纯网络计算）")
print("=" * 64)

inp = sess.get_inputs()[0]
fixed = []
for d in inp.shape:
    fixed.append(d if isinstance(d, int) and d > 0 else 1)
if len(fixed) == 4:  # NCHW，动态的高/宽按 640 填
    fixed[2] = fixed[2] if fixed[2] > 3 else 640
    fixed[3] = fixed[3] if fixed[3] > 3 else 640

dummy = np.random.rand(*fixed).astype(np.float32)

for _ in range(10):  # 预热
    sess.run(None, {inp.name: dummy})

lat = []
for _ in range(100):
    t0 = time.perf_counter()
    sess.run(None, {inp.name: dummy})
    lat.append((time.perf_counter() - t0) * 1000)

print(f"  输入形状       : {tuple(fixed)}")
print(f"  纯前向均值     : {np.mean(lat):.1f} ms")
print(f"  纯前向中位数   : {np.median(lat):.1f} ms")
print(f"  标准差         : {np.std(lat):.1f} ms")
print("""
  📖 怎么读（这一步决定下一步怎么修）：
    · 纯前向 ~120 ms → 慢在 ONNX 图本身（算子没融合、有冗余）
                       解法：用 simplify=True 重新导出
    · 纯前向 ~30 ms  → 图没问题，慢在 ultralytics 的包装层/后处理
                       解法：自己写 ORT 推理（真实部署本来就该自己写前后处理）
""")
print("=" * 64)
