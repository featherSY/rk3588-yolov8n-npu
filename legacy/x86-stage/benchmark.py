# -*- coding: utf-8 -*-
"""
第一课收尾：严谨测量 CPU 推理基线（10 分钟）

为什么要写这个脚本？
    单次 model.predict() 受「首次内存分配 / 系统调度 / CPU 频率波动」影响很大，
    同一台机器在不同时刻跑，结果能差 30% 以上。
    直接把单次数据写进 README，面试官一句"这数据怎么测的"就能问倒你。
    可信的工程做法：预热若干帧 → 连续推理 N 帧 → 报平均值 + 中位数 + 标准差。

口径说明（很重要，README 里要写清楚）：
    本脚本测的是【端到端耗时】= 读图 + 预处理 + 网络前向 + 后处理(NMS)。
    而 ultralytics 自己打印的 Speed 行里那个 inference 只算【纯网络前向】。
    两个口径都要记，不能混用比较。建议 README 里优先用端到端，更贴近真实业务。

运行：python benchmark.py
"""
import time
import statistics
from ultralytics import YOLO

IMG = "test.jpg"        # 换成你自己的图片文件名
WARMUP = 10             # 预热帧数：跑掉首次初始化开销，不计入统计
RUNS = 100              # 正式测量帧数：越多越稳，100 帧大约十几秒

model = YOLO("yolov8n.pt")

# 1) 预热：让 CPU 缓存、内存池、线程都进入稳态
print(f"预热 {WARMUP} 帧 ...")
for _ in range(WARMUP):
    model.predict(source=IMG, device="cpu", verbose=False)

# 2) 正式测量：逐帧计时
print(f"正式测量 {RUNS} 帧 ...")
latencies = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    model.predict(source=IMG, device="cpu", verbose=False)
    latencies.append((time.perf_counter() - t0) * 1000)  # 秒 -> 毫秒

# 3) 统计输出
mean = statistics.mean(latencies)
med = statistics.median(latencies)
print("\n===== CPU 推理基线（端到端） =====")
print(f"输入图片   : {IMG}")
print(f"模型       : yolov8n.pt (FP32)")
print(f"设备       : CPU")
print(f"样本数     : {RUNS} 帧（预热 {WARMUP} 帧不计）")
print(f"平均耗时   : {mean:.1f} ms")
print(f"中位数     : {med:.1f} ms")
print(f"最快/最慢  : {min(latencies):.1f} / {max(latencies):.1f} ms")
print(f"标准差     : {statistics.stdev(latencies):.1f} ms")
print(f"等效帧率   : {1000 / mean:.1f} FPS")
print("\n把这张表抄进 README，就是后面 INT8 量化的对比基准线。")
