# -*- coding: utf-8 -*-
"""
第一课 Demo：用 Python 加载 YOLOv8n 对一张图片做目标检测
适用：Windows / macOS / Linux，CPU 即可运行
运行前：先激活虚拟环境，再执行  python first_detect.py
"""
from ultralytics import YOLO

# 1) 加载预训练模型
#    yolov8n.pt 是官方 nano 模型（约 6MB），首次运行会自动从官网下载到本地缓存
#    n = nano，是 YOLOv8 五个尺寸(n/s/m/l/x)里最小的一个，专为边缘/端侧设计
model = YOLO("yolov8n.pt")

# 2) 准备图片来源（二选一，把不用的那行注释掉）
#    ⚠️ 注意：pip 安装的 ultralytics 不带示例图，bus.jpg 只在 GitHub 源码仓库里，
#       所以要用"官网 URL（自动拉取）"或"自己的图"这两种方式之一
# IMG = "https://ultralytics.com/images/bus.jpg"   # 方式 A：联网自动拉取官方示例巴士图（零素材）
IMG = "test.jpg"                               # 方式 B：自己的图，放到脚本同目录（最稳）

# 3) 对图片做"推理"（inference，就是"用训练好的模型去预测新图片"）
#    save=True: 把画好检测框的结果图存到 runs/detect/predict/ 目录
#    conf=0.25: 置信度阈值，低于 0.25 的预测不显示（过滤低概率误检）
#    device="cpu": 强制用 CPU 推理（第一课先不碰 GPU）
results = model.predict(source=IMG, save=True, conf=0.25, device="cpu")

# 3) 解析并打印检测结果
for r in results:
    boxes = r.boxes          # 所有检测框
    names = r.names          # 类别映射，如 {0: 'person', 1: 'bicycle', ...}
    print(f"\n[{r.path}] 共检测到 {len(boxes)} 个目标：")
    for b in boxes:
        cls_id = int(b.cls[0])          # 类别编号
        conf = float(b.conf[0])         # 置信度（0~1）
        xyxy = b.xyxy[0].tolist()       # 框坐标 [x1, y1, x2, y2] 左上/右下像素
        print(f"  {names[cls_id]:<10s}  置信度 {conf:.2f}  坐标 {[round(x, 1) for x in xyxy]}")

print("\n推理完成 ✅  结果图已保存到 runs/detect/predict/ 目录，打开看看效果。")
