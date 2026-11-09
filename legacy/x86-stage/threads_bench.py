import torch, time
from ultralytics import YOLO

net = YOLO('yolov8n.pt').model.eval()
img = torch.zeros(1, 3, 640, 640)

print(f"{'线程':>4}  {'中位数':>8}  {'最快':>8}")
for n in [1, 2, 4, 8, 12, 16, 20]:
    torch.set_num_threads(n)
    with torch.no_grad():
        for _ in range(10):
            net(img)                      # 预热
        ts = []
        for _ in range(50):
            t = time.perf_counter()
            net(img)
            ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    print(f"{n:>4}  {ts[len(ts)//2]:>7.2f}ms  {ts[0]:>7.2f}ms")
