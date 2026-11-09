for n in 6 8 10 20; do
python - "$n" <<'PY'
import sys, torch, time
from ultralytics import YOLO
n = int(sys.argv[1])
torch.set_num_threads(n)
net = YOLO('yolov8n.pt').model.eval()
img = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    for _ in range(10): net(img)
    ts = []
    for _ in range(50):
        t = time.perf_counter(); net(img); ts.append((time.perf_counter()-t)*1000)
ts.sort()
print(f"{n:>3} 线程: 中位数 {ts[len(ts)//2]:7.2f} ms   最快 {ts[0]:7.2f} ms")
PY
done
