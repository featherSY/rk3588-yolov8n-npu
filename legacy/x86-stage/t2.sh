for n in 8 20; do
python - "$n" <<'PY'
import sys, torch, time
from ultralytics import YOLO
n = int(sys.argv[1])
torch.set_num_threads(n)
m = YOLO('yolov8n.pt')
for _ in range(10):
    m.predict(source='test.jpg', device='cpu', verbose=False)
ts = []
for _ in range(30):
    t = time.perf_counter()
    m.predict(source='test.jpg', device='cpu', verbose=False)
    ts.append((time.perf_counter() - t) * 1000)
ts.sort()
print(f"{n:>3} 线程: 中位数 {ts[len(ts)//2]:7.2f} ms   最快 {ts[0]:7.2f} ms")
PY
done
