for n in 0 1 2 4 8 10 20; do
python - "$n" <<'PY'
import sys, time, numpy as np, onnxruntime as ort
n = int(sys.argv[1])
so = ort.SessionOptions()
if n > 0:
    so.intra_op_num_threads = n          # n=0 时不设置，测 ORT 默认行为
sess = ort.InferenceSession('yolov8n.onnx', so, providers=['CPUExecutionProvider'])
name = sess.get_inputs()[0].name
x = np.zeros((1, 3, 640, 640), dtype=np.float32)
for _ in range(10): sess.run(None, {name: x})     # 预热
ts = []
for _ in range(50):
    t = time.perf_counter()
    sess.run(None, {name: x})
    ts.append((time.perf_counter() - t) * 1000)
ts.sort()
tag = "默认" if n == 0 else f"{n:>3} 线程"
print(f"{tag}: 中位数 {ts[len(ts)//2]:7.2f} ms   最快 {ts[0]:7.2f} ms")
PY
done
