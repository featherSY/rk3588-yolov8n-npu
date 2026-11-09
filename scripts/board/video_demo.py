# -*- coding: utf-8 -*-
"""
RK3588 板端 YOLOv8n 视频流实时检测 demo（无摄像头 / 无显示器也能跑）

设计目标：补上项目规划 §2 的「⑥ 实时演示」，不依赖摄像头、不依赖显示器。

输入源三选一（--source）：
  1) 图片目录   逐张当帧，例如 coco/val2017（零额外素材，推荐）
  2) 视频文件   mp4 / avi
  3) V4L2 设备  /dev/videoN（以后插了 USB 摄像头可直接切，无需改代码）

输出：
  - 带框视频写到 <项目根>/results/demo_output.mp4
  - 可选 --http 8080 起 MJPEG 流，PC 浏览器打开 http://100.75.157.90:8080 看实时画面
  - 同时统计「纯前向 rknn.inference()」与「端到端（含前后处理）」两个延迟口径，
    用来推进报告 P0 的口径对齐问题

复用 infer_image.py 的全部前后处理，本文件不重复实现，也不修改任何现有脚本。
部署：放到板端 /home/zsy/rk3588-yolo/scripts/，与 infer_image.py 同目录。
"""

import os
import sys
import time
import glob
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import cv2
from rknnlite.api import RKNNLite

from infer_image import (CLASSES, RESULT_DIR, letterbox, infer_once,
                         unpack_outputs, apply_box_scale, postprocess, map_back)


# ------------------------- MJPEG over HTTP -------------------------

_jpeg_lock = threading.Lock()
_jpeg_bytes = None

PAGE = b"""<!doctype html><html><head><meta charset="utf-8">
<title>RK3588 YOLOv8n</title></head>
<body style="margin:0;background:#111;color:#eee;font-family:monospace;text-align:center">
<h3>RK3588 NPU &middot; YOLOv8n video stream</h3>
<img src="/stream" style="max-width:96%;border:1px solid #444">
<p>Ctrl-C on the board stops the stream</p>
</body></html>"""


def publish(bgr):
    """把当前帧编码成 jpeg，供 HTTP 推流线程取用"""
    global _jpeg_bytes
    ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if ok:
        with _jpeg_lock:
            _jpeg_bytes = buf.tobytes()


class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *unused):
        pass

    def do_GET(self):
        if self.path.startswith('/stream'):
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            try:
                while True:
                    with _jpeg_lock:
                        jpg = _jpeg_bytes
                    if jpg is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(b'Content-Length: ' + str(len(jpg)).encode() + b'\r\n\r\n')
                    self.wfile.write(jpg)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)


def start_http(port):
    srv = ThreadingHTTPServer(('0.0.0.0', port), MJPEGHandler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ------------------------- 输入源 -------------------------

IMAGE_EXTS = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')


def open_source(src):
    if os.path.isdir(src):
        files = []
        for e in IMAGE_EXTS:
            files.extend(glob.glob(os.path.join(src, e)))
        files = sorted(set(files))
        if not files:
            print('!! 目录里没有图片:', src)
            sys.exit(1)
        print('   输入源: 图片目录，共 %d 张' % len(files))
        return 'images', files
    if src.startswith('/dev/'):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not cap.isOpened():
            print('!! 打不开 V4L2 设备:', src)
            sys.exit(1)
        print('   输入源: V4L2 摄像头', src)
        return 'camera', cap
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print('!! 打不开视频文件:', src)
        sys.exit(1)
    print('   输入源: 视频文件', src)
    return 'video', cap


def read_frame(kind, obj, idx):
    if kind == 'images':
        return cv2.imread(obj[idx]) if idx < len(obj) else None
    ok, f = obj.read()
    return f if ok else None


def rewind(kind, obj):
    if kind == 'images':
        return 0
    obj.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return 0


# ------------------------- 绘制与统计 -------------------------

def draw(frame, det, fps, infer_ms):
    for d in det:
        x1, y1, x2, y2, s, c = d
        name = CLASSES[c] if c < len(CLASSES) else str(c)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, '%s %.2f' % (name, s), (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (340, 54), (0, 0, 0), -1)
    cv2.putText(frame, 'FPS %.1f | infer %.1f ms' % (fps, infer_ms), (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'det %d | RK3588 NPU' % len(det), (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def report(times):
    if not times:
        return None
    a = np.array(times)
    return (float(a.mean()), float(np.percentile(a, 50)),
            float(np.percentile(a, 95)), float(1000.0 / a.mean()))


# ------------------------- 主流程 -------------------------

def main():
    ap = argparse.ArgumentParser(description='RK3588 YOLOv8n 视频流实时检测（无需摄像头/显示器）')
    ap.add_argument('--model', required=True)
    ap.add_argument('--source', required=True, help='视频文件 / 图片目录 / /dev/videoN')
    ap.add_argument('--out', default=os.path.join(RESULT_DIR, 'demo_output.mp4'),
                    help='带框视频，默认落到 results/')
    ap.add_argument('--http', type=int, default=0, help='MJPEG 端口，0=关闭')
    ap.add_argument('--loop', action='store_true', help='循环播放，配合 --http 做持续演示')
    ap.add_argument('--max-frames', type=int, default=200, help='非循环模式最多处理多少帧')
    ap.add_argument('--fps', type=float, default=25.0, help='输出视频帧率')
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--nms', type=float, default=0.45)
    ap.add_argument('--core', default='012', choices=['0', '1', '2', '012'])
    ap.add_argument('--layout', default='auto', choices=['auto', 'nhwc', 'nchw'])
    ap.add_argument('--box-scale', default='auto',
                    help='boxes 量程还原；head_fp16 模型插过 Div(8)，auto 会自动乘 8')
    ap.add_argument('--warmup', type=int, default=10)
    args = ap.parse_args()

    core_map = {
        '0': RKNNLite.NPU_CORE_0,
        '1': RKNNLite.NPU_CORE_1,
        '2': RKNNLite.NPU_CORE_2,
        '012': RKNNLite.NPU_CORE_0_1_2,
    }

    kind, obj = open_source(args.source)
    write_video = not (args.loop and args.http)
    if args.loop and args.http:
        print('   --loop + --http：只推流，不写 mp4（避免无限写文件）')

    print('=' * 56)
    print(' 模型 :', args.model)
    print(' core :', args.core, '| conf', args.conf, '| nms', args.nms)
    print('=' * 56)

    rknn = RKNNLite()
    if rknn.load_rknn(args.model) != 0:
        print('!! load_rknn 失败'); rknn.release(); sys.exit(1)
    if rknn.init_runtime(core_mask=core_map[args.core]) != 0:
        print('!! init_runtime 失败'); rknn.release(); sys.exit(1)
    print('   NPU 初始化成功')

    if args.http:
        start_http(args.http)
        print('   MJPEG 已启动，PC 浏览器打开 http://100.75.157.90:%d' % args.http)

    # 预热：用第一帧，让 NPU 频率爬上去
    warm = read_frame(kind, obj, 0)
    if warm is not None:
        p, _, _ = letterbox(warm, args.size)
        r = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        for _ in range(args.warmup):
            infer_once(rknn, r, layout=args.layout)
        print('   预热 %d 帧完成' % args.warmup)

    idx = 0
    frame_no = 0
    layout_key = args.layout
    writer = None
    video_path = args.out
    out_size = None
    t_infer_all, t_total_all, recent = [], [], []

    try:
        while True:
            if not args.loop and frame_no >= args.max_frames:
                break
            frame = read_frame(kind, obj, idx)
            idx += 1
            if frame is None:
                if args.loop:
                    idx = rewind(kind, obj)
                    frame = read_frame(kind, obj, idx)
                    idx += 1
                    if frame is None:
                        print('!! 无法回到输入源开头，退出')
                        break
                else:
                    break
            frame_no += 1

            t0 = time.perf_counter()
            oh, ow = frame.shape[:2]
            padded, scale, pad = letterbox(frame, args.size)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

            t1 = time.perf_counter()
            out, tag = infer_once(rknn, rgb, layout=layout_key)
            t2 = time.perf_counter()
            layout_key = tag.split()[0].lower()   # 首帧后固定输入布局，避免每帧试错

            boxes, scores = unpack_outputs(out)
            if boxes is None or scores is None:
                print('!! 认不出输出张量'); break
            boxes, _, _ = apply_box_scale(boxes, args.box_scale)
            det = postprocess(boxes, scores, args.conf, args.nms)
            det = map_back(det, scale, pad, ow, oh)
            t3 = time.perf_counter()

            t_infer = (t2 - t1) * 1000.0
            t_total = (t3 - t0) * 1000.0
            t_infer_all.append(t_infer)
            t_total_all.append(t_total)
            recent.append(t_total / 1000.0)
            if len(recent) > 30:
                recent.pop(0)
            fps = 1.0 / (sum(recent) / len(recent))

            vis = draw(frame.copy(), det, fps, t_infer)
            publish(vis)

            if write_video:
                if writer is None:
                    out_size = (vis.shape[1], vis.shape[0])
                    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
                    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                             args.fps, out_size)
                    if not writer.isOpened():
                        video_path = os.path.splitext(args.out)[0] + '.avi'
                        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'),
                                                 args.fps, out_size)
                        print('   mp4v 不可用，降级为', video_path)
                    print('   写出视频:', video_path, out_size)
                if vis.shape[1] != out_size[0] or vis.shape[0] != out_size[1]:
                    vis = cv2.resize(vis, out_size)
                writer.write(vis)

            if frame_no % 20 == 0:
                print('  frame %4d | det %2d | infer %.1f ms | e2e %.1f ms | %.1f FPS'
                      % (frame_no, len(det), t_infer, t_total, fps))
    except KeyboardInterrupt:
        print('\n   Ctrl-C，结束推流')
    finally:
        if writer is not None:
            writer.release()
            print('   视频已保存:', video_path)
        if kind in ('video', 'camera'):
            obj.release()
        rknn.release()

    print('-' * 56)
    print('   共处理 %d 帧' % frame_no)
    for name, arr in (('纯前向 rknn.inference()', t_infer_all),
                      ('端到端 含前后处理', t_total_all)):
        r = report(arr)
        if r:
            print('   %-26s 平均 %.2f ms | P50 %.2f | P95 %.2f | %.1f FPS'
                  % ((name,) + r))
    print('-' * 56)
    print('   提示：纯前向可与报告里的 33.30 ms 对比；端到端口径用于和 x86 26.5 ms 对齐（P0）')


if __name__ == '__main__':
    main()
