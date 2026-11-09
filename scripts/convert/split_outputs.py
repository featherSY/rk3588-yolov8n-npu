# -*- coding: utf-8 -*-
"""
Plan C：把 YOLOv8 head 的最终输出拆成「两个独立的图输出」，再转 INT8。

背景：RKNN 会把整个输出张量按统一量程做 int8 量化。YOLOv8 的输出是
      box(0~641) 和 cls(0~1 量级) 拼在一起的 [1,84,8400]，量程被 box 主导，
      步长约 2.5，导致所有类别分被抹成 0。

解决办法：不让它们合并。在最终 Concat 之前各自接一个 Identity 作为图输出，
      这样 RKNN 会给两个张量各自算量程，cls 的步长约为 0.08。

用法：
    python split_outputs.py --onnx ~/yolov8n.onnx \
        --new-onnx ~/yolov8n_2out.onnx --dataset ~/calib \
        --img ~/calib/000000000009.jpg --size 640 \
        --out ~/rknn_out/yolov8n_2out_int8.rknn
"""

import os
import sys
import argparse

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def collect_images(d, num):
    return [os.path.join(d, n) for n in sorted(os.listdir(d))
            if os.path.isfile(os.path.join(d, n)) and n.lower().endswith(IMG_EXTS)][:num]


def letterbox(img, size=640, color=(114, 114, 114)):
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nw, nh = int(round(w * s)), int(round(h * s))
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    left = (size - nw) // 2
    return cv2.copyMakeBorder(im, top, size - nh - top, left, size - nw - left,
                              cv2.BORDER_CONSTANT, value=color)


def probe_shapes(onnx_path, names):
    """用 onnxruntime 实测新增输出的真实形状（比静态推断可靠）"""
    try:
        import numpy as np
        import onnxruntime as ort
    except Exception:
        return None
    try:
        s = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print("[warn] onnxruntime 打不开:", e)
        return None
    feed = {}
    for i in s.get_inputs():
        shp = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        feed[i.name] = np.zeros(shp, dtype=np.float32)
    try:
        outs = s.run(None, feed)
    except Exception as e:
        print("[warn] ORT 前向失败:", e)
        return None
    got = {}
    for i, o in enumerate(s.get_outputs()):
        if o.name in names:
            got[o.name] = list(outs[i].shape)
    return got


def split_graph(onnx_path, out_path, use_identity=True, box_scale=1.0):
    """把最终输出拆成两个：boxes[1,4,8400] 和 scores[1,80,8400]

    box_scale: 给 boxes 分支先除一个常数再输出。
        boxes 值域 0~641，INT8 只有 256 档 -> 步长约 5px（640 尺度），
        小目标的框会被这一步长带偏。除以 N 后量程变 641/N，步长同比变细，
        推理时把 boxes 乘回 N 即可。不改变网络任何计算，只改量化量程。
    """
    import onnx
    from onnx import helper, TensorProto, shape_inference, numpy_helper
    import numpy as _np

    m = onnx.load(onnx_path)
    try:
        m = shape_inference.infer_shapes(m)
    except Exception as e:
        print("[warn] shape_inference 失败（不影响）:", e)

    info = {}
    for vi in list(m.graph.input) + list(m.graph.value_info) + list(m.graph.output):
        info[vi.name] = vi

    prod = {}
    for n in m.graph.node:
        for o in n.output:
            prod[o] = n

    outs = [o.name for o in m.graph.output]
    print("[split] 原图输出:", outs)
    target = outs[0]

    # 往上找，直到碰到一个有多个输入的节点（Concat）
    node = prod.get(target)
    hop = 0
    while node is not None and len(node.input) < 2 and hop < 6:
        node = prod.get(node.input[0])
        hop += 1
    if node is None or len(node.input) < 2:
        print("!! 没找到多输入节点，无法拆分")
        sys.exit(1)
    print("[split] 合并节点: %s  输入: %s" % (node.op_type, list(node.input)))

    def default_dims(tag, vi_src):
        """给某个张量拼一组合法的维度：能静态推断就用实数，否则用符号名"""
        if vi_src is None or not vi_src.type.HasField('tensor_type'):
            return None
        dims = []
        for k, d in enumerate(vi_src.type.tensor_type.shape.dim):
            if d.HasField('dim_value') and d.dim_value > 0:
                dims.append(d.dim_value)
            elif d.dim_param:
                dims.append('%s_%s' % (tag, d.dim_param))
            else:
                dims.append('%s_d%d' % (tag, k))
        return dims

    added = []
    scale_added = False
    for i, src in enumerate(node.input):
        name = 'yolo_out_%d' % i
        cur = src
        # 第 0 个输入是 boxes（/model.22/Mul_2_output_0），第 1 个是 scores
        if box_scale != 1.0 and i == 0:
            if not scale_added:
                m.graph.initializer.append(numpy_helper.from_array(
                    _np.array([box_scale], dtype=_np.float32), name='box_scale_const'))
                scale_added = True
            cur = 'boxes_scaled'
            m.graph.node.append(helper.make_node(
                'Div', [src, 'box_scale_const'], [cur], name='BoxScaleDiv'))
            print("[split] boxes 分支插入 Div(%.6g)：量程 ~%.0f -> ~%.1f，"
                  "量化步长同比细 %.6gx" % (box_scale, 641.0, 641.0 / box_scale, box_scale))
            print("[split]   !! 推理后必须把 boxes 乘回 %.6g" % box_scale)
        dims = default_dims(name, info.get(cur))
        new_vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
        if use_identity:
            m.graph.node.append(helper.make_node('Identity', [cur], [name],
                                                 name='Identity_out_%d' % i))
        m.graph.output.append(new_vi)
        added.append(name)
        print("[split] 新增输出: %s  静态维度=%s" % (name, dims))

    tmp = out_path + '.tmp.onnx'
    onnx.save(m, tmp)

    # 用 ORT 实测去校准形状
    real = probe_shapes(tmp, set(added))
    m2 = onnx.load(tmp)
    if real:
        for o in m2.graph.output:
            if o.name in real:
                new = helper.make_tensor_value_info(o.name, TensorProto.FLOAT, real[o.name])
                m2.graph.output.remove(o)
                m2.graph.output.append(new)
                print("[split] 形状修正 %s -> %s" % (o.name, real[o.name]))
    else:
        print("[warn] 没能实测形状，保持动态维度（RKNN 一般能自己推出来）")

    try:
        onnx.checker.check_model(m2)
        print("[split] checker 通过")
    except Exception as e:
        print("[warn] checker 未通过（可继续）:", e)

    onnx.save(m2, out_path)
    print("[split] 已保存: %s (%.2f MB)" % (out_path, os.path.getsize(out_path) / 1048576.0))
    return out_path


def health(outs):
    """对多输出做体检：按通道数挑出 boxes(4) 和 scores(80)"""
    print("\n[health] 输出 %d 个张量:" % len(outs))
    res = {}
    for k, o in enumerate(outs):
        a = np.array(o)
        print("  [%d] shape=%s dtype=%s" % (k, a.shape, a.dtype))
        res[k] = a

    boxes = None
    scores = None
    for k, a in res.items():
        if a.ndim != 3:
            continue
        c = a.shape[1]
        if c == 4:
            boxes = a[0]
        elif c in (80, 84, 85):
            scores = a[0]
    if boxes is None or scores is None:
        print("  ！！ 没同时找到 4 通道和 80 通道的输出")
        return False

    cmax = scores.max(axis=0)
    prob = 1.0 / (1.0 + np.exp(-cmax))
    uniq = int(np.unique(np.round(boxes.reshape(-1), 4)).size)
    print("  boxes : mean=%.2f max=%.2f  unique=%d/%d"
          % (boxes.mean(), boxes.max(), uniq, boxes.reshape(-1).size))
    print("  scores: max=%.4f mean=%.4f  min=%.4f"
          % (cmax.max(), cmax.mean(), cmax.min()))
    print("  sigmoid>0.25 的 anchor: %d / %d"
          % (int((prob > 0.25).sum()), prob.size))
    ok = cmax.max() > 1e-6
    print("  判定: %s" % ("✅ 分类分回来了" if ok else "❌ 仍是 0"))
    return ok


def main():
    global np, cv2
    import numpy as np
    import cv2
    from rknn.api import RKNN

    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--new-onnx', default='yolov8n_2out.onnx')
    ap.add_argument('--dataset', default='.')
    ap.add_argument('--img', required=True)
    ap.add_argument('--size', type=int, default=640)
    ap.add_argument('--num', type=int, default=200)
    ap.add_argument('--out', default='yolov8n_2out_int8.rknn')
    ap.add_argument('--no-identity', action='store_true',
                    help='不插 Identity 节点（若 RKNN 报不支持时试这个）')
    ap.add_argument('--box-scale', type=float, default=1.0,
                    help='boxes 分支先除这个常数再输出，用于压缩 INT8 量程、'
                         '细化量化步长（如 8）。推理端要用 --box-scale 乘回来')
    args = ap.parse_args()

    for f in (args.onnx, args.img):
        if not os.path.isfile(f):
            print("!! 不存在:", f); sys.exit(1)

    print("[1/3] 拆分 ONNX 图 ...")
    new_onnx = split_graph(args.onnx, args.new_onnx, not args.no_identity,
                           box_scale=args.box_scale)

    img = cv2.imread(args.img)
    inp = cv2.cvtColor(letterbox(img, args.size), cv2.COLOR_BGR2RGB)[None].astype(np.uint8)

    files = collect_images(args.dataset, args.num)
    if not files:
        print("!! 校准目录没图:", args.dataset); sys.exit(1)
    txt = os.path.expanduser('~/calib_list2.txt')
    open(txt, 'w').write('\n'.join(files) + '\n')
    print("校准集: %d 张" % len(files))

    print("[2/3] 转 INT8 ...")
    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]],
                target_platform='rk3588', quantized_dtype='w8a8',
                quantized_algorithm='normal')
    if rknn.load_onnx(model=new_onnx) != 0:
        print("!! load_onnx 失败"); rknn.release(); sys.exit(1)
    if rknn.build(do_quantization=True, dataset=txt, rknn_batch_size=1) != 0:
        print("!! build 失败"); rknn.release(); sys.exit(1)

    print("[3/3] init_runtime + 验证 ...")
    if rknn.init_runtime() != 0:
        print("!! init_runtime 失败"); rknn.release(); sys.exit(1)
    outs = rknn.inference(inputs=[inp])
    ok = health(outs)

    if ok:
        if rknn.export_rknn(args.out) == 0:
            print("\n✅ 导出成功: %s (%.2f MB)" % (args.out, os.path.getsize(args.out) / 1048576.0))
        else:
            print("\n!! 导出失败")
    else:
        print("\n❌ 失败。试 --no-identity 再跑一次")
    rknn.release()


if __name__ == '__main__':
    main()
