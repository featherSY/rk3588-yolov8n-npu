#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""split_logits.py —— 拆成 boxes +「sigmoid 之前的 logits」双输出

为什么需要：
  原 split_outputs.py 拆出的第二个输出是 Sigmoid 之后的结果（0~1）。
  RKNN 把输出张量强制量化成 int8 后，整个 0~1 只剩约 145 个可用档位，
  6.8 万个框的分数撞在 145 个值上，NMS 排序信息丢失 -> mAP 掉 11 点。

  改成输出 logits（约 -15~+15）后：
    - int8 步长 = 30/255 ≈ 0.118（logit 空间）
    - PC/板端自己做 sigmoid，分数精度由 float 计算保证
    - 排序信息保住，不再是台阶
  代价：后处理里要加一次 sigmoid。

用法（WSL rknn-env）：
  python split_logits.py \\
    --onnx models/yolov8n.onnx \\
    --new-onnx models/yolov8n_2out_logits.onnx \\
    --box-scale 8 --dry-run          # 只看图结构，不写文件

  # 确认后去掉 --dry-run，再用 convert / hybrid 脚本转 rknn
"""

import os
import sys
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--new-onnx', default='yolov8n_2out_logits.onnx')
    ap.add_argument('--box-scale', type=float, default=1.0,
                    help='boxes 分支除以该常数压缩量程（推理端要乘回）')
    ap.add_argument('--logits-input-index', type=int, default=None,
                    help='Concat 的哪个输入是 sigmoid 之后的分数（默认自动找）')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    import onnx
    from onnx import helper, TensorProto, shape_inference

    m = onnx.load(args.onnx)
    try:
        m = shape_inference.infer_shapes(m)
    except Exception as e:
        print("[warn] shape_inference:", e)

    prod = {}
    for n in m.graph.node:
        for o in n.output:
            prod[o] = n

    outs = [o.name for o in m.graph.output]
    print("[logits] 原图输出:", outs)
    target = outs[0]

    # 往上找多输入节点（Concat）
    node = prod.get(target)
    hop = 0
    while node is not None and len(node.input) < 2 and hop < 6:
        node = prod.get(node.input[0])
        hop += 1
    if node is None or len(node.input) < 2:
        print("!! 没找到多输入节点"); sys.exit(1)
    print("[logits] 合并节点: %s" % node.op_type)

    # 逐个输入往上追：找到以 Sigmoid 结尾的那条链，取它的输入
    chosen = []
    for i, src in enumerate(node.input):
        p = prod.get(src)
        trace = []
        cur, curn = src, p
        while curn is not None and len(trace) < 5:
            trace.append('%s(%s)' % (curn.op_type, cur))
            if len(curn.input) != 1:
                break
            cur = curn.input[0]
            curn = prod.get(cur)
        print("[logits]   输入[%d] %-45s 回溯链: %s"
              % (i, src, ' <- '.join(trace)))

        if p is not None and p.op_type == 'Sigmoid':
            logits = p.input[0]
            print("[logits]      ^ 这是 sigmoid 输出，改用它的输入（logits）: %s" % logits)
            chosen.append(logits)
        else:
            chosen.append(src)

    if chosen[1] == node.input[1]:
        print("!! 没找到 sigmoid 节点，第二路仍是 sigmoid 输出，logits 方案不适用")
        sys.exit(1)

    dims_map = {}
    for vi in list(m.graph.value_info) + list(m.graph.output):
        if vi.name in chosen:
            dims_map[vi.name] = [d.dim_value if d.HasField('dim_value') else d.dim_param
                                 for d in vi.type.tensor_type.shape.dim]

    print("\n[logits] 将要输出的两个张量:")
    print("   yolo_out_0 (boxes) : %s" % chosen[0])
    print("   yolo_out_1 (logits): %s  dims=%s" % (chosen[1], dims_map.get(chosen[1])))

    if args.dry_run:
        print("\n--dry-run：未写文件。确认无误后去掉 --dry-run。")
        return

    import numpy as np
    from onnx import numpy_helper

    if args.box_scale != 1.0:
        m.graph.initializer.append(numpy_helper.from_array(
            np.array([args.box_scale], dtype=np.float32), name='box_scale_const'))
        m.graph.node.append(helper.make_node(
            'Div', [chosen[0], 'box_scale_const'], ['boxes_scaled'],
            name='BoxScaleDiv'))
        chosen[0] = 'boxes_scaled'
        print("[logits] boxes 分支插入 Div(%.6g)，推理后要乘回" % args.box_scale)

    for i, src in enumerate(chosen):
        nm = 'yolo_out_%d' % i
        dims = dims_map.get(src)
        m.graph.output.append(
            helper.make_tensor_value_info(nm, TensorProto.FLOAT, dims or [1, 0, 8400]))
        m.graph.node.append(helper.make_node('Identity', [src], [nm],
                                             name='Identity_out_%d' % i))
        print("[logits] 新增输出 %s <- %s" % (nm, src))

    onnx.checker.check_model(m)
    onnx.save(m, args.new_onnx)
    print("\n✅ 已保存: %s (%.2f MB)"
          % (args.new_onnx, os.path.getsize(args.new_onnx) / 1048576.0))
    print("   后续：用 convert_w16.py 或 hybrid_quant.py 转 rknn；")
    print("   推理端 postprocess 需先 sigmoid(logits) 再取 max 作为置信度。")


if __name__ == '__main__':
    main()
