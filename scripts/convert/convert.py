# -*- coding: utf-8 -*-
"""
YOLOv8 ONNX -> RKNN (INT8) 转换脚本

用法示例：
    python convert.py --onnx /path/to/yolov8n.onnx --dataset /path/to/calib_images \
                      --out yolov8n_int8.rknn --size 640 --num 200

参数说明：
    --onnx      输入 ONNX 路径（需为无固定 batch 动态维度或已固定 1x3xHxW）
    --dataset   校准集图片目录（放 100~300 张有代表性的图）
    --out       输出 .rknn 路径
    --size      输入边长，YOLOv8 通常用 640
    --num       校准用图片数量，建议 100~300
    --dtype     int8（默认，w8a8）/ w8a16 / fp16 / fp32
    --mean/std  归一化参数，YOLOv8 常用 mean=0 std=255
"""

import os
import sys
import argparse
import numpy as np
import cv2
from rknn.api import RKNN

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def collect_images(d):
    fs = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p) and name.lower().endswith(IMG_EXTS):
            fs.append(p)
    return fs


def make_dataset_txt(files, out_path):
    with open(out_path, 'w') as f:
        for p in files:
            f.write(p + '\n')
    return out_path


def probe_onnx(onnx_path):
    """打印 ONNX 的输入输出信息，用来确认是否含 NMS、输出头布局"""
    try:
        import onnx
        m = onnx.load(onnx_path)
        print("[onnx] 输入:")
        for i in m.graph.input:
            shape = [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
            print("   ", i.name, shape)
        print("[onnx] 输出:")
        for o in m.graph.output:
            shape = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
            print("   ", o.name, shape)
    except Exception as e:
        print("[onnx] 无法解析（不影响转换）:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True, help='输入 ONNX 路径')
    ap.add_argument('--dataset', default='.', help='校准集图片目录（仅 INT8 需要）')
    ap.add_argument('--out', default='yolov8n_int8.rknn', help='输出 rknn 路径')
    ap.add_argument('--size', type=int, default=640, help='输入边长')
    ap.add_argument('--num', type=int, default=200, help='校准图片数')
    ap.add_argument('--dtype', default='int8', choices=['int8', 'w8a16', 'fp16', 'fp32'],
                    help='int8=权重+激活都量化(w8a8)；w8a16=仅权重量化、激活保 fp16')
    ap.add_argument('--algo', default='normal', choices=['normal', 'mmse', 'kl_divergence'],
                    help='量化算法，int8 结果不对时可试 mmse')
    ap.add_argument('--opt', type=int, default=3, help='optimization_level 0~3，出问题可降到 2')
    ap.add_argument('--mean', default='0 0 0', help='mean_values，空格分隔')
    ap.add_argument('--std', default='255 255 255', help='std_values，空格分隔')
    ap.add_argument('--txt', default='calib_list.txt', help='临时校准清单文件')
    args = ap.parse_args()

    mean = [[float(x) for x in args.mean.split()]]
    std = [[float(x) for x in args.std.split()]]

    print("=" * 50)
    print(" ONNX -> RKNN 转换")
    print("   输入     :", args.onnx)
    print("   校准集   :", args.dataset)
    print("   输出     :", args.out)
    print("   尺寸     :", args.size)
    print("   量化类型 :", args.dtype)
    print("=" * 50)

    if not os.path.isfile(args.onnx):
        print("!! ONNX 文件不存在:", args.onnx)
        sys.exit(1)

    probe_onnx(args.onnx)

    # int8 和 w8a16 都属于「量化」，需要校准集 + build(do_quantization=True)
    do_quant = args.dtype in ('int8', 'w8a16')

    # 只有 INT8 需要校准集，fp16/fp32 不需要
    if do_quant:
        files = collect_images(args.dataset)
        if not files:
            print("!! 校准目录里没有图片:", args.dataset)
            sys.exit(1)
        if len(files) < args.num:
            print("[warn] 校准集只有 %d 张，少于请求的 %d 张，全部使用" % (len(files), args.num))
        files = files[:args.num]
        print("[dataset] 使用 %d 张校准图" % len(files))
        make_dataset_txt(files, args.txt)
    else:
        args.txt = None
        print("[dataset] 非量化模式，跳过校准集")

    rknn = RKNN(verbose=True)

    print("[1/4] config ...")
    # 注意：quantized_dtype 只在量化时传，非量化传 None 会报
    #   Invalid quantized_dtype 'None'
    cfg = dict(
        mean_values=mean,
        std_values=std,
        target_platform='rk3588',
        optimization_level=args.opt,
    )
    if do_quant:
        # w8a8  = 权重 8bit / 激活 8bit（标准 INT8，最快）
        # w8a16 = 权重 8bit / 激活 16bit（精度更稳，分类头不易塌）
        cfg['quantized_dtype'] = 'w8a8' if args.dtype == 'int8' else 'w8a16'
        cfg['quantized_algorithm'] = args.algo
    print("   config:", cfg)
    rknn.config(**cfg)

    print("[2/4] load_onnx ...")
    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print("!! load_onnx 失败，ret =", ret)
        rknn.release()
        sys.exit(1)

    print("[3/4] build（量化 %s，这一步最慢，可能几分钟）..." % ('开' if do_quant else '关'))
    ret = rknn.build(do_quantization=do_quant, dataset=args.txt, rknn_batch_size=1)
    if ret != 0:
        print("!! build 失败，ret =", ret)
        rknn.release()
        sys.exit(1)

    print("[4/4] export ...")
    ret = rknn.export_rknn(args.out)
    if ret != 0:
        print("!! export 失败，ret =", ret)
        rknn.release()
        sys.exit(1)

    size_mb = os.path.getsize(args.out) / 1024.0 / 1024.0
    print("=" * 50)
    print(" 转换完成: %s  (%.2f MB)" % (args.out, size_mb))
    print(" 下一步：把该文件传到板子 /home/zsy/rk3588-yolo/models/")
    print("=" * 50)

    rknn.release()


if __name__ == '__main__':
    main()
