# RK3588 YOLOv8n NPU 部署 · 最终评测报告

> 结论：**INT8 主干 + FP16 检测头的混合量化方案，把 INT8 的 mAP 掉点从 10.6 点收回到 1.68 点（mAP@0.5:0.95），达到 <2 点目标；延迟 33.30 ms / 30.0 FPS，仅比全 INT8 慢 4.62 ms，速度优势保持。** 定为默认部署配置。  
> 误差定位结论：INT8 掉点几乎全部来自 `model.22` 检测头内部，其中 **cv2 + DFL 框回归分支**是主要来源；分数输出的 1/255 台阶不是主因（head FP16 输出仍带台阶，精度却几乎无损）。

---

## 1. 测试条件与口径

| 项目    | 设置                                                                              |
| ----- | ------------------------------------------------------------------------------- |
| 数据集   | COCO val2017 子集，按文件名排序取前 **500 张**（各方案同一批图）                                     |
| 标注    | `instances_val2017.json`                                                        |
| 评估口径  | conf=0.001、nms=0.7、max-dets=300、COCO 101-point 插值 AP                            |
| 预处理   | letterbox 640×640，灰边 114，RGB，NHWC uint8                                         |
| 输入归一化 | RKNN config `mean=0 / std=255`；x86 ORT 输入 float32 /255                          |
| 速度口径  | NPU 锁 1 GHz（userspace）+ CPU performance，warmup 10 + 200 帧，仅计 `rknn.inference()` |
| 工具链   | rknn-toolkit2 / rknn-toolkit-lite2 **2.3.2**，driver 0.9.8，RKNPU v2              |
| 环境    | 转换与 ORT 评估在 WSL `~/rknn-env`（Python 3.12）；推理评估在 RK3588 板端                       |

> 注意：x86 26.5 ms 是**端到端**（含前后处理），板端 28.68/46.75 ms 是**纯网络前向**，两者不可直接相减比较；本报告只在同一口径内部比较板端各方案。

---

## 2. 精度结果（500 张，conf001 / nms0.7 / max-dets300）

| # | 方案                               | 模型体积        | mAP@0.5      | mAP@0.5:0.95 | 相对 FP32 掉点              |
| - | -------------------------------- | ----------- | ------------ | ------------ | ----------------------- |
| 1 | x86 ORT **FP32 基线**              | 12.26 MB    | **0.5423**   | **0.3913**   | —                       |
| 2 | RK3588 全 **FP16**                | —           | 0.5417       | 0.3909       | −0.0006 / −0.0004（≈无损）  |
| 3 | RK3588 全 **INT8**                | 4.74 MB     | 0.4599       | 0.2855       | −0.0824 / **−0.1058**   |
| 4 | INT8 + boxes Div(8) 压量程          | 4.74 MB     | 0.4629       | 0.2788       | −0.0794 / −0.1125       |
| 5 | 混合量化：仅 **cv3** FP16              | 5.25 MB     | 0.4626       | 0.2806       | −0.0797 / −0.1107       |
| 6 | INT8 + **logits 输出**             | —           | 0.4629       | 0.2787       | −0.0794 / −0.1126       |
| 7 | 输出张量设 FP16                       | 4.83 MB     | 未评估（体检证明未生效） | —            | —                       |
| 8 | **混合量化：整个 `model.22` head FP16** | **5.67 MB** | **0.5299**   | **0.3745**   | **−0.0124 / −0.0168** ✅ |

**达标判定**：方案 8 相对基线掉点 1.24 点（mAP50）/ 1.68 点（mAP50-95），**均 <2 点**。相对全 INT8 回血 **+7.00 / +8.90 点**。

补充口径：conf=0.25 下全 INT8 为 0.4195 / 0.2504，掉点 −0.0931；该口径掉点更大，说明低置信度区间受量化影响更重。

### 方案 8 的诊断细节（对比 FP32）

| 指标                  | FP32 基线                  | head FP16                | 变化         |
| ------------------- | ------------------------ | ------------------------ | ---------- |
| 每图平均框数              | 152.3                    | 114.8                    | −24.6%     |
| 框总数                 | 76 130                   | 57 424                   | −24.6%     |
| 置信度 P10 / P50 / P90 | 0.0013 / 0.0041 / 0.0532 | 0.0039 / 0.0077 / 0.0810 | 低分段被抬高     |
| 框面积中位数              | 1488 px²                 | 1315 px²                 | **−11.6%** |

- 掉点最多的 5 类：`snowboard −0.1479`、`sandwich −0.0869`、`mouse −0.0839`、`parking meter −0.0756`、`sports ball −0.0722` —— 均为小目标 / 细长物体，对框回归精度最敏感。
- 反超的 3 类：`bed +0.0172`、`cat +0.0207`、`fork +0.0447`。
- 80 类平均掉点 −0.0168，与总 mAP 一致，说明是**全局性、非个例**的精度恢复。
- `P10 = 0.0039` 恰好等于 `1/255`：输出层仍是 INT8 台阶，低分预测被台阶吞掉，这解释了框数减少 24.6%。

---

## 3. 速度结果

| 方案                   | core | 平均延迟         | FPS      | 相对 x86 FP32 (26.5 ms) |
| -------------------- | ---- | ------------ | -------- | --------------------- |
| x86 ORT FP32（端到端基线）  | —    | 26.5 ms      | 37.8     | 1.00×                 |
| RK3588 全 FP16        | 012  | 46.75 ms     | 21.4     | —                     |
| RK3588 全 INT8 (2out) | 012  | **28.68 ms** | 34.9     | —                     |
| RK3588 head FP16     | 012  | **33.30 ms** | **30.0** | —                     |

|                                |           |           |                                                                                   |
| ------------------------------ | --------- | --------- | --------------------------------------------------------------------------------- |
| head FP16 实测明细：\*\*平均 33.30 ms | P50 31.80 | P95 41.59 | 最快 27.76\*\*（warmup 10 + 200 帧，仅 `rknn.inference()`，NPU 1 GHz + CPU performance）。 |

**"保持速度优势"判定：达标。** 只把检测头（18 个卷积）转 FP16、主干仍 INT8，代价 **+4.62 ms（+16.1%）**，换来 **+7.00 / +8.90 点 mAP**；相对全 FP16 仍快 **13.45 ms（−28.8%）**。以每毫秒换回的精度计，这是本轮性价比最高的方案。

---

## 4. 误差定位：四个假设的完整验证链

掉点 10.6 点远超 YOLOv8n INT8 的正常水平（1~3 点），因此逐层排除：

| 假设                          | 干预手段                                    | 结果                                     | 判定       |
| --------------------------- | --------------------------------------- | -------------------------------------- | -------- |
| ① 框坐标输出量程过粗（0~640 压进 256 档） | boxes 分支插 `Div(8)` 压缩量程，评估时乘回           | 0.4629 / 0.2788                        | ❌ 证伪     |
| ② cv3 分类分支权重有离群值            | 混合量化，仅 15 个 cv3 张量转 FP16                | 0.4626 / 0.2806                        | ❌ 证伪     |
| ③ 分数 1/255 台阶破坏 NMS 排序      | (a) 出口改 sigmoid 前 logits；(b) 输出张量设 FP16 | (a) 0.4629 / 0.2787；(b) 台阶吻合 0.999，未生效 | ❌ 证伪     |
| ④ **误差在 cv2 + DFL 框回归分支内部** | 整个 `model.22` 的 51 个合法张量转 FP16          | **0.5299 / 0.3745**                    | ✅ **成立** |

**关键反证**：方案 8 的输出层仍是 INT8（`P10=0.0039=1/255`、台阶吻合度 0.999），分数台阶完全没变，mAP 却几乎回到 FP32。这直接排除了"分数台阶是主因"，也解释了为什么 logits 路线无效。

**量化台阶的数学根源**：`yolo_out_1` 的 scale = `0.9833/255 = 0.0038561`，量程 [0, 0.983]，256 档，分数分辨率锁死为 1/255。FP32 有 66 702 个不同分数，INT8 只有 147 个且全部落在台阶上。这是 RK3588 输出层 INT8 的固有属性，**改 cfg 无效**。

**误差在框分支的物理表现**：框面积中位数系统性偏小 11.6%，小目标类掉点最重。`DFL` 的 softmax + 加权求和在 INT8 下累积误差，直接扭曲框的位置与尺寸；把 head 转 FP16 后该累积误差消失。

---

## 5. 关键工程结论（踩过的坑，可复用）

1. **venv 隔离**：`rknn_toolkit2==2.3.2` 只装在 `~/rknn-env`，WSL 下必须先 `source ~/rknn-env/bin/activate`，用 `python` 而非 `python3`。
2. **校准集路径**：rknn 对 dataset txt 中的相对路径按 `$HOME` 解析而非 CWD，必须写 `os.path.abspath()`。
3. **dtype 支持**：RK3588 后端只真正支持 **FP16 / INT8**。`w16a16i` 等 int16 是 API 占位，会 `Lut fuse dtype not support` + `Illegal instruction (core dumped)`；`w8a16` 仅 RK3566/3568 支持。
4. **混合量化 cfg 认"张量名"不认"节点名"**：`custom_quantize_layers` 必须是 `/model.22/cv3.0/.../Conv_output_0` 这类输出张量名，写 `/model.22/cv3.0/.../Conv` 会报 `Invalid operands name`。唯一可靠来源是 cfg 自身的 `quantize_parameters` 段。
5. **非法 operand**：`Constant_*`（权重常量）、`images`（网络输入）、`*_int8`（量化变体）、`*_2mul_i1`（二元 op 常量操作数）都不能进 `custom_quantize_layers`，否则 step2 直接拒收。
6. **输出层 dtype 改不动**：`custom_quantize_layers` 只对**中间层**有效；模型输出张量的 dtype 在 `build` 阶段固化，`hybrid_quantization_step2` 改不了顶层输出。设置后仍输出 `float32` 且台阶吻合 0.999。
   |   |   |   |
   | - | - | - |
7. **判断量化是否生效看台阶吻合度，不看 dtype**：runtime 输出永远是反量化后的 `float32`；用 `|v×255 − round(v×255)| < 0.02` 的占比判定是否仍在 1/255 网格上。
8. **体积判据**：`.rknn` 体积只由**有权重的层**决定，DFL/Concat/Reshape 等无权重层转 FP16 不涨体积。4.74 MB（全 INT8）→ 5.25 MB（cv3 FP16）→ **5.67 MB（cv2+cv3 全 FP16）** 才是正确标定。
9. **`--box-scale 8`**：boxes 分支插了 `Div(8)`（输出 max≈80），评估必须乘回 8，否则框坐标缩成 1/8、mAP 崩到接近 0。**CLI 可视化同理**：`eval_map.py` 带了该参数所以 mAP 正常（0.5299），但 `infer_image.py` 原先没带，0~80 的坐标被 letterbox 的 `top=80` padding 减成负数再 clamp 到 0，所有框退化成 `y1=y2=0` 的贴顶细线 → 检测到 7 个目标却画不出框。**评估脚本与推理脚本的坐标还原必须同步**，现已给 `infer_image.py` 加 `--box-scale auto`（按 `max<120` 自动乘 8）。
10. **多余输出**：拆分后的模型带 3 个输出，其中 `[1,84,8400]` 是未使用的原始 concat 输出，每次推理白搬约 2.8 MB，可在收尾优化时删除。
11. **模拟器限制**：PC 模拟器不认 `core_mask`，`init_runtime()` 不要传参；`load_rknn` 后无法在 PC 直接推理，输出体检必须上板。
12. **step1 产物位置**：`hybrid_quantization_step1` 把 `.model/.data/.quantization.cfg` 写在 CWD，复用时可用 `--skip-step1`。
13. **step2 签名**：2.3.2 实际为 `hybrid_quantization_step2(model_input, data_input, model_quantization_cfg)`，无 `model_script` 参数；`The node ... is specified repeatedly` 是 WARN，可忽略。

---

## 6. 部署选型建议

| 场景                  | 推荐方案                    | 精度                                   | 速度                      | 体积          |
| ------------------- | ----------------------- | ------------------------------------ | ----------------------- | ----------- |
| **精度 + 速度均衡（默认推荐）** | **INT8 主干 + FP16 head** | **0.5299 / 0.3745**（掉 1.24 / 1.68 点） | **33.30 ms / 30.0 FPS** | **5.67 MB** |
| 精度绝对优先              | 全 FP16                  | 0.5417 / 0.3909（≈无损）                 | 46.75 ms / 21.4 FPS     | 较大          |
| 速度绝对优先、可接受漏检        | 全 INT8                  | 0.4599 / 0.2855（掉 8.2 / 10.6 点）      | 28.68 ms / 34.9 FPS     | 4.74 MB     |

**实测 33.30 ms ≤ 35 ms 阈值，方案 8 定为默认部署配置**：相比全 INT8 多花 4.62 ms 即把掉点从 10.6 点压到 1.68 点，且比全 FP16 快 28.8%。若业务对 33 ms 仍敏感，可退化为全 INT8；若要求精度无损，改用全 FP16。

---

## 7. 复现命令（全部真实路径，可直接复制）

### 7.1 转换（WSL）

```bash
cd /mnt/d/01_STUDY/rk3588
source ~/rknn-env/bin/activate
python scripts/fix_hybrid.py --cfg yolov8n_2out_bs8.quantization.cfg \
  --pattern model.22 --out models/yolov8n_head_fp16.rknn
```

### 7.2 精度评估（板端 → WSL 对比）

```bash
sudo LD_LIBRARY_PATH=/home/zsy/rk3588-yolo/lib python3 /home/zsy/rk3588-yolo/scripts/eval_map.py --backend rknn --model /home/zsy/rk3588-yolo/models/yolov8n_head_fp16.rknn --img-dir /home/zsy/rk3588-yolo/coco/val2017 --ann /home/zsy/rk3588-yolo/coco/annotations/instances_val2017.json --limit 500 --conf 0.001 --nms 0.7 --max-dets 300 --box-scale 8 --out /home/zsy/rk3588-yolo/results/preds_head_fp16_500.json
```

```powershell
scp zsy@100.75.157.90:/home/zsy/rk3588-yolo/results/preds_head_fp16_500.json D:\01_STUDY\rk3588\results\
```

```bash
cd /mnt/d/01_STUDY/rk3588 && python downloads/scripts/compare_map.py results/preds_ort_fp32_500.json results/preds_head_fp16_500.json
```

### 7.3 速度实测（板端，已得 33.30 ms / 30.0 FPS）

```bash
echo userspace | sudo tee /sys/class/devfreq/fdab0000.npu/governor
echo 1000000000 | sudo tee /sys/class/devfreq/fdab0000.npu/userspace/set_freq
sudo sh -c 'for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo performance > $f; done'
sudo LD_LIBRARY_PATH=/home/zsy/rk3588-yolo/lib python3 /home/zsy/rk3588-yolo/scripts/infer_image.py --model /home/zsy/rk3588-yolo/models/yolov8n_head_fp16.rknn --img /home/zsy/rk3588-yolo/coco/val2017/000000001532.jpg --repeat 200 --core 012
```

### 7.4 输出层体检（板端）

```bash
sudo LD_LIBRARY_PATH=/home/zsy/rk3588-yolo/lib python3 /home/zsy/rk3588-yolo/scripts/diag_outputs.py --model /home/zsy/rk3588-yolo/models/yolov8n_out_fp16.rknn --img /home/zsy/rk3588-yolo/coco/val2017/000000001532.jpg
```

---

## 8. 输出目录约定

**两边各有一个 `results/`，各管各的，不要混：**


| 环境 | 目录 | 存什么 | 怎么指定 |
|---|---|---|---|
| 板端 | `~/rk3588-yolo/results/` | 结果图 `result*.jpg`、预测 json | **结果图**：`infer_image.py --out` 不写时默认进这里；预测 json 一直用显式 `--out results/xxx.json` |
| 本机 PC | `D:\01_STUDY\rk3588\results\` | FP32 基线 json、对比 md、从板端回传的 preds | 一直在用，命令里都显式写 `results/xxx.json`，**脚本未改动** |

唯一的代码改动：板端 `infer_image.py` 的 `--out` 默认值从 `result.jpg`（落在当前目录）改成 `<项目根>/results/result.jpg`，并自动 `makedirs`。之前测速时生成的 `result.jpg` / `result_fixed.jpg` 落在 `~` 下，需手动移一次。

`eval_map.py`、`compare_map.py` **保持原样**——它们一直靠显式 `--out results/...` 和 `--md results/...` 指定路径，不需要改默认值。转换产物 `.rknn` 归 `models/`。

已完成项：head FP16 延迟实测（33.30 ms / 30.0 FPS，速度优势达标）、板端 CLI 画框异常（`--box-scale auto`）、板端结果图默认落 `results/`。

---

## 9. 遗留项与后续路线

### P0 · 先把基线口径对齐（唯一影响结论可信度的事）

**当前 26.5 ms 与 33.30 ms 不可直接比较**：x86 26.5 ms 是 `benchmark.py` 的**端到端**（读图+预处理+前向+NMS，100 帧平均 33.7 ms 的那套口径里 26.5 ms 是热稳态值），而板端 33.30 ms 只计 `rknn.inference()`。报告 §3 已用「—」回避了对比列，但结论里若写"NPU 比 x86 快/慢"就不严谨。

要做的：在 x86 上用 `eval_map.py` 同款前后处理跑 500 张，同时单独计时纯前向，得到与板端同口径的 x86 数字。这样才能回答项目最初的问题「同一 YOLOv8n，x86 与 RK3588 差多少」。

### P1 · 删掉多余的 84 通道输出（直接再提速）

模型带 3 个输出，其中 `[1,84,8400]` 是未使用的原始 concat 输出，每帧白搬约 **2.8 MB**。在 ONNX 里把 `output0` 从图输出中摘掉再重转，预计能让 33.30 ms 再降几毫秒，且体积同步缩小。**这是目前唯一还能无损提速的点**，成本只是重跑一次 `split_outputs.py` + `fix_hybrid.py`。

### P2 · 全量 5000 张复评（把结论从「500 张子集」升级成「完整 val2017」）

500 张约 17 s，5000 张约 3 分钟，成本极低。全量结果可以直接写进简历/报告标题，置信度完全不同。

### P3 · 工程化收尾

- 摄像头实时 demo（GStreamer / v4l2 + 板端 CLI 串起来）
- 一键复现脚本：把 §7 的 4 段命令打包成 `run_all.sh` + PowerShell 推送脚本
- 把 NPU/CPU 锁频写成开机服务（当前重启即失效）

**建议顺序**：P0 → P2（都是纯评测、不改模型、十几分钟）→ P1（要重转模型，收益是速度）→ P3。若只想收尾交差，做 P0 + P2 就够让报告从"内部记录"变成"可对外引用"。
