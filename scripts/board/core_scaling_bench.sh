#!/usr/bin/env bash
# ============================================================
#  RK3588 板端：NPU 调度扩展性 + 推理中负载采样（一次跑完）
#
#  用法：sudo bash /home/zsy/rk3588-yolo/scripts/core_scaling_bench.sh
#
#  产出（全部落在 results/）：
#    core_{0,1,2,012}.log / .mp4   四组逐帧统计与可视化视频
#    npu_load_samples.txt          推理中三核负载 + 电源域状态采样
#    core_scaling_summary.txt      汇总表
#
#  性质：只写 results/，不改模型、不改设备树、不编内核、不动驱动。
#  前置：NPU 锁频 1GHz + CPU performance 必须已生效（脚本会检查）。
# ============================================================
set -u

P=/home/zsy/rk3588-yolo
S=$P/scripts
R=$P/results
M=$P/models/yolov8n_head_fp16.rknn
SRC=$P/coco/val2017
LIB=$P/lib
DEMO=$S/video_demo.py

mkdir -p "$R"

echo "============================================================"
echo " RK3588 NPU 调度扩展性 + 推理中负载采样"
echo " 模型 : $M"
echo " 输入 : $SRC (200 帧/组)"
echo " 输出 : $R"
echo "============================================================"

# ---------- 0. 预检：文件与锁频 ----------
for f in "$DEMO" "$S/infer_image.py" "$M"; do
    [ -f "$f" ] || { echo "!! 缺失: $f"; exit 1; }
done
[ -d "$SRC" ] || { echo "!! 缺失图片目录: $SRC"; exit 1; }

NPU_GOV=$(cat /sys/class/devfreq/fdab0000.npu/governor 2>/dev/null || echo '?')
NPU_CUR=$(cat /sys/class/devfreq/fdab0000.npu/cur_freq 2>/dev/null || echo '0')
CPU_GOV=$(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor 2>/dev/null || echo '?')
echo " 锁频状态: NPU governor=$NPU_GOV cur=$NPU_CUR | CPU policy0=$CPU_GOV"

if [ "$NPU_GOV" != "userspace" ] || [ "$NPU_CUR" != "1000000000" ] || [ "$CPU_GOV" != "performance" ]; then
    echo
    echo "!! 锁频未生效，四组数据不可比。先执行下面三条，再重跑本脚本："
    echo "   echo userspace | sudo tee /sys/class/devfreq/fdab0000.npu/governor"
    echo "   echo 1000000000 | sudo tee /sys/class/devfreq/fdab0000.npu/userspace/set_freq"
    echo "   sudo sh -c 'for f in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do echo performance > \$f; done'"
    exit 1
fi

# ---------- 1. 后台采样：NPU 三核负载 + 电源域状态 ----------
SAMPLE=$R/npu_load_samples.txt
: > "$SAMPLE"
(
    while true; do
        L=$(cat /sys/kernel/debug/rknpu/load 2>/dev/null | tr '\n' ' ')
        G=$(grep -E '^npu |^nputop ' /sys/kernel/debug/pm_genpd/pm_genpd_summary 2>/dev/null | head -2 | tr '\n' ' ')
        printf '%s | %s| %s\n' "$(date +%H:%M:%S.%3N)" "$L" "$G"
        sleep 0.2
    done
) >> "$SAMPLE" 2>&1 &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true' EXIT

# ---------- 2. 四组 core 配置，各 200 帧 ----------
for c in 0 1 2 012; do
    echo
    echo "===== core=$c ====="
    env LD_LIBRARY_PATH="$LIB" python3 "$DEMO" \
        --model "$M" --source "$SRC" \
        --out "$R/core_${c}.mp4" \
        --max-frames 200 --warmup 10 \
        --core "$c" --box-scale auto \
        > "$R/core_${c}.log" 2>&1
    grep -E 'core :|预热|纯前向 rknn|端到端 含|视频已保存|!!|Error|Traceback' "$R/core_${c}.log" || true
done

kill "$SAMPLER" 2>/dev/null || true
trap - EXIT

# ---------- 3. 汇总 ----------
SUM=$R/core_scaling_summary.txt
{
    echo "RK3588 NPU 调度扩展性汇总（200 帧/组, warmup 10, 锁频 1GHz + CPU performance）"
    echo "生成时间: $(date '+%F %T')"
    echo
    printf '%-6s | %-44s | %-44s\n' "core" "纯前向" "端到端"
    for c in 0 1 2 012; do
        FWD=$(grep '纯前向 rknn' "$R/core_${c}.log" | sed 's/^ *//')
        E2E=$(grep '端到端 含前后处理' "$R/core_${c}.log" | sed 's/^ *//')
        printf '%-6s | %-44s | %-44s\n' "$c" "${FWD:-无}" "${E2E:-无}"
    done
    echo
    echo "NPU load 采样（共 $(wc -l < "$SAMPLE") 行，非零核出现次数）："
    for i in 0 1 2; do
        printf '  Core%s: %s\n' "$i" "$(grep -o "Core${i}: *[1-9][0-9]*%" "$SAMPLE" | wc -l)"
    done
    echo
    echo "电源域状态样本（前 3 条 / 后 3 条）："
    head -3 "$SAMPLE"
    echo "  ..."
    tail -3 "$SAMPLE"
} | tee "$SUM"

echo
echo "完成。把 core_scaling_summary.txt 的内容贴回来即可。"
