#!/bin/bash
# RK3588 板端环境初始化 —— Stage 0（最小侵入版，适用于多人共用板子）
#
# 设计原则：
#   1. 不修改现有 apt 源，只"新增"一个清华源文件（可随时删除）
#   2. 只装 1 个系统级包：python3-venv
#   3. 所有 Python 依赖装进 ~/rk3588-yolo/venv，不碰系统 dist-packages
#   4. 所有 apt 命令带 timeout，不会假死
#
# 用法：bash setup_board.sh
# 撤销：sudo rm /etc/apt/sources.list.d/tuna.list && rm -rf ~/rk3588-yolo/venv

WORK=/home/zsy/rk3588-yolo
VENV=$WORK/venv

echo "=== 工作目录: $WORK ==="
mkdir -p "$WORK"/{env,models,dataset,scripts,logs,packages}

echo ""
echo "[1/5] 新增清华 apt 源（不改动已有源）"
if [ ! -f /etc/apt/sources.list.d/tuna.list ]; then
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports noble main restricted universe multiverse" \
        | sudo tee /etc/apt/sources.list.d/tuna.list > /dev/null
    echo "已写入 /etc/apt/sources.list.d/tuna.list"
else
    echo "已存在，跳过"
fi

echo ""
echo "[2/5] apt update（最多 120 秒，超时不会卡死）"
if timeout 120 sudo apt-get update -qq; then
    echo "apt_ok"
else
    echo "!! apt update 超时/失败 —— 可能仍卡在其他境外源上"
    echo "   可以先手动试：sudo apt-get update -o Acquire::http::Timeout=5"
fi

echo ""
echo "[3/5] 装 python3-venv（唯一的系统级包，约几十 KB）"
if timeout 180 sudo apt-get install -y python3-venv; then
    echo "venv_ok"
else
    echo "!! 装 python3-venv 失败，后面的 venv 可能创建不了"
fi

echo ""
echo "[4/5] 创建项目虚拟环境（继承系统已有包，不污染系统）"
if [ ! -d "$VENV" ]; then
    python3 -m venv --system-site-packages "$VENV" && echo "venv 已创建" || echo "!! venv 创建失败"
else
    echo "已存在，跳过"
fi
source "$VENV"/bin/activate
python -V
which python pip

echo ""
echo "[5/5] 在 venv 内装依赖（走清华 PyPI 源）"
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple rknn_toolkit_lite2==2.3.2
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy opencv-python-headless

echo ""
echo "—— 初始化结束 ——"
echo "以后每次开新终端都要先激活：source $VENV/bin/activate"
echo "然后自检：$VENV/bin/python env/check_env.py"
