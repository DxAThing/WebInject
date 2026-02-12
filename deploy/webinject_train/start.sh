#!/bin/bash
# ============================================================
# start.sh — WebInject U-Net 训练抢占式实例启动脚本
# ============================================================
#
# 部署步骤:
#   1. 将 webinject_train.zip 上传到实例并解压
#   2. 将 LMDB 数据文件放入 ./data/ 目录:
#        data/iMac_M1_24.lmdb/
#        data/Dell_S2722QC.lmdb/
#   3. 运行: bash start.sh
#
# 目录结构 (部署后):
#   webinject_train/
#   ├── config.py
#   ├── model.py
#   ├── dataset.py
#   ├── train.py
#   ├── requirements.txt
#   ├── start.sh
#   ├── data/                 ← 放入 LMDB 文件
#   │   ├── iMac_M1_24.lmdb/
#   │   └── Dell_S2722QC.lmdb/
#   └── checkpoints/          ← 自动生成
#
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "WebInject U-Net Training — Spot Instance Launcher"
echo "Time: $(date)"
echo "Dir:  $(pwd)"
echo "============================================================"

# ---------- 环境检查 ----------

# 安装依赖 (首次运行)
if ! python -c "import torch; import lmdb" 2>/dev/null; then
    echo "[Setup] 安装 Python 依赖..."
    pip install -r requirements.txt
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
fi

# 检查 CUDA
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA 不可用!'
print(f'CUDA OK: {torch.cuda.get_device_name(0)}')
"

# 检查 LMDB 数据
if [ ! -d "./data" ]; then
    echo "[ERROR] ./data/ 目录不存在，请放入 LMDB 数据文件"
    exit 1
fi

echo ""

# ---------- 创建目录 ----------
mkdir -p checkpoints

# ---------- 容错训练循环 ----------
# 抢占式实例被回收后，systemd/cron 重启此脚本即可自动续传

MAX_RETRIES=999
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    echo "[Run $((RETRY+1))] 启动训练 @ $(date)"
    
    python train.py
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[Done] 训练正常完成!"
        break
    fi
    
    RETRY=$((RETRY+1))
    echo "[Restart] 训练进程退出 (code=$EXIT_CODE)，5 秒后重启..."
    sleep 5
done

echo "============================================================"
echo "训练结束 @ $(date)"
echo "============================================================"
