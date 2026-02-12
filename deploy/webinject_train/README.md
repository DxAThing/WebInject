# WebInject U-Net ICC 映射网络训练 — 抢占式实例部署包

## Quick Start

```bash
# 1. 解压
unzip webinject_train.zip
cd webinject_train

# 2. 安装依赖
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. 将 LMDB 数据放入 ./data/
#    data/iMac_M1_24.lmdb/
#    data/Dell_S2722QC.lmdb/

# 4. 启动训练 (支持断点续传)
bash start.sh
```

## 目录结构

```
webinject_train/
├── config.py           # 超参配置
├── model.py            # U-Net 架构
├── dataset.py          # LMDB 数据加载器
├── train.py            # 训练器 (断点续传 + 原子保存)
├── requirements.txt    # Python 依赖
├── start.sh            # 抢占式实例启动脚本
├── README.md
├── data/               ← 需手动放入 LMDB 数据
│   ├── iMac_M1_24.lmdb/
│   └── Dell_S2722QC.lmdb/
└── checkpoints/        ← 训练自动生成
```

## 断点续传

训练器启动时自动检测 `checkpoints/{monitor}_latest.pth`，若存在则恢复:
- Model Weights
- Optimizer State
- Scheduler State
- Last Epoch

保存使用原子操作 (`.tmp` → `rename`)，防止实例中断导致文件损坏。

## 抢占式实例配置

`start.sh` 内置自动重启循环。也可配置 systemd:

```ini
[Service]
ExecStart=/bin/bash /path/to/webinject_train/start.sh
Restart=always
RestartSec=10
```
