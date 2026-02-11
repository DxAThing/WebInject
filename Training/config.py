# ============================================================
# config.py — U-Net 映射网络训练流水线的统一配置中心
# ============================================================
# 所有路径、超参数在此定义，严禁使用 argparse。
# ============================================================

import os

# ======================= 路径配置 ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "Dataset")

# 上一阶段产出
DATASET_METADATA_JSON = os.path.join(DATASET_DIR, "data", "dataset_metadata.json")
SCREENSHOTS_DIR = os.path.join(DATASET_DIR, "data", "screenshots")        # ICC 变换后 (Target)
RAW_SCREENSHOTS_DIR = os.path.join(DATASET_DIR, "data", "screenshots_raw")  # 原始像素 (Input)

# ======================= 训练配置 ==========================

TRAIN_CONFIG = {
    "BATCH_SIZE": 16,
    "LEARNING_RATE": 0.005,
    "NUM_EPOCHS": 200,
    "SAVE_INTERVAL": 1,           # 每 N 个 epoch 保存一次 checkpoint
    "CHECKPOINT_DIR": os.path.join(BASE_DIR, "checkpoints"),
    "LMDB_DIR": os.path.join(BASE_DIR, "data"),  # 每个 monitor 一个 LMDB
    "CROP_SIZE": 512,             # 4K 截图随机裁剪尺寸
    "NUM_WORKERS": 4,             # DataLoader worker 数
    "PIN_MEMORY": True,
}

# ======================= 目标显示器规格 ====================
# 复制自 Dataset/config.py，每个 Monitor 训练一个独立的 U-Net

MONITORS = {
    "iMac_M1_24": {
        "width": 4480,
        "height": 2520,
        "icc_file": "Display P3.icc",
    },
    "Dell_S2722QC": {
        "width": 3840,
        "height": 2160,
        "icc_file": "sRGB_v4_ICC_preference.icc",
    },
}

# ======================= 网页分类域 ========================

DOMAINS = ["Blog", "Commerce", "Education", "Healthcare", "Portfolio"]
