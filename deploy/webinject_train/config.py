# ============================================================
# config.py — U-Net 映射网络训练流水线 (抢占式实例部署版)
# ============================================================
# 部署到抢占式实例时，所有路径基于本目录，无需 Dataset 目录。
# 数据已通过 pack_data.py 打包为 LMDB，直接放在 ./data/ 下。
# ============================================================

import os

# ======================= 路径配置 ==========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 部署版: LMDB 数据直接位于 ./data/ 下 (无需引用 Dataset 目录)
# pack_data.py 仍保留 DATASET_DIR 引用，但实例上不运行打包
DATASET_DIR = os.path.join(BASE_DIR, "source_data")  # 仅 pack_data 用，部署时可忽略
DATASET_METADATA_JSON = os.path.join(DATASET_DIR, "dataset_metadata.json")
SCREENSHOTS_DIR = os.path.join(DATASET_DIR, "screenshots")
RAW_SCREENSHOTS_DIR = os.path.join(DATASET_DIR, "screenshots_raw")

# ======================= 训练配置 ==========================

TRAIN_CONFIG = {
    "BATCH_SIZE": 16,
    "LEARNING_RATE": 0.005,
    "NUM_EPOCHS": 200,
    "SAVE_INTERVAL": 1,
    "CHECKPOINT_DIR": os.path.join(BASE_DIR, "checkpoints"),
    "LMDB_DIR": os.path.join(BASE_DIR, "data"),
    "CROP_SIZE": 512,
    "NUM_WORKERS": 4,
    "PIN_MEMORY": True,
}

# ======================= 目标显示器规格 ====================

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
