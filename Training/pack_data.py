# ============================================================
# pack_data.py — 将散碎图片打包为 LMDB 数据库
# ============================================================
# 解决机械硬盘 / 网络存储的 IOPS 瓶颈。
# 为每个 Monitor 生成一个独立的 LMDB 文件。
#
# 用法:
#   python pack_data.py
# ============================================================

import json
import os
import io
import pickle
import lmdb
from PIL import Image

from config import (
    DATASET_METADATA_JSON,
    SCREENSHOTS_DIR,
    RAW_SCREENSHOTS_DIR,
    MONITORS,
    TRAIN_CONFIG,
)


def _read_image_bytes(path: str) -> bytes:
    """读取图片并转换为 PNG bytes (统一格式，节省体积)。"""
    with Image.open(path) as img:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def pack_monitor_lmdb(monitor_name: str, records: list, lmdb_dir: str) -> None:
    """
    将某个 Monitor 的所有 (raw, icc) 图片对打包进一个 LMDB。

    Key 设计: f"{monitor_name}_{index:08d}"
    Value:    pickle.dumps({"input": raw_png_bytes, "target": icc_png_bytes})
    Meta:     b"__keys__" → pickle.dumps(all_key_list)
    """
    lmdb_path = os.path.join(lmdb_dir, f"{monitor_name}.lmdb")
    os.makedirs(lmdb_dir, exist_ok=True)

    # 先收集所有有效样本
    samples = []
    skipped = 0

    for record in records:
        # 在 record 的 screenshots / raw_screenshots 中查找 monitor_name 对应条目
        target_file = None
        input_file = None

        for s in record.get("screenshots", []):
            if monitor_name in s:
                target_file = s
                break

        for r in record.get("raw_screenshots", []):
            if monitor_name in r:
                input_file = r
                break

        if target_file is None or input_file is None:
            skipped += 1
            continue

        target_path = os.path.join(SCREENSHOTS_DIR, target_file)
        input_path = os.path.join(RAW_SCREENSHOTS_DIR, input_file)

        if not os.path.isfile(target_path) or not os.path.isfile(input_path):
            skipped += 1
            continue

        samples.append((input_path, target_path))

    if not samples:
        print(f"  [!] {monitor_name}: 无有效样本，跳过")
        return

    print(f"  [{monitor_name}] 有效样本: {len(samples)}, 跳过: {skipped}")

    # 估算 LMDB map_size: 每张 4K PNG 约 20MB，× 2 (input+target) × 样本数 × 1.5 安全系数
    estimated_size = len(samples) * 2 * 20 * 1024 * 1024
    map_size = max(estimated_size * 2, 1 * 1024 * 1024 * 1024)  # 至少 1GB

    env = lmdb.open(lmdb_path, map_size=map_size)
    keys = []

    with env.begin(write=True) as txn:
        for idx, (input_path, target_path) in enumerate(samples):
            key = f"{monitor_name}_{idx:08d}"
            keys.append(key)

            print(f"    打包 [{idx + 1}/{len(samples)}] {key}", end="\r")

            input_bytes = _read_image_bytes(input_path)
            target_bytes = _read_image_bytes(target_path)

            value = pickle.dumps({
                "input": input_bytes,
                "target": target_bytes,
            })

            txn.put(key.encode("utf-8"), value)

        # 存储所有 Key 的索引
        txn.put(b"__keys__", pickle.dumps(keys))

    env.close()
    print(f"    ✓ {monitor_name}: {len(keys)} 样本已打包 → {lmdb_path}")


def main():
    print("=" * 60)
    print("LMDB 数据打包工具")
    print("=" * 60)

    # 读取 dataset_metadata.json
    if not os.path.isfile(DATASET_METADATA_JSON):
        raise FileNotFoundError(
            f"找不到元数据文件: {DATASET_METADATA_JSON}\n"
            "请先运行 Dataset 阶段的流水线。"
        )

    with open(DATASET_METADATA_JSON, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    records = metadata.get("records", [])
    print(f"共 {len(records)} 条记录")

    lmdb_dir = TRAIN_CONFIG["LMDB_DIR"]

    # 为每个 Monitor 打包一个 LMDB
    for monitor_name in MONITORS:
        print(f"\n>>> 打包 Monitor: {monitor_name}")
        pack_monitor_lmdb(monitor_name, records, lmdb_dir)

    print("\n" + "=" * 60)
    print("全部打包完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
