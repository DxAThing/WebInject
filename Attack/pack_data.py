# ============================================================
# pack_data.py — 将截图 + 元数据打包为 LMDB 二进制数据库
# ============================================================
# 解决攻击流水线中散碎文件 I/O 的性能瓶颈:
#   - 670 张截图 (PNG) 合并为一个 LMDB 文件
#   - 每条记录包含: PNG 字节流 + prompts + histories
#   - 攻击/评估阶段直接从 LMDB 读取，零随机 I/O
#
# Key 设计:
#   webpage_id (str) → pickle({
#       "image_bytes":    PNG bytes,
#       "target_prompts": [...],
#       "user_prompts":   [...],
#       "shadow_histories": [[...], ...],
#       "user_histories":   [[...], ...],
#       "html_file":      "Blog/blog_real_10.html",
#   })
#   b"__keys__"   → pickle(all_key_list)
#   b"__count__"  → pickle(int)
#
# 用法:
#   python pack_data.py
# ============================================================

import io
import json
import os
import pickle
import time

import lmdb
from PIL import Image
from tqdm import tqdm

from config import (
    ATTACK_CONFIG,
    DATASET_METADATA_JSON,
    LMDB_PATH,
    PROMPTS_JSON,
    SCREENSHOTS_DIR,
)


def _read_image_bytes(path: str) -> bytes:
    """读取图片并转换为 PNG bytes (统一格式)。"""
    with Image.open(path) as img:
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


def _html_file_to_webpage_id(html_file: str) -> str:
    """从 html_file 路径中提取 webpage_id。"""
    basename = os.path.basename(html_file)
    return os.path.splitext(basename)[0]


def pack_attack_lmdb() -> None:
    """
    将所有截图 + 元数据打包为 LMDB 数据库。

    流程:
      1. 解析 dataset_metadata.json 获取记录列表
      2. 解析 prompts.json 获取 prompts
      3. 逐条读取截图 PNG + 合并元数据
      4. 写入 LMDB
    """
    print("=" * 60)
    print("LMDB 打包: Attack 数据集")
    print("=" * 60)

    # ---- 加载元数据 ----
    with open(DATASET_METADATA_JSON, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(PROMPTS_JSON, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)

    records = metadata.get("records", [])
    print(f"  元数据记录数: {len(records)}")

    # ---- 收集有效样本 ----
    samples = []
    skipped = 0

    for entry in records:
        html_file = entry["html_file"]
        webpage_id = _html_file_to_webpage_id(html_file)

        # 截图文件名: 优先用 metadata 中 screenshot 字段, 否则用 webpage_id.png
        screenshot_name = entry.get("screenshot", f"{webpage_id}.png")
        screenshot_path = os.path.join(SCREENSHOTS_DIR, screenshot_name)

        if not os.path.isfile(screenshot_path):
            skipped += 1
            continue

        # 从 prompts.json 查找 prompts
        prompt_entry = prompts_data.get(
            html_file, prompts_data.get(html_file.replace("/", "\\"), {})
        )

        samples.append({
            "webpage_id": webpage_id,
            "html_file": html_file,
            "screenshot_path": screenshot_path,
            "target_prompts": prompt_entry.get("target_prompts", []),
            "user_prompts": prompt_entry.get("user_prompts", []),
            "shadow_histories": entry.get("shadow_histories", []),
            "user_histories": entry.get("user_histories", []),
        })

    print(f"  有效样本: {len(samples)}, 跳过(截图缺失): {skipped}")

    if not samples:
        print("[!] 无有效样本, 终止打包")
        return

    # ---- 创建 LMDB ----
    os.makedirs(os.path.dirname(LMDB_PATH), exist_ok=True)

    # 估算 map_size: 每张截图约 0.5MB + 元数据, × 2 安全系数
    estimated_bytes = sum(
        os.path.getsize(s["screenshot_path"]) for s in samples
    )
    map_size = max(estimated_bytes * 3, 512 * 1024 * 1024)  # 至少 512MB

    print(f"  LMDB 路径: {LMDB_PATH}")
    print(f"  预估数据量: {estimated_bytes / 1024 / 1024:.1f} MB")
    print(f"  LMDB map_size: {map_size / 1024 / 1024:.0f} MB")

    env = lmdb.open(LMDB_PATH, map_size=map_size)
    keys = []
    total_bytes = 0
    start_time = time.time()

    with env.begin(write=True) as txn:
        for sample in tqdm(samples, desc="LMDB 打包", unit="条", dynamic_ncols=True):
            webpage_id = sample["webpage_id"]
            keys.append(webpage_id)

            # 读取截图为 PNG bytes
            image_bytes = _read_image_bytes(sample["screenshot_path"])

            value = pickle.dumps({
                "image_bytes": image_bytes,
                "html_file": sample["html_file"],
                "target_prompts": sample["target_prompts"],
                "user_prompts": sample["user_prompts"],
                "shadow_histories": sample["shadow_histories"],
                "user_histories": sample["user_histories"],
            })

            txn.put(webpage_id.encode("utf-8"), value)
            total_bytes += len(value)

        # 存储索引
        txn.put(b"__keys__", pickle.dumps(keys))
        txn.put(b"__count__", pickle.dumps(len(keys)))

    env.close()

    elapsed = time.time() - start_time
    final_size = os.path.getsize(os.path.join(LMDB_PATH, "data.mdb"))

    print(f"\n{'=' * 60}")
    print(f"  打包完成!")
    print(f"  样本数: {len(keys)}")
    print(f"  原始数据: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  LMDB 文件: {final_size / 1024 / 1024:.1f} MB")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    pack_attack_lmdb()
