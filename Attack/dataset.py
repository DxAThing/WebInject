# ============================================================
# dataset.py — 攻击流水线的高性能数据加载器
# ============================================================
# 两种数据源 (自动回退):
#   1. LMDB 二进制数据库 (pack_data.py 打包产出, 推荐)
#   2. 散碎文件 (JSON + PNG, 兼容旧流程)
#
# 性能优化:
#   - GPU Prefetch: 启动时将所有截图 Tensor 加载到 GPU 显存
#   - LMDB 随机读取: O(1) key-value 查找, 零随机 I/O
#   - 批量采样: get_batch_pairs() 为梯度累积提供多组 (prompt, history)
#   - 断点续传: 对比已完成 delta 文件跳过已处理网页
# ============================================================

import io
import json
import logging
import os
import pickle
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import (
    ATTACK_CONFIG,
    DATASET_METADATA_JSON,
    LMDB_PATH,
    PROMPTS_JSON,
    SCREENSHOTS_DIR,
)

logger = logging.getLogger(__name__)


# ======================= 数据结构 ==========================

@dataclass
class WebpageRecord:
    """一个网页的完整数据记录。"""
    webpage_id: str                         # 例如 "blog_real_1"
    html_file: str                          # 相对路径，如 "Blog/blog_real_1.html"
    screenshot_path: str                    # 截图绝对路径 (文件模式下使用)
    shadow_histories: List[List[str]]       # 攻击训练用历史
    user_histories: List[List[str]]         # 评估用历史
    target_prompts: List[str]              # 攻击训练用提示
    user_prompts: List[str]                # 评估用提示


# ======================= 工具函数 ==========================

def _html_file_to_webpage_id(html_file: str) -> str:
    """
    从 html_file 路径中提取 webpage_id。
    例如: "Blog/blog_real_1.html" -> "blog_real_1"
    """
    basename = os.path.basename(html_file)
    return os.path.splitext(basename)[0]


def _find_screenshot_path(
    webpage_id: str, screenshot_hint: str, screenshots_dir: str
) -> str:
    """
    按优先级查找截图文件:
      1. metadata 中 screenshot 字段指定的文件名
      2. webpage_id.png
    """
    candidates = [
        os.path.join(screenshots_dir, screenshot_hint),
        os.path.join(screenshots_dir, f"{webpage_id}.png"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]  # 返回第一个候选路径 (可能不存在, 后续会报错)


# ======================= 核心加载器 ========================

class AttackDataset:
    """
    攻击流水线数据加载器 (支持 LMDB 与文件双模式)。

    功能:
        - 解析 LMDB 或 dataset_metadata.json + prompts.json
        - 提供断点续传支持 (get_unprocessed_webpages)
        - GPU Prefetch: 所有截图一次性加载到 GPU
        - 批量采样: get_batch_pairs() 为梯度累积服务
    """

    def __init__(
        self,
        metadata_path: str = DATASET_METADATA_JSON,
        prompts_path: str = PROMPTS_JSON,
        screenshots_dir: str = SCREENSHOTS_DIR,
        lmdb_path: str = LMDB_PATH,
        prefetch_to_gpu: bool = ATTACK_CONFIG["PREFETCH_TO_GPU"],
    ):
        self.metadata_path = metadata_path
        self.prompts_path = prompts_path
        self.screenshots_dir = screenshots_dir
        self.lmdb_path = lmdb_path

        # PIL -> Tensor 转换 (归一化到 [0, 1])
        self.to_tensor = transforms.ToTensor()

        # LMDB 环境 (延迟打开)
        self._lmdb_env = None
        self._use_lmdb = os.path.isdir(lmdb_path)

        # 构建记录字典: webpage_id -> WebpageRecord
        self.records: Dict[str, WebpageRecord] = {}

        # GPU 截图缓存: webpage_id -> Tensor (3, H, W) on GPU
        self._gpu_cache: Dict[str, torch.Tensor] = {}
        self._prefetch_enabled = prefetch_to_gpu

        # LMDB 原始字节缓存: webpage_id -> PNG bytes (按需加载, 不在 init 时全量读入)
        self._image_bytes_cache: Dict[str, bytes] = {}

        if self._use_lmdb:
            logger.info(f"[Dataset] 使用 LMDB 数据源: {lmdb_path}")
            self._parse_records_lmdb()
        else:
            logger.info("[Dataset] LMDB 不存在, 回退到散碎文件模式")
            self._parse_records_file(metadata_path, prompts_path)

        logger.info(f"[Dataset] 共加载 {len(self.records)} 条网页记录")

    # -------------------- LMDB 解析 -------------------------

    def _open_lmdb(self):
        """延迟打开 LMDB 环境 (避免 fork 安全问题)。"""
        if self._lmdb_env is None:
            self._lmdb_env = __import__("lmdb").open(
                self.lmdb_path, readonly=True, lock=False, readahead=True
            )
        return self._lmdb_env

    def _parse_records_lmdb(self) -> None:
        """从 LMDB 解析所有记录 (仅元数据, 图像字节按需加载)。"""
        env = self._open_lmdb()
        with env.begin(buffers=True) as txn:
            # 读取索引
            raw_keys = txn.get(b"__keys__")
            if raw_keys is None:
                logger.warning("[Dataset] LMDB 中无 __keys__, 回退到文件模式")
                self._use_lmdb = False
                self._parse_records_file(self.metadata_path, self.prompts_path)
                return

            keys = pickle.loads(raw_keys)
            for webpage_id in keys:
                raw = txn.get(webpage_id.encode("utf-8"))
                if raw is None:
                    continue
                entry = pickle.loads(raw)

                # 按需加载: 仅解析元数据, 图像字节不在此处缓存
                # 需要时通过 _lazy_load_image_bytes() 从 LMDB 读取

                self.records[webpage_id] = WebpageRecord(
                    webpage_id=webpage_id,
                    html_file=entry.get("html_file", ""),
                    screenshot_path=f"lmdb://{webpage_id}",
                    shadow_histories=entry.get("shadow_histories", []),
                    user_histories=entry.get("user_histories", []),
                    target_prompts=entry.get("target_prompts", []),
                    user_prompts=entry.get("user_prompts", []),
                )

    # -------------------- 文件解析 (兼容旧流程) ----------------

    def _parse_records_file(self, metadata_path: str, prompts_path: str) -> None:
        """从散碎 JSON + PNG 文件解析记录。"""
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        for entry in metadata.get("records", []):
            html_file = entry["html_file"]
            webpage_id = _html_file_to_webpage_id(html_file)
            screenshot_hint = entry.get("screenshot", f"{webpage_id}.png")
            screenshot_path = _find_screenshot_path(
                webpage_id, screenshot_hint, self.screenshots_dir
            )

            prompt_entry = prompts.get(
                html_file, prompts.get(html_file.replace("/", "\\"), {})
            )

            self.records[webpage_id] = WebpageRecord(
                webpage_id=webpage_id,
                html_file=html_file,
                screenshot_path=screenshot_path,
                shadow_histories=entry.get("shadow_histories", []),
                user_histories=entry.get("user_histories", []),
                target_prompts=prompt_entry.get("target_prompts", []),
                user_prompts=prompt_entry.get("user_prompts", []),
            )

    # -------------------- GPU Prefetch --------------------------

    def prefetch_all_to_gpu(self, device: str = "cuda") -> None:
        """
        一次性将所有截图加载到 GPU 显存。

        670 × 1920×1080×3×4 bytes ≈ 15 GB (float32)。
        如果显存不足，会自动回退到按需加载模式。

        优势:
            - PGD 优化中零 I/O 延迟
            - 避免反复 CPU→GPU 传输
        """
        if not self._prefetch_enabled:
            logger.info("[Prefetch] 已禁用 GPU prefetch")
            return

        logger.info(f"[Prefetch] 开始将 {len(self.records)} 张截图加载到 {device} ...")
        loaded = 0

        try:
            pbar = tqdm(
                self.records.items(),
                desc="GPU Prefetch",
                unit="img",
                total=len(self.records),
                dynamic_ncols=True,
            )
            for webpage_id, record in pbar:
                tensor = self._load_tensor_internal(webpage_id, record)
                self._gpu_cache[webpage_id] = tensor.to(device)
                loaded += 1

                if loaded % 100 == 0 and torch.cuda.is_available():
                    mem_gb = torch.cuda.memory_allocated() / 1024**3
                    pbar.set_postfix(mem=f"{mem_gb:.1f}GB")

            # 报告显存占用
            if torch.cuda.is_available():
                mem_gb = torch.cuda.memory_allocated() / 1024**3
                logger.info(
                    f"[Prefetch] 完成! 加载 {loaded} 张, "
                    f"当前显存占用: {mem_gb:.2f} GB"
                )

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(
                f"[Prefetch] 显存不足, 已加载 {loaded}/{len(self.records)}, "
                f"未加载部分将按需加载: {e}"
            )

    def _lazy_load_image_bytes(self, webpage_id: str) -> Optional[bytes]:
        """
        按需从 LMDB 读取单个网页的图像字节。

        仅在需要时才打开 LMDB 事务读取, 避免启动时将所有图像
        字节全部加载到内存 (670 张 × ~500KB ≈ 300MB+)。
        读取后缓存到 _image_bytes_cache, 下次直接命中。
        """
        if webpage_id in self._image_bytes_cache:
            return self._image_bytes_cache[webpage_id]

        if not self._use_lmdb:
            return None

        env = self._open_lmdb()
        with env.begin(buffers=True) as txn:
            raw = txn.get(webpage_id.encode("utf-8"))
            if raw is None:
                return None
            entry = pickle.loads(raw)
            img_bytes = bytes(entry["image_bytes"])
            self._image_bytes_cache[webpage_id] = img_bytes
            return img_bytes

    def evict_image_cache(self, webpage_id: Optional[str] = None) -> None:
        """
        清除图像字节缓存, 释放 CPU 内存。

        在按需加载模式下, 处理完一批网页后应调用此方法
        释放不再需要的图像字节, 避免内存持续增长。

        参数:
            webpage_id: 指定网页 ID 则仅清除该条目, None 则清空全部。
        """
        if webpage_id:
            self._image_bytes_cache.pop(webpage_id, None)
        else:
            self._image_bytes_cache.clear()

    def _load_tensor_internal(
        self, webpage_id: str, record: WebpageRecord
    ) -> torch.Tensor:
        """从 LMDB (按需) 或文件加载截图 Tensor (CPU)。"""
        # 尝试按需从 LMDB 加载
        img_bytes = self._lazy_load_image_bytes(webpage_id)
        if img_bytes is not None:
            buf = io.BytesIO(img_bytes)
            img = Image.open(buf).convert("RGB")
        else:
            # 文件模式
            if not os.path.exists(record.screenshot_path):
                raise FileNotFoundError(f"截图文件不存在: {record.screenshot_path}")
            img = Image.open(record.screenshot_path).convert("RGB")

        return self.to_tensor(img)  # (3, H, W), [0, 1], float32

    # -------------------- 断点续传 --------------------------

    def get_unprocessed_webpages(
        self, output_dir: str = ATTACK_CONFIG["DELTA_OUTPUT_DIR"]
    ) -> List[WebpageRecord]:
        """
        对比元数据与 DELTA_OUTPUT_DIR，返回尚未完成 delta 优化的网页列表。

        断点续传核心逻辑:
            - 已完成的网页会生成 delta_{webpage_id}.pt 文件。
            - 如果该文件存在则跳过，否则加入待处理列表。
            - 不计入 .tmp 文件 (原子写入的中间产物)。
        """
        unprocessed = []
        for webpage_id, record in self.records.items():
            delta_path = os.path.join(output_dir, f"delta_{webpage_id}.pt")
            if not os.path.exists(delta_path):
                unprocessed.append(record)
        return unprocessed

    def get_all_webpages(self) -> List[WebpageRecord]:
        """返回所有网页记录。"""
        return list(self.records.values())

    # -------------------- 按需加载 --------------------------

    def load_screenshot_tensor(
        self, record: WebpageRecord, device: str = "cpu"
    ) -> torch.Tensor:
        """
        加载指定网页的截图 Tensor。

        优先从 GPU 缓存中返回 (零延迟),
        否则从 LMDB/文件加载并传输到目标设备。

        返回: shape = (3, H, W)，数值范围 [0, 1]。
        """
        # 优先: GPU 缓存命中
        if record.webpage_id in self._gpu_cache:
            cached = self._gpu_cache[record.webpage_id]
            if str(cached.device) == device or (
                device == "cuda" and cached.is_cuda
            ):
                return cached
            return cached.to(device)

        # 回退: 从 LMDB 或文件加载
        tensor = self._load_tensor_internal(record.webpage_id, record)
        return tensor.to(device)

    def get_record(self, webpage_id: str) -> Optional[WebpageRecord]:
        """按 webpage_id 查询单条记录。"""
        return self.records.get(webpage_id)

    # -------------------- 批量采样 (梯度累积) -------------------

    def get_batch_pairs(
        self,
        record: WebpageRecord,
        batch_size: int,
        pair_type: str = "shadow",
    ) -> List[Tuple[str, List[str]]]:
        """
        从 record 中随机采样多组 (prompt, history) 对。

        用于 PGD 梯度累积: 每步对 batch_size 组 pair 计算 loss 并累积梯度,
        等效于 mini-batch SGD, 梯度信号更稳定, 收敛更快。

        参数:
            record: 网页记录
            batch_size: 采样数
            pair_type: "shadow" = 训练集 (target_prompts + shadow_histories)
                       "user"   = 测试集 (user_prompts + user_histories)

        返回:
            [(prompt, history), ...] 长度为 batch_size
        """
        if pair_type == "shadow":
            prompts = record.target_prompts or ["Perform the next action on the webpage."]
            histories = record.shadow_histories or [[]]
        else:
            prompts = record.user_prompts or ["Perform the next action on the webpage."]
            histories = record.user_histories or [[]]

        pairs = []
        for _ in range(batch_size):
            p = random.choice(prompts)
            h = random.choice(histories)
            pairs.append((p, h))
        return pairs

    # -------------------- 资源清理 ----------------------------

    def release_gpu_cache(self, webpage_id: Optional[str] = None) -> None:
        """释放 GPU 缓存 (全部或指定网页)。"""
        if webpage_id:
            self._gpu_cache.pop(webpage_id, None)
        else:
            self._gpu_cache.clear()

    def close(self) -> None:
        """关闭 LMDB 环境 + 释放缓存。"""
        self._gpu_cache.clear()
        self._image_bytes_cache.clear()
        if self._lmdb_env is not None:
            self._lmdb_env.close()
            self._lmdb_env = None
