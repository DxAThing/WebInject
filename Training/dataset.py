# ============================================================
# dataset.py — LMDB 数据加载器
# ============================================================
# 从 pack_data.py 打包的 LMDB 中高效读取训练样本。
# 支持:
#   - 随机裁剪 (RandomCrop) 处理 4K 截图
#   - 随机扰动 δ' (论文要求的 AddRandomPerturbation)
#   - 标准 ToTensor 归一化
# ============================================================

import io
import pickle
import random

import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import TRAIN_CONFIG


class AddRandomPerturbation:
    """
    添加微小随机扰动 δ' (复现论文中的对抗鲁棒性增强)。
    在像素空间 [0, 1] 上添加均匀分布噪声 U(-epsilon, epsilon)，
    然后 clamp 到 [0, 1]。
    """

    def __init__(self, epsilon: float = 0.02):
        self.epsilon = epsilon

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = (torch.rand_like(tensor) * 2 - 1) * self.epsilon
        return torch.clamp(tensor + noise, 0.0, 1.0)


class SynchronizedRandomCrop:
    """
    对 input 和 target 使用相同的随机裁剪区域。
    必须保证 (input, target) 裁剪位置一致。
    """

    def __init__(self, crop_size: int):
        self.crop_size = crop_size

    def __call__(self, input_img: Image.Image, target_img: Image.Image):
        w, h = input_img.size
        cs = self.crop_size

        # 如果图片比裁剪尺寸小，则 resize 到裁剪尺寸
        if w < cs or h < cs:
            scale = max(cs / w, cs / h)
            new_w, new_h = int(w * scale) + 1, int(h * scale) + 1
            input_img = input_img.resize((new_w, new_h), Image.BILINEAR)
            target_img = target_img.resize((new_w, new_h), Image.BILINEAR)
            w, h = new_w, new_h

        # 随机选择裁剪起点
        x = random.randint(0, w - cs)
        y = random.randint(0, h - cs)

        input_crop = input_img.crop((x, y, x + cs, y + cs))
        target_crop = target_img.crop((x, y, x + cs, y + cs))

        return input_crop, target_crop


class LMDBDataset(Dataset):
    """
    从 LMDB 读取 (input, target) 图片对。

    参数:
        lmdb_path: LMDB 文件路径
        crop_size: 随机裁剪尺寸
        is_training: 是否为训练模式 (控制数据增强)
        perturbation_epsilon: 随机扰动幅度
    """

    def __init__(
        self,
        lmdb_path: str,
        crop_size: int = TRAIN_CONFIG["CROP_SIZE"],
        is_training: bool = True,
        perturbation_epsilon: float = 0.02,
    ):
        self.lmdb_path = lmdb_path
        self.crop_size = crop_size
        self.is_training = is_training

        # 打开 LMDB (只读模式, 不加锁以支持多进程 DataLoader)
        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        # 读取 Key 列表
        with self.env.begin(write=False) as txn:
            raw = txn.get(b"__keys__")
            if raw is None:
                raise ValueError(f"LMDB 中缺少 __keys__ 元信息: {lmdb_path}")
            self.keys = pickle.loads(raw)

        # 同步随机裁剪
        self.sync_crop = SynchronizedRandomCrop(crop_size)

        # ToTensor: PIL Image → [0, 1] float tensor
        self.to_tensor = transforms.ToTensor()

        # 随机扰动 (仅用于 input)
        self.perturbation = AddRandomPerturbation(perturbation_epsilon)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        key = self.keys[index]

        with self.env.begin(write=False) as txn:
            raw = txn.get(key.encode("utf-8"))
            if raw is None:
                raise KeyError(f"LMDB 中找不到 key: {key}")

        data = pickle.loads(raw)

        # 解码为 PIL Image
        input_img = Image.open(io.BytesIO(data["input"])).convert("RGB")
        target_img = Image.open(io.BytesIO(data["target"])).convert("RGB")

        if self.is_training:
            # 同步随机裁剪
            input_img, target_img = self.sync_crop(input_img, target_img)

        # 转为 Tensor [0, 1]
        input_tensor = self.to_tensor(input_img)
        target_tensor = self.to_tensor(target_img)

        if self.is_training:
            # 对 input 添加随机扰动 δ'
            input_tensor = self.perturbation(input_tensor)

        return input_tensor, target_tensor

    def __del__(self):
        """安全关闭 LMDB 环境。"""
        if hasattr(self, "env") and self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
