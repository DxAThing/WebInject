# ============================================================
# train.py — 高可用 U-Net ICC 映射网络训练器
# ============================================================
#
# 【抢占式实例 (Spot Instance) 部署说明】
#
# 由于训练环境为抢占式实例，实例可能在任意时刻被回收。
# 本脚本已内置完整的断点续传 (Auto-Resume) 机制:
#   - 启动时自动检测并恢复最新 Checkpoint
#   - 保存时使用原子操作 (先写 .tmp 再 rename)，防止写入中断导致损坏
#
# 推荐在实例启动脚本中使用以下命令，实现"被抢占后自动重启":
#
#   #!/bin/bash
#   cd /path/to/Training
#   while true; do
#       python train.py
#       echo "训练进程退出 (exit code: $?)，5 秒后重启..."
#       sleep 5
#   done
#
# 或使用 systemd 配置自动重启:
#
#   [Unit]
#   Description=WebInject U-Net Training
#   After=network.target
#
#   [Service]
#   Type=simple
#   WorkingDirectory=/path/to/Training
#   ExecStart=/usr/bin/python train.py
#   Restart=always
#   RestartSec=10
#
#   [Install]
#   WantedBy=multi-user.target
#
# ============================================================

import os
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import TRAIN_CONFIG, MONITORS
from model import UNet
from dataset import LMDBDataset


class Trainer:
    """
    U-Net ICC 映射网络训练器。

    封装完整的训练生命周期:
      - 设备自动选择
      - 模型 / 优化器 / 调度器初始化
      - Checkpoint 自动恢复 (断点续传)
      - 训练循环 & 日志
      - 原子 Checkpoint 保存

    每个 Trainer 实例对应一个 Monitor (显示器) 的训练任务。
    """

    def __init__(self, monitor_name: str, device: torch.device | None = None):
        self.monitor_name = monitor_name
        self.device = device or self._detect_device()

        self.checkpoint_dir = TRAIN_CONFIG["CHECKPOINT_DIR"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.num_epochs: int = TRAIN_CONFIG["NUM_EPOCHS"]
        self.save_interval: int = TRAIN_CONFIG["SAVE_INTERVAL"]
        self.start_epoch: int = 0

        # --- 数据集 ---
        self.dataloader = self._build_dataloader()

        # --- 模型 / 优化器 / 调度器 ---
        self.model = UNet(in_channels=3, out_channels=3).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=TRAIN_CONFIG["LEARNING_RATE"]
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs
        )

        # --- 断点续传: 自动恢复 Checkpoint ---
        self._resume_from_checkpoint()

    # -------------------- 初始化辅助 --------------------

    @staticmethod
    def _detect_device() -> torch.device:
        """自动选择最佳计算设备。"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[Device] CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("[Device] Apple MPS")
        else:
            device = torch.device("cpu")
            print("[Device] CPU")
        return device

    def _build_dataloader(self) -> DataLoader | None:
        """构建 DataLoader，若数据不可用则返回 None。"""
        lmdb_path = os.path.join(
            TRAIN_CONFIG["LMDB_DIR"], f"{self.monitor_name}.lmdb"
        )
        if not os.path.exists(lmdb_path):
            print(f"  [!] LMDB 不存在: {lmdb_path}")
            print(f"      请先运行 pack_data.py 打包数据。")
            return None

        dataset = LMDBDataset(
            lmdb_path=lmdb_path,
            crop_size=TRAIN_CONFIG["CROP_SIZE"],
            is_training=True,
        )

        if len(dataset) == 0:
            print(f"  [!] 数据集为空: {self.monitor_name}")
            return None

        dataloader = DataLoader(
            dataset,
            batch_size=TRAIN_CONFIG["BATCH_SIZE"],
            shuffle=True,
            num_workers=TRAIN_CONFIG["NUM_WORKERS"],
            pin_memory=TRAIN_CONFIG["PIN_MEMORY"],
            drop_last=True,
        )

        print(f"  样本数: {len(dataset)}")
        print(f"  Batch 数: {len(dataloader)}")
        return dataloader

    # -------------------- Checkpoint 管理 --------------------

    @property
    def latest_checkpoint_path(self) -> str:
        """当前 Monitor 最新 Checkpoint 的路径。"""
        return os.path.join(self.checkpoint_dir, f"{self.monitor_name}_latest.pth")

    def _epoch_checkpoint_path(self, epoch: int) -> str:
        """某个 epoch 的 Checkpoint 路径。"""
        return os.path.join(
            self.checkpoint_dir, f"{self.monitor_name}_epoch_{epoch:04d}.pth"
        )

    def _save_checkpoint(self, path: str, epoch: int, loss: float) -> None:
        """
        原子保存 Checkpoint。
        先写入 .tmp 文件，再通过 os.replace 原子覆盖目标文件，
        防止在保存瞬间实例中断导致权重文件损坏。
        """
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "loss": loss,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        tmp_path = path + ".tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)  # 原子操作

    def _load_checkpoint(self, path: str) -> int:
        """
        加载 Checkpoint，恢复所有训练状态。
        返回: 上次完成的 epoch 编号。
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint["epoch"]
        loss = checkpoint.get("loss", float("inf"))
        timestamp = checkpoint.get("timestamp", "unknown")

        print(
            f"  [Resume] 从 Checkpoint 恢复: Epoch {epoch}, "
            f"Loss {loss:.6f}, 保存于 {timestamp}"
        )
        return epoch

    def _resume_from_checkpoint(self) -> None:
        """检测并恢复最新 Checkpoint (断点续传)。"""
        ckpt_path = self.latest_checkpoint_path
        if os.path.isfile(ckpt_path):
            print(f"  发现 Checkpoint: {ckpt_path}")
            self.start_epoch = self._load_checkpoint(ckpt_path) + 1

    # -------------------- 训练 --------------------

    def _train_one_epoch(self, epoch: int) -> float:
        """执行单个 epoch 的训练，返回平均 loss。"""
        assert self.dataloader is not None, "DataLoader 未初始化"

        self.model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            # 每 10 个 batch 打印一次进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(self.dataloader):
                print(
                    f"  Epoch [{epoch + 1}/{self.num_epochs}] "
                    f"Batch [{batch_idx + 1}/{len(self.dataloader)}] "
                    f"Loss: {loss.item():.6f}",
                    end="\r",
                )

        return epoch_loss / max(batch_count, 1)

    def train(self) -> None:
        """执行完整的训练流程。"""
        print("\n" + "=" * 60)
        print(f"训练 Monitor: {self.monitor_name}")
        print("=" * 60)

        if self.dataloader is None:
            print(f"  [!] 数据不可用，跳过 {self.monitor_name}")
            return

        if self.start_epoch >= self.num_epochs:
            print(
                f"  [Skip] {self.monitor_name} 已完成全部 "
                f"{self.num_epochs} 个 Epoch，跳过。"
            )
            return

        print(
            f"  从 Epoch {self.start_epoch} 开始训练 "
            f"(共 {self.num_epochs} Epoch)"
        )
        print(
            f"  LR: {TRAIN_CONFIG['LEARNING_RATE']}, "
            f"Batch: {TRAIN_CONFIG['BATCH_SIZE']}"
        )

        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_start = time.time()

            avg_loss = self._train_one_epoch(epoch)

            self.scheduler.step()
            elapsed = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"  Epoch [{epoch + 1}/{self.num_epochs}] "
                f"Avg Loss: {avg_loss:.6f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            # --- 保存 Checkpoint ---
            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(
                    self.latest_checkpoint_path, epoch, avg_loss
                )

                # 每 10 个 epoch 额外保存一份带编号的副本
                if (epoch + 1) % 10 == 0:
                    epoch_path = self._epoch_checkpoint_path(epoch + 1)
                    self._save_checkpoint(epoch_path, epoch, avg_loss)
                    print(f"  Checkpoint 已保存: {epoch_path}")

        print(f"\n  [Done] {self.monitor_name} 训练完成!")


# ======================= 主入口 ============================


def main():
    print("=" * 60)
    print("WebInject U-Net ICC 映射网络训练器")
    print(f"启动时间: {datetime.datetime.now().isoformat()}")
    print("=" * 60)

    device = Trainer._detect_device()

    # 论文要求: 为每个 Monitor 训练一个网络 N_d
    for monitor_name in MONITORS:
        trainer = Trainer(monitor_name, device=device)
        trainer.train()

    print("\n" + "=" * 60)
    print("全部 Monitor 训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
