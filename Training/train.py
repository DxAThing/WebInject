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


# ======================= 辅助函数 ==========================


def get_device() -> torch.device:
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


def get_checkpoint_path(checkpoint_dir: str, monitor_name: str) -> str:
    """获取某个 Monitor 的最新 Checkpoint 路径。"""
    return os.path.join(checkpoint_dir, f"{monitor_name}_latest.pth")


def get_epoch_checkpoint_path(
    checkpoint_dir: str, monitor_name: str, epoch: int
) -> str:
    """获取某个 Monitor 某个 Epoch 的 Checkpoint 路径。"""
    return os.path.join(checkpoint_dir, f"{monitor_name}_epoch_{epoch:04d}.pth")


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    loss: float,
) -> None:
    """
    原子保存 Checkpoint。
    先写入 .tmp 文件，再通过 os.replace 原子覆盖目标文件，
    防止在保存瞬间实例中断导致权重文件损坏。
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)  # 原子操作


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
) -> int:
    """
    加载 Checkpoint，恢复所有训练状态。
    返回: 上次完成的 epoch 编号 (下次训练从 epoch+1 开始)。
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint.get("loss", float("inf"))
    timestamp = checkpoint.get("timestamp", "unknown")

    print(f"  ✓ 从 Checkpoint 恢复: Epoch {epoch}, Loss {loss:.6f}, 保存于 {timestamp}")

    return epoch


def get_lmdb_path(monitor_name: str) -> str:
    """获取某个 Monitor 的 LMDB 路径。"""
    return os.path.join(TRAIN_CONFIG["LMDB_DIR"], f"{monitor_name}.lmdb")


# ======================= 单个 Monitor 训练 =================


def train_one_monitor(monitor_name: str, device: torch.device) -> None:
    """为单个 Monitor 训练一个 U-Net 网络 N_d。"""

    print("\n" + "=" * 60)
    print(f"训练 Monitor: {monitor_name}")
    print("=" * 60)

    # --- 路径准备 ---
    checkpoint_dir = TRAIN_CONFIG["CHECKPOINT_DIR"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    lmdb_path = get_lmdb_path(monitor_name)
    if not os.path.exists(lmdb_path):
        print(f"  [!] LMDB 不存在: {lmdb_path}")
        print(f"      请先运行 pack_data.py 打包数据。")
        return

    # --- Dataset & DataLoader ---
    dataset = LMDBDataset(
        lmdb_path=lmdb_path,
        crop_size=TRAIN_CONFIG["CROP_SIZE"],
        is_training=True,
    )

    if len(dataset) == 0:
        print(f"  [!] 数据集为空，跳过 {monitor_name}")
        return

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

    # --- 模型 + 优化器 + 调度器 ---
    model = UNet(in_channels=3, out_channels=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["LEARNING_RATE"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TRAIN_CONFIG["NUM_EPOCHS"]
    )

    # --- 断点续传: Auto-Resume ---
    start_epoch = 0
    ckpt_path = get_checkpoint_path(checkpoint_dir, monitor_name)

    if os.path.isfile(ckpt_path):
        print(f"  发现 Checkpoint: {ckpt_path}")
        start_epoch = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
        start_epoch += 1  # 从下一个 epoch 开始

    if start_epoch >= TRAIN_CONFIG["NUM_EPOCHS"]:
        print(f"  ✓ {monitor_name} 已完成全部 {TRAIN_CONFIG['NUM_EPOCHS']} 个 Epoch，跳过。")
        return

    print(f"  从 Epoch {start_epoch} 开始训练 (共 {TRAIN_CONFIG['NUM_EPOCHS']} Epoch)")
    print(f"  LR: {TRAIN_CONFIG['LEARNING_RATE']}, Batch: {TRAIN_CONFIG['BATCH_SIZE']}")

    # --- 训练循环 ---
    for epoch in range(start_epoch, TRAIN_CONFIG["NUM_EPOCHS"]):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            # 每 10 个 batch 打印一次进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(
                    f"  Epoch [{epoch + 1}/{TRAIN_CONFIG['NUM_EPOCHS']}] "
                    f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                    f"Loss: {loss.item():.6f}",
                    end="\r",
                )

        # Epoch 结束
        scheduler.step()
        avg_loss = epoch_loss / max(batch_count, 1)
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch [{epoch + 1}/{TRAIN_CONFIG['NUM_EPOCHS']}] "
            f"Avg Loss: {avg_loss:.6f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # --- 保存 Checkpoint ---
        if (epoch + 1) % TRAIN_CONFIG["SAVE_INTERVAL"] == 0:
            # 保存 latest (原子操作)
            save_checkpoint(
                ckpt_path, model, optimizer, scheduler, epoch, avg_loss
            )

            # 每 10 个 epoch 额外保存一份带 epoch 编号的副本
            if (epoch + 1) % 10 == 0:
                epoch_path = get_epoch_checkpoint_path(
                    checkpoint_dir, monitor_name, epoch + 1
                )
                save_checkpoint(
                    epoch_path, model, optimizer, scheduler, epoch, avg_loss
                )
                print(f"  💾 Checkpoint 已保存: {epoch_path}")

    print(f"\n  ✓ {monitor_name} 训练完成!")


# ======================= 主入口 ============================


def main():
    print("=" * 60)
    print("WebInject U-Net ICC 映射网络训练器")
    print(f"启动时间: {datetime.datetime.now().isoformat()}")
    print("=" * 60)

    device = get_device()

    # 论文要求: 为每个 Monitor 训练一个网络 N_d
    for monitor_name in MONITORS:
        train_one_monitor(monitor_name, device)

    print("\n" + "=" * 60)
    print("全部 Monitor 训练完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
