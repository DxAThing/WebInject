# ============================================================
# model.py — 标准 U-Net 架构 (用于 ICC 色彩映射)
# ============================================================
# 输入: 3 通道 RGB 原始像素图
# 输出: 3 通道 RGB ICC 映射后图像
# Padding 策略: same (输入输出尺寸严格一致)
# 最后一层: Linear (无 Activation)，配合 MSELoss 使用
# ============================================================

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm => ReLU) × 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """MaxPool => DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样 => 拼接 Skip Connection => DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # 处理因 MaxPool 导致的尺寸差异 (保证 same padding 语义)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    标准 U-Net 架构
    ─────────────────────────────────────
    Encoder:  64 → 128 → 256 → 512 → 1024
    Decoder:  1024 → 512 → 256 → 128 → 64
    Output:   64 → 3 (Linear, 无 Activation)
    ─────────────────────────────────────
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # Encoder (下采样路径)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Decoder (上采样路径)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # 输出层: 1×1 conv, 无 Activation (Linear output for MSELoss)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)     # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512
        x5 = self.down4(x4)  # 1024

        # Decoder + Skip Connections
        x = self.up1(x5, x4)  # 512
        x = self.up2(x, x3)   # 256
        x = self.up3(x, x2)   # 128
        x = self.up4(x, x1)   # 64

        return self.outc(x)   # 3


if __name__ == "__main__":
    # 快速验证: 输入输出尺寸一致
    model = UNet(in_channels=3, out_channels=3)
    dummy = torch.randn(1, 3, 512, 512)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    assert dummy.shape == out.shape, "输入输出尺寸不一致!"
    print("✓ U-Net 验证通过")

    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
