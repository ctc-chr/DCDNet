import torch
import torch.nn as nn
import math
from functools import partial
from modules.TtS_new import TtS
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MBConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        mid_channels = int(in_channels * expand_ratio)
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        # Expansion phase
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv3d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm3d(mid_channels),
                Swish()
            )
        else:
            self.expand = None

        # Depthwise convolution
        self.dw_conv = nn.Sequential(
            nn.Conv3d(mid_channels, mid_channels, kernel_size, stride,
                      padding=kernel_size//2, groups=mid_channels, bias=False),
            nn.BatchNorm3d(mid_channels),
            Swish()
        )

        # Squeeze-and-Excitation (3D version)
        if se_ratio is not None:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(mid_channels, se_channels, 1),
                Swish(),
                nn.Conv3d(se_channels, mid_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None

        # Output phase
        self.project = nn.Sequential(
            nn.Conv3d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        residual = x

        if self.expand is not None:
            x = self.expand(x)

        x = self.dw_conv(x)

        if self.se is not None:
            x = x * self.se(x)

        x = self.project(x)

        if self.use_residual:
            x += residual

        return x

class EfficientNet3D(nn.Module):
    def __init__(self, width_coef=1.0, depth_coef=1.0, dropout=0.2, num_classes=1000):
        super().__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        TtS_channels = [16, 24, 40, 80, 112, 192, 320, 1280]
        depths = [1, 2, 2, 3, 3, 4, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        strides = [1, 2, 2, 2, 1, 2, 1]
        expand_ratios = [1, 6, 6, 6, 6, 6, 6]

        # Scale channels and depths
        channels = [math.ceil(width_coef * c) for c in channels]
        depths = [math.ceil(depth_coef * d) for d in depths]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(channels[0]),
            Swish()
        )
        # Blocks
        blocks = []
        in_ch = channels[0]
        for i in range(7):
            out_ch = channels[i+1]
            for j in range(depths[i]):
                stride = strides[i] if j == 0 else 1
                blocks.append(MBConvBlock3D(in_ch, out_ch, kernel_sizes[i], stride, expand_ratios[i]))
                in_ch = out_ch
            blocks.append(TtS(TtS_channels[i]))
        self.blocks = nn.Sequential(*blocks)

        # Head
        self.head = nn.Sequential(
            nn.Conv3d(in_ch, channels[-1], 1, bias=False),
            nn.BatchNorm3d(channels[-1]),
            Swish(),
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

def efficientnet3d_b0(num_classes=1000):
    return EfficientNet3D(1.0, 1.0, 0.2, num_classes)

def efficientnet3d_b1(num_classes=1000):
    return EfficientNet3D(1.0, 1.1, 0.2, num_classes)