"""
通道注意力模块
"""

import torch
from torch import nn


class senet(nn.Module):
    def __init__(self, channels, ratio=8):
        super(senet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, False),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels, False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, t, h, w = x.size()
        # b,c,t,h,w -> b,c,1,1,1
        avg = self.avg_pool(x).view([b, c])
        # b,c -> b,c // ratio -> b,c -> b,c,1,1
        fc = self.fc(avg).view([b, c, 1, 1, 1])

        return fc * x


if __name__ == '__main__':
    model = senet(512)
    print(model)
    x = torch.ones([1, 512, 16, 28, 28])
    result = model(x)
    print(result[0])
