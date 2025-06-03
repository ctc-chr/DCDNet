import torch.nn as nn
import torch
import torch.nn.functional as f


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        # 断言法：kernel_size必须为3或7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 三元操作：如果kernel_size的值等于7，则padding被设置为3；否则（即kernel_size的值为3），padding被设置为1。
        padding = 3 if kernel_size == 7 else 1
        # 定义一个卷积层，输入通道数为2，输出通道数为1
        self.conv1 = nn.Conv3d(2, 1, (1, kernel_size, kernel_size), padding=(0, padding, padding), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # (N, C, T, H, W)，dim=1沿着通道维度C，计算张量的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 将平均值和最大值在通道维度上拼接
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


if __name__ == '__main__':
    model = SpatialAttention(kernel_size=7)
    inputs = torch.ones([2, 256, 4, 8, 8])
    outputs = model(inputs)
    print(outputs.shape)
