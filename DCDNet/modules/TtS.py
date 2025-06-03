import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TtS(nn.Module):
    def __init__(self, C, plan):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.bn = nn.BatchNorm3d(C)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.plan = plan

    def forward(self, input):
        B, C, T, H, W = input.shape

        if self.plan == 1:
            x0 = input[:, :, [0], :, :]
            x0_conv = self.conv1(x0)  # (N, C, 1, H, W)
            # x0_conv = self.bn(x0_conv)
            x0_conv = self.relu(x0_conv)
            avg_out = torch.mean(x0_conv, dim=1, keepdim=True)  # (N, 1, 1, H, W)
            output_x = self.sigmoid(avg_out)

            for t in range(1, T):
                x_front = input[:, :, [t - 1], :, :]  # (N, C, 1, H, W)
                x_self = input[:, :, [t], :, :]  # (N, C, 1, H, W)
                fx = x_self - x_front
                score = self.conv1(fx)
                # score = self.bn(score)
                score = self.relu(score)
                avg_out = torch.mean(score, dim=1, keepdim=True)  # (N, 1, 1, H, W)
                avg_out = self.sigmoid(avg_out)
                output_x = torch.cat([output_x, avg_out], dim=2)
            return output_x * input  # (N, 1, T, H, W)

        if self.plan == 2:
            x0 = input[:, :, [0], :, :]
            x0_conv = self.conv1(x0)  # (N, C, 1, H, W)
            # x0_conv = self.bn(x0_conv)
            x0_conv = self.relu(x0_conv)
            avg_out = torch.mean(x0_conv, dim=1, keepdim=True)  # (N, 1, 1, H, W)
            output_x = avg_out

            for t in range(1, T):
                x_front = input[:, :, [t - 1], :, :]  # (N, C, 1, H, W)
                x_self = input[:, :, [t], :, :]  # (N, C, 1, H, W)
                fx = x_self - x_front
                score = self.conv1(fx)
                # score = self.bn(score)
                score = self.relu(score)
                avg_out = torch.mean(score, dim=1, keepdim=True)  # (N, 1, 1, H, W)
                output_x = torch.cat([output_x, avg_out], dim=2)
            return output_x  # (N, 1, T, H, W)




if __name__ == '__main__':
    x = torch.rand([1, 3, 4, 28, 28])
    model = TtS(3,1)

    y = model(x)
    print(y.shape)
