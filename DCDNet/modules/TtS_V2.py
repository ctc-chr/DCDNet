import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TtS(nn.Module):
    def __init__(self, C):
        super().__init__()
        reduction_channel = C // 16

        self.conv1 = nn.Conv3d(in_channels=C, out_channels=C, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.bn = nn.BatchNorm3d(C)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.spatial_aggregation1 = nn.Conv3d(C, C, kernel_size=(3, 3, 3),
                                              padding=(1, 1, 1), groups=C)
        self.spatial_aggregation2 = nn.Conv3d(C, C, kernel_size=(3, 3, 3),
                                              padding=(2, 1, 1), dilation=(2, 1, 1), groups=C)
        self.spatial_aggregation3 = nn.Conv3d(C, C, kernel_size=(3, 3, 3),
                                              padding=(3, 1, 1), dilation=(3, 1, 1), groups=C)

        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)

    def forward(self, input):
        B, C, T, H, W = input.shape
        input1 = self.spatial_aggregation1(input)
        input2 = self.spatial_aggregation2(input)
        input3 = self.spatial_aggregation3(input)

        x1_0 = input1[:, :, [0], :, :]
        x1_output = x1_0
        for t in range(1, T):
            x1_front = input1[:, :, [t - 1], :, :]  # (N, C, 1, H, W)
            x1_self = input1[:, :, [t], :, :]  # (N, C, 1, H, W)
            x1_score = x1_self - x1_front
            x1_output = torch.cat([x1_output, x1_score], dim=2)

        if T >= 2:
            x2_0 = input2[:, :, [0], :, :]
            x2_1 = input2[:, :, [1], :, :]
            x2_output = torch.cat([x2_0, x2_1], dim=2)
            for t in range(2, T):
                x2_front = input2[:, :, [t - 2], :, :]  # (N, C, 1, H, W)
                x2_self = input2[:, :, [t], :, :]  # (N, C, 1, H, W)
                x2_score = x2_self - x2_front
                x2_output = torch.cat([x2_output, x2_score], dim=2)
        else:
            x2_output = input2

        if T > 2:
            x3_0 = input3[:, :, [0], :, :]
            x3_1 = input3[:, :, [1], :, :]
            x3_2 = input3[:, :, [2], :, :]
            x3_output = torch.cat([x3_0, x3_1, x3_2], dim=2)

            for t in range(3, T):
                x3_front = input3[:, :, [t - 3], :, :]  # (N, C, 1, H, W)
                x3_self = input3[:, :, [t], :, :]  # (N, C, 1, H, W)
                x3_score = x3_self - x3_front
                x3_output = torch.cat([x3_output, x3_score], dim=2)
        elif T == 2:
            x3_0 = input3[:, :, [0], :, :]
            x3_1 = input3[:, :, [1], :, :]
            x3_output = torch.cat([x3_0, x3_1-x3_0], dim=2)
        else:
            x3_output = input3


        output1 = input * (self.sigmoid(x1_output) - 0.5) * self.weights[0]
        output2 = input * (self.sigmoid(x2_output) - 0.5) * self.weights[1]
        output3 = input * (self.sigmoid(x3_output) - 0.5) * self.weights[2]

        output = output1 + output2 + output3
        return output  # (N, 1, T, H, W)


if __name__ == '__main__':
    x = torch.rand([1, 3, 64, 28, 28])
    model = TtS(3)

    y = model(x)
    print(y.shape)