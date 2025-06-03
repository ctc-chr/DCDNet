import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from modules.SpatialAttention import SpatialAttention
from modules.TtS import TtS

__all__ = ['resnet50', 'resnet101', 'resnet152', 'resnet200']

sf = torch.autograd.Variable(torch.zeros(1, 3, 64, 112, 112))


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        # 断言法：kernel_size必须为3或7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 三元操作：如果kernel_size的值等于7，则padding被设置为3；否则（即kernel_size的值为3），padding被设置为1。
        padding = 3 if kernel_size == 7 else 1
        # 定义一个卷积层，输入通道数为2，输出通道数为1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, is_middle):
        retain = x
        # (N, C, T, H, W)，dim=1沿着通道维度C，计算张量的平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (N, 1, T, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (N, 1, T, H, W)
        # 将平均值和最大值在通道维度上拼接
        x = torch.cat([avg_out, max_out], dim=1)  # (N, 2, T, H, W)

        if is_middle == 0:
            x = self.conv1(x)  # (N, 1, T, H, W)
        else:
            x = self.conv1(x) + sf  # (N, 1, T, H, W)
            # x = self.conv1(x)  # (N, 1, T, H, W)
        return retain * self.sigmoid(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1, is_fast=0, is_middle=0):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        # self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.is_fast = is_fast
        self.is_middle = is_middle
        # self.psap = PSAP(planes * 4)
        self.psacp = PSC(planes * 4)

        self.sa = SpatialAttention(3)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # if self.is_fast == 1:
        #     # out = self.psap(out)
        #     out = self.psacp(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.is_fast == 1:
            out = self.psacp(out,sf,self.is_middle)
            # out = self.sa(out, self.is_middle)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class SlowFast(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], class_num=10, dropout=0.5):
        super(SlowFast, self).__init__()
        self.fast_inplanes = 8
        self.fast_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = nn.BatchNorm3d(8)
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast_res2 = self._make_layer_fast(block, 8, layers[0], head_conv=3)
        self.fast_res3 = self._make_layer_fast(
            block, 16, layers[1], stride=2, head_conv=3)
        self.fast_res4 = self._make_layer_fast(
            block, 32, layers[2], stride=2, head_conv=3)
        self.fast_res5 = self._make_layer_fast(
            block, 64, layers[3], stride=2, head_conv=3)

        fast_ratio = 2
        self.lateral_fast_p1 = nn.Conv3d(8, 8 * fast_ratio, kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False,
                                         padding=(2, 0, 0))
        self.lateral_fast_res2 = nn.Conv3d(32, 32 * fast_ratio, kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False,
                                           padding=(2, 0, 0))
        self.lateral_fast_res3 = nn.Conv3d(64, 64 * fast_ratio, kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False,
                                           padding=(2, 0, 0))
        self.lateral_fast_res4 = nn.Conv3d(128, 128 * fast_ratio, kernel_size=(5, 1, 1), stride=(8, 1, 1), bias=False,
                                           padding=(2, 0, 0))

        self.middle_inplanes = 8

        self.middle_conv1 = nn.Conv3d(3, 8, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.middle_bn1 = nn.BatchNorm3d(8)
        self.middle_relu = nn.ReLU(inplace=True)
        self.middle_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.sf_maxpool1 = nn.MaxPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.sf_maxpool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.sf_maxpool3 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.middle_res2 = self._make_layer_middle(block, 8, layers[0], head_conv=3)
        self.middle_res3 = self._make_layer_middle(
            block, 16, layers[1], stride=2, head_conv=3)
        self.middle_res4 = self._make_layer_middle(
            block, 32, layers[2], stride=2, head_conv=3)
        self.middle_res5 = self._make_layer_middle(
            block, 64, layers[3], stride=2, head_conv=3)

        middle_ratio = 1
        self.lateral_middle_p1 = nn.Conv3d(8, 8 * middle_ratio, kernel_size=(5, 1, 1), stride=(2, 1, 1), bias=False,
                                           padding=(2, 0, 0))
        self.lateral_middle_res2 = nn.Conv3d(32, 32 * middle_ratio, kernel_size=(5, 1, 1), stride=(2, 1, 1), bias=False,
                                             padding=(2, 0, 0))
        self.lateral_middle_res3 = nn.Conv3d(64, 64 * middle_ratio, kernel_size=(5, 1, 1), stride=(2, 1, 1), bias=False,
                                             padding=(2, 0, 0))
        self.lateral_middle_res4 = nn.Conv3d(128, 128 * middle_ratio, kernel_size=(5, 1, 1), stride=(2, 1, 1),
                                             bias=False,
                                             padding=(2, 0, 0))
        self.ttsPre = TtS(3, 1)

        self.tts0 = TtS(3,2)
        self.tts1 = TtS(3,2)
        self.tts2 = TtS(32,2)
        self.tts3 = TtS(64,2)
        self.tts4 = TtS(128,2)

        self.slow_inplanes = 64 + 64 // 8 * (fast_ratio + middle_ratio)
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.slow_res2 = self._make_layer_slow(block, 64, layers[0], head_conv=1)
        self.slow_res3 = self._make_layer_slow(
            block, 128, layers[1], stride=2, head_conv=1)
        self.slow_res4 = self._make_layer_slow(
            block, 256, layers[2], stride=2, head_conv=3)
        self.slow_res5 = self._make_layer_slow(
            block, 512, layers[3], stride=2, head_conv=3)
        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(self.fast_inplanes * 2 + 2048, class_num, bias=False)

    def forward(self, input):
        fast, lateral1 = self.FastPath(input[:, :, ::2, :, :])
        middle, lateral2 = self.MiddlePath(input[:, :, ::8, :, :])
        slow = self.SlowPath(input[:, :, ::16, :, :], lateral1, lateral2)
        x = torch.cat([slow, fast, middle], dim=1)
        x = self.dp(x)
        x = self.fc(x)
        return x

    def SlowPath(self, input, lateral1, lateral2):
        x = self.slow_conv1(input)
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x)
        # x = torch.cat([x, x0], dim=1)
        x = torch.cat([x, lateral1[0]], dim=1)
        x = torch.cat([x, lateral2[0]], dim=1)
        x = self.slow_res2(x)
        # x = torch.cat([x, x1], dim=1)
        x = torch.cat([x, lateral1[1]], dim=1)
        x = torch.cat([x, lateral2[1]], dim=1)
        x = self.slow_res3(x)
        # x = torch.cat([x, x2], dim=1)
        x = torch.cat([x, lateral1[2]], dim=1)
        x = torch.cat([x, lateral2[2]], dim=1)
        x = self.slow_res4(x)
        # x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, lateral1[3]], dim=1)
        x = torch.cat([x, lateral2[3]], dim=1)
        x = self.slow_res5(x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x

    def FastPath(self, input):
        lateral = []
        x = self.fast_conv1(input)
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_maxpool(x)
        lateral_p = self.lateral_fast_p1(pool1)
        lateral.append(lateral_p)

        res2 = self.fast_res2(pool1)
        lateral_res2 = self.lateral_fast_res2(res2)
        lateral.append(lateral_res2)

        res3 = self.fast_res3(res2)
        lateral_res3 = self.lateral_fast_res3(res3)
        lateral.append(lateral_res3)

        res4 = self.fast_res4(res3)
        lateral_res4 = self.lateral_fast_res4(res4)
        lateral.append(lateral_res4)

        res5 = self.fast_res5(res4)
        x = nn.AdaptiveAvgPool3d(1)(res5)
        x = x.view(-1, x.size(1))

        return x, lateral

    def MiddlePath(self, input):
        lateral = []

        global sf
        input = self.ttsPre(input)
        sf = self.sf_maxpool1(input)
        sf = self.tts1(sf)

        x = self.middle_conv1(input)
        x = self.middle_bn1(x)
        x = self.middle_relu(x)
        pool1 = self.middle_maxpool(x)
        lateral_p = self.lateral_middle_p1(pool1)
        lateral.append(lateral_p)

        res2 = self.middle_res2(pool1)
        lateral_res2 = self.lateral_middle_res2(res2)
        lateral.append(lateral_res2)

        sf = self.sf_maxpool2(res2)
        sf = self.tts2(sf)

        res3 = self.middle_res3(res2)
        lateral_res3 = self.lateral_middle_res3(res3)
        lateral.append(lateral_res3)

        sf = self.sf_maxpool2(res3)
        sf = self.tts3(sf)

        res4 = self.middle_res4(res3)
        lateral_res4 = self.lateral_middle_res4(res4)
        lateral.append(lateral_res4)

        sf = self.sf_maxpool3(res4)
        sf = self.tts4(sf)

        res5 = self.middle_res5(res4)
        x = nn.AdaptiveAvgPool3d(1)(res5)
        x = x.view(-1, x.size(1))

        sf = 0
        return x, lateral

    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1, is_fast=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.fast_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample, head_conv=head_conv, is_fast=is_fast))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, head_conv=head_conv, is_fast=is_fast))
        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1, is_fast=0):
        downsample = None
        if stride != 1 or self.slow_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.slow_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.slow_inplanes, planes, stride, downsample, head_conv=head_conv, is_fast=is_fast))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, head_conv=head_conv, is_fast=is_fast))

        self.slow_inplanes = planes * block.expansion + planes * block.expansion // 8 * 3
        return nn.Sequential(*layers)

    def _make_layer_middle(self, block, planes, blocks, stride=1, head_conv=1, is_fast=1, is_middle=1):
        downsample = None
        if stride != 1 or self.middle_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.middle_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1, stride, stride),
                    bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.middle_inplanes, planes, stride, downsample, head_conv=head_conv, is_fast=is_fast,
                            is_middle=is_middle))
        self.middle_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.middle_inplanes, planes, head_conv=head_conv, is_fast=is_fast, is_middle=is_middle))
        return nn.Sequential(*layers)


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = SlowFast(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


if __name__ == "__main__":
    num_classes = 31
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, 64, 224, 224))
    model = resnet50(class_num=num_classes)
    output = model(input_tensor)
    # print(model)
