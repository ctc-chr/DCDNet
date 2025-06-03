import math
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from config import params
from modules.SpatialAttention import SpatialAttention
from modules.PSAC import PSACP
from modules.PSAC_TtS_V2 import PSC
# from modules.PSAC_TtS_V3 import PSC
from modules.TtS_new import TtS

sf = torch.autograd.Variable(torch.zeros(1, 3, 64, 112, 112))


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

        self.sa = SpatialAttention(3)
        self.psacp = PSACP(3)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out = self.sa(out)
        # out = self.psacp(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # self.sa = SpatialAttention(3)
        # self.psacp = PSACP(4 * planes)
        # self.tts = TtS(planes,2)
        self.psc = PSC(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # tts = self.tts(out)
        # out = self.psc(out, tts, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        # out = self.sa(out) * out
        # out = self.psacp(out)
        # out = self.psc(out, sf, 1)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=params['num_classes']):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.ttsPre0 = TtS(3)
        self.ttsPre1 = TtS(64)
        self.ttsPre2 = TtS(256)
        self.ttsPre3 = TtS(512)
        self.ttsPre4 = TtS(1024)
        self.ttsPre5 = TtS(2048)
        # self.tts1 = TtS(64, 2)
        # self.tts2 = TtS(256, 2)
        # self.tts3 = TtS(512, 2)
        # self.tts4 = TtS(1024, 2)

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        # self.maxpool_s = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.maxpool_s1 = nn.Conv3d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool_s2 = nn.Conv3d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.maxpool_s3 = nn.Conv3d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False)

        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        global sf
        # x = self.ttsPre0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        # x_s = x
        # sf = self.tts1(x_s)
        # x = self.ttsPre1(x)
        x = self.layer1(x)
        # x_s = x
        # x_s = self.maxpool_s1(x_s)
        # sf = self.tts2(x_s)
        # x = x + self.ttsPre2(x) * self.alpha[0]

        x = self.layer2(x)
        # x_s = x
        # x_s = self.maxpool_s2(x_s)
        # sf = self.tts3(x_s)

        # x = x + self.ttsPre3(x) * self.alpha[0]
        x = self.layer3(x)

        # x_s = x
        # x_s = self.maxpool_s3(x_s)
        # sf = self.tts4(x_s)

        x = x + self.ttsPre4(x) * self.alpha[1]
        x = self.layer4(x)

        # x = x + self.ttsPre5(x) * self.alpha[2]

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # x = self.dropout(x)
        x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


if __name__ == '__main__':
    input_tensor = torch.autograd.Variable(torch.rand(1, 3, 32, 224, 224))
    # input_tensor = torch.ones([1, 3, 16, 28, 28])
    model = generate_model(50)

    # flops, params = profile(model, inputs=(input_tensor,))
    # print('********FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('********Params = ' + str(params / 1000 ** 2) + 'M')
    #
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    # print('Trainable Parameters: %.3fM' % parameters)

    output = model(input_tensor)
    # print(output[0])
    print(output.shape)
    # print(model)
    # begin = time.time()
    # y = model(input_tensor)
    # end = time.time()
    # print(end - begin)
