#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch.nn as nn
import time
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models
import numpy as np
import math
import os
from scipy import misc
import numpy as np
import scipy
import matplotlib.pyplot as plt


from cifarclassify.utils import numpy_utils
from cifarclassify.utils import imagenet_utils


# conv batchnorm
def conv_bn(inp, oup, stride):
    """
    :param inp:
    :param oup:
    :param stride:
    :return:
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    """
    :param inp:
    :param oup:
    :return:
    """
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


# 反向残差模块
class InvertedResidual(nn.Module):
    """
    :param
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        # 仅仅当stride==1和inp==oup时使用残差连接
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    :param
    """
    def __init__(self, n_classes=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        # 反向残差块设置
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # building last several layers
        # 构建最后几层
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features.append(nn.AvgPool2d(input_size/32))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.last_channel, n_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.features(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

if __name__ == '__main__':

    image_height, image_width, image_channel = (224, 224, 3)
    input = misc.imread('../../../data/cat.jpg')
    # 按照imagenet的图像格式预处理
    input = imagenet_utils.imagenet_preprocess(input)

    n_classes = 1000
    model = MobileNetV2(n_classes=n_classes)
    model.eval()
    # 训练模型为gpu模型
    # model.load_state_dict(torch.load(os.path.expanduser('~/Data/mobilenetv2.pth.tar'), map_location=lambda storage, loc: storage))
    # x = Variable(torch.randn(1, image_channel, image_height, image_width))
    x = Variable(torch.FloatTensor(torch.from_numpy(input)))
    y = Variable(torch.LongTensor(np.ones(1, dtype=np.int)))
    # print(x.shape)
    start = time.time()
    pred = model(x)
    end = time.time()
    print("MobileNetV2 forward time:", end-start)

    imagenet_utils.get_imagenet_label(pred)

