# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import time
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models
import numpy as np
import os
from scipy import misc
import numpy as np
import scipy
import matplotlib.pyplot as plt
from torch.utils import model_zoo

from cifarclassify.utils import imagenet_utils


class Conv_BN_Relu(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, pad=1, use_relu=True):
        super(Conv_BN_Relu, self).__init__()
        self.use_relu = use_relu
        if self.use_relu:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.convs(x)
        return out


class StemBlock(nn.Module):
    def __init__(self, inp=3, num_init_features=32):
        super(StemBlock, self).__init__()

        self.stem_1 = Conv_BN_Relu(inp, num_init_features, 3, 2, 1)
        self.stem_2a = Conv_BN_Relu(num_init_features, int(num_init_features / 2), 1, 1, 0)
        self.stem_2b = Conv_BN_Relu(int(num_init_features / 2), num_init_features, 3, 2, 1)
        self.stem_2c = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem_3 = Conv_BN_Relu(num_init_features * 2, num_init_features, 1, 1, 0)

    def forward(self, x):

        # --------------stem_1--------------
        stem_1_out = self.stem_1(x)
        # --------------stem_1--------------

        # --------------stem_2--------------
        stem_2a_out = self.stem_2a(stem_1_out)
        stem_2b_out = self.stem_2b(stem_2a_out)

        stem_2c_out = self.stem_2c(stem_1_out)
        # --------------stem_2--------------

        # --------------stem_3--------------
        out = self.stem_3(torch.cat((stem_2b_out, stem_2c_out), 1))
        # --------------stem_3--------------

        return out


class DenseBlock(nn.Module):
    def __init__(self, inp, inter_channel, growth_rate):
        super(DenseBlock, self).__init__()
        # print('inter_channel:', inter_channel)
        # print('growth_rate:', growth_rate)

        self.cb1_a = Conv_BN_Relu(inp, inter_channel, 1, 1, 0)
        self.cb1_b = Conv_BN_Relu(inter_channel, growth_rate, 3, 1, 1)

        self.cb2_a = Conv_BN_Relu(inp, inter_channel, 1, 1, 0)
        self.cb2_b = Conv_BN_Relu(inter_channel, growth_rate, 3, 1, 1)
        self.cb2_c = Conv_BN_Relu(growth_rate, growth_rate, 3, 1, 1)

    def forward(self, x):
        cb1_a_out = self.cb1_a(x)
        cb1_b_out = self.cb1_b(cb1_a_out)

        cb2_a_out = self.cb2_a(x)
        cb2_b_out = self.cb2_b(cb2_a_out)
        cb2_c_out = self.cb2_c(cb2_b_out)

        out = torch.cat((x, cb1_b_out, cb2_c_out), 1)  # dense

        return out


class TransitionBlock(nn.Module):
    def __init__(self, inp, oup, with_pooling=True):
        super(TransitionBlock, self).__init__()
        if with_pooling:
            self.tb = nn.Sequential(Conv_BN_Relu(inp, oup, 1, 1, 0),
                                    nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.tb = Conv_BN_Relu(inp, oup, 1, 1, 0)

    def forward(self, x):
        out = self.tb(x)
        return out


class PeleeNet(nn.Module):
    def __init__(self, n_classes=1000, num_init_features=32, growthRate=32, nDenseBlocks=[3, 4, 8, 6], bottleneck_width=[1, 2, 4, 4], pretrained=False):
        super(PeleeNet, self).__init__()

        self.stage = nn.Sequential()
        self.n_classes = n_classes
        self.num_init_features = num_init_features

        inter_channel = list()
        total_filter = list()
        dense_inp = list()

        self.half_growth_rate = int(growthRate / 2)

        # building stemblock
        self.stage.add_module('stage_0', StemBlock(3, num_init_features))

        #
        for i, b_w in enumerate(bottleneck_width):

            inter_channel.append(int(self.half_growth_rate * b_w / 4) * 4)  # different stage different inter channel

            if i == 0:
                total_filter.append(num_init_features + growthRate * nDenseBlocks[i])
                dense_inp.append(self.num_init_features)
            else:
                total_filter.append(total_filter[i - 1] + growthRate * nDenseBlocks[i])
                dense_inp.append(total_filter[i - 1])

            if i == len(nDenseBlocks) - 1:
                # 最后一层不加池化层
                with_pooling = False
            else:
                with_pooling = True

            # building middle stageblock
            self.stage.add_module('stage_{}'.format(i + 1), self._make_dense_transition(dense_inp[i], total_filter[i], inter_channel[i], nDenseBlocks[i], with_pooling=with_pooling))

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(total_filter[len(nDenseBlocks) - 1], self.n_classes)
        )

        self._initialize_weights()

    def _make_dense_transition(self, dense_inp, total_filter, inter_channel, ndenseblocks, with_pooling=True):
        layers = []

        for i in range(ndenseblocks):
            layers.append(DenseBlock(dense_inp, inter_channel, self.half_growth_rate))
            dense_inp += self.half_growth_rate * 2

        # Transition Layer without Compression
        layers.append(TransitionBlock(dense_inp, total_filter, with_pooling))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.stage(x)

        # global average pooling layer
        x = F.avg_pool2d(x, kernel_size=7)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        # out = F.log_softmax(x, dim=1)

        return out

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


def main():
    n_classes = 1000
    model = PeleeNet(n_classes=n_classes, pretrained=False)
    model.eval()

    input_data = misc.imread('../../../data/cat.jpg')
    # 按照imagenet的图像格式预处理
    input_data = imagenet_utils.imagenet_preprocess(input_data)

    # x = Variable(torch.randn(1, 3, 224, 224))
    x = Variable(torch.FloatTensor(torch.from_numpy(input_data)))
    y = Variable(torch.LongTensor(np.ones(1, dtype=np.int)))
    # print(x.shape)
    start = time.time()
    pred = model(x)
    end = time.time()
    print("PeleeNet forward time:", end-start)
    imagenet_utils.get_imagenet_label(pred)


if __name__ == '__main__':
    main()
