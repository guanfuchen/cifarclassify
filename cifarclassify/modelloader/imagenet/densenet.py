# -*- coding: utf-8 -*-
# 下面代码来自于[densenet.pytorch](https://github.com/bamos/densenet.pytorch)

import torch
import time
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math


class Bottleneck(nn.Module):
    """
    DenseNet中的Bottleneck，由BN+ReLU+Conv(1x1)+BN+ReLU+Conv(3x3)组成，其中1x1的feature map通道数为4*growthRate
    """
    def __init__(self, nChannels, growthRate):
        """
        :param nChannels: 输入该Bottleneck的通道数，growthRate是每一个网络层的输出通道数k
        :param growthRate:
        """
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """
        Bottleneck前向传播
        :param x:
        :return:
        """
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    """
    SingleLayer是DenseNet中每一个Block中最小网络单位，由BN+ReLU+Conv(3x3)组成
    """
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """
        SingleLayer前向传播
        :param x: 这里的输入是经过先前所有网络层concat而成的feature map
        :return:
        """
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    """
    DenseNet网络中Block间的网络层组成，主要作用是改变feature map的大小，由BN+ReLU+Conv+MaxPooling组成
    """
    def __init__(self, nChannels, nOutChannels):
        """
        :param nChannels: 输入到Transition的通道数
        :param nOutChannels: Transition输出的通道数，可以一致k，也可以变化
        """
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)

    def forward(self, x):
        """
        Transition前向传播
        :param x: DenseNet先前block的feature map
        :return:
        """
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    """
    DenseNet网络构造
    """
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        """
        :param growthRate: 其中Block中每一个网络层输出的通道数，也就是k
        :param depth: depth是DenseBlocks的深度参考
        :param reduction: reduction是每一个transition是否降低输出通道数，也就是growthRate*reduction
        :param nClasses:
        :param bottleneck:
        """
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)

        # 网络中参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        """
        make dense是DenseNet中的一个Dense Block
        :param nChannels: Dense Block的输入通道
        :param growthRate: Dense Block中的growthRate
        :param nDenseBlocks: Dense Block中的layer数量
        :param bottleneck: 是否使用bottleneck，即1x1+3x3，不使用则为3x3
        :return:
        """
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                # 如果使用bottleneck那么网络层中使用bottleneck，块中每一个输入通道数目为nChannels+l*growthRate
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        DenseNet前向传播
        :param x:
        :return:
        """
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = self.fc(out)
        # out = F.log_softmax(out)
        return out


if __name__ == '__main__':
    n_classes = 10
    model = DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=n_classes)
    model.eval()
    # model.init_vgg16()
    x = Variable(torch.randn(1, 3, 32, 32))
    y = Variable(torch.LongTensor(np.ones(1, dtype=np.int)))
    # print(x.shape)
    start = time.time()
    pred = model(x)
    end = time.time()
    # print(model)
    print("DenseNet forward time:", end - start)
    # start = time.time()
    # vgg_16 = models.vgg16(pretrained=False)
    # pred = vgg_16(x)
    # end = time.time()
    # print("vgg16 forward time:", end-start)