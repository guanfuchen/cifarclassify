# -*- coding: utf-8 -*-
import time
import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import os
from scipy import misc
import numpy as np
import scipy
import matplotlib.pyplot as plt
import torchvision

from cifarclassify.utils import imagenet_utils

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3卷积（padding）
    :param in_planes:
    :param out_planes:
    :param stride:
    :return:
    """
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    """
    BasicBlock
    """
    expansion = 1  # 最后一层是前一层的expansion倍

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes=inplanes, out_planes=planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes)
        self.bn2 = nn.BatchNorm2d(num_features=planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        :param x:
        :return:
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        :param x:
        :return:
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ Constructs  a ResNet template
    """
    def __init__(self, block, layers, n_classes=1000):
        """
        :param block: BasicBlock or Bottleneck
        :param layers:
        :param num_classes:
        """
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)  # padding=(kernel_size-1)/2 bias=False
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # padding=(kernel_size-1)/2
        self.layer1 = self._make_layer(block=block, planes=64, blocks=layers[0])
        self.layer2 = self._make_layer(block=block, planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block=block, planes=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block=block, planes=512, blocks=layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(in_features=512*block.expansion, out_features=n_classes)


        # 初始化卷积层和BN层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # stride = 1表示第一层，不需要下采样（使用maxpool下采样了），stride = 2表示第二，三，四层，需要下采样
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.inplanes, out_channels=planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=planes * block.expansion)
            )

        layers = []
        # blocks中的第一层决定是否有下采样，其中第一个block的第一层没有下采样，其他block的第一层有下采样
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        """
        :param x:
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        # print('x.size():{}'.format(x.size()))

        x = self.layer1(x)
        # print('x.size():{}'.format(x.size()))
        x = self.layer2(x)
        # print('x.size():{}'.format(x.size()))
        x = self.layer3(x)
        # print('x.size():{}'.format(x.size()))
        x = self.layer4(x)
        # print('x.size():{}'.format(x.size()))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model

    :param pretrained: If True, returns a model pre-trained on ImageNet
    :param kwargs:
    """
    model = ResNet(BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in {'fc.bias', 'fc.weight'}}
        pretrained_dict.update(model.state_dict())
        # print(pretrained_dict.keys())
        model.load_state_dict(pretrained_dict)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


if __name__ == '__main__':
    model = torchvision.models.resnet152(pretrained=True)
    # model = torchvision.models.resnet34(pretrained=True)
    # model = resnet34(pretrained=True)
    model.eval()

    input_data = misc.imread('../../../data/cat.jpg')
    # 按照imagenet的图像格式预处理
    input_data = imagenet_utils.imagenet_preprocess(input_data)


    x = Variable(torch.FloatTensor(torch.from_numpy(input_data)))
    y = Variable(torch.LongTensor(np.ones(1, dtype=np.int)))
    # print(x.shape)
    start = time.time()
    pred = model(x)
    end = time.time()
    print("resnet152 forward time:", end-start)

    imagenet_utils.get_imagenet_label(pred)
