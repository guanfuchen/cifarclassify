#!/usr/bin/python
# -*- coding: UTF-8 -*-
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

from cifarclassify.utils import imagenet_utils

class AlexNet(nn.Module):
    """
    :param
    """
    def __init__(self, n_classes=10):
        super(AlexNet, self).__init__()
        # features和classifier的结构和vgg16等类似
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    n_classes = 10
    model = AlexNet(n_classes=n_classes)

    # input = misc.imread('../../data/cat.jpg')
    # 按照imagenet的图像格式预处理
    # input = imagenet_utils.imagenet_preprocess(input)

    x = Variable(torch.randn(1, 3, 32, 32))
    # x = Variable(torch.FloatTensor(torch.from_numpy(input)))
    y = Variable(torch.LongTensor(np.ones(1, dtype=np.int)))
    # print(x.shape)
    start = time.time()
    pred = model(x)
    end = time.time()
    print("AlexNet forward time:", end-start)

    # imagenet_utils.get_imagenet_label(pred)
