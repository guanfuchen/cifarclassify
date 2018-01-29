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
    def __init__(self, n_classes=1000):
        super(AlexNet, self).__init__()
        # features和classifier的结构和vgg16等类似
        self.features = nn.Sequential(
            # (224-11+2*2)/4+1=55
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # (55-3)/2+1=27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (27-5+2*2)/1+1=27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # (27-3)/2+1=13
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (13-3+2*1)/1+1=13
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # (13-3+2*1)/1+1=13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # (13-3+2*1)/1+1=13
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # (13-3)/2+1=6
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 分类器使用Linear全连接层，特征层使用Conv2d卷积层
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # 特征层的输出为256*6*6，转换为4096的输出
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    n_classes = 1000
    model = AlexNet(n_classes=n_classes)
    model_pretrain_filename = os.path.expanduser('~/Data/alexnet-owt-4df8aa71.pth')
    if os.path.exists(model_pretrain_filename):
        model.load_state_dict(torch.load(model_pretrain_filename))

    input = misc.imread('../../data/cat.jpg')
    # 按照imagenet的图像格式预处理
    input = imagenet_utils.imagenet_preprocess(input)

    # x = Variable(torch.randn(1, 3, 224, 224))
    x = Variable(torch.FloatTensor(torch.from_numpy(input)))
    y = Variable(torch.LongTensor(np.ones(1, dtype=np.int)))
    # print(x.shape)
    start = time.time()
    pred = model(x)
    end = time.time()
    print("AlexNet forward time:", end-start)

    imagenet_utils.get_imagenet_label(pred)
