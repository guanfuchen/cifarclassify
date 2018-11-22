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

from cifarclassify.utils import imagenet_utils

class mobilenet_conv_bn_relu(nn.Module):
    """
    :param
    """
    def __init__(self, in_channels, out_channels, stride):
        super(mobilenet_conv_bn_relu, self).__init__()
        self.cbr_seq = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.cbr_seq(x)
        return x


class mobilenet_conv_dw_relu(nn.Module):
    """
    :param
    """
    def __init__(self, in_channels, out_channels, stride):
        super(mobilenet_conv_dw_relu, self).__init__()
        self.cbr_seq = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.cbr_seq(x)
        return x


class MobileNet(nn.Module):
    """
    :param
    """
    def __init__(self, n_classes=1000):
        super(MobileNet, self).__init__()
        self.conv1_bn = mobilenet_conv_bn_relu(3, 32, 2)
        self.conv2_dw = mobilenet_conv_dw_relu(32, 64, 1)
        self.conv3_dw = mobilenet_conv_dw_relu(64, 128, 2)
        self.conv4_dw = mobilenet_conv_dw_relu(128, 128, 1)
        self.conv5_dw = mobilenet_conv_dw_relu(128, 256, 2)
        self.conv6_dw = mobilenet_conv_dw_relu(256, 256, 1)
        self.conv7_dw = mobilenet_conv_dw_relu(256, 512, 2)
        self.conv8_dw = mobilenet_conv_dw_relu(512, 512, 1)
        self.conv9_dw = mobilenet_conv_dw_relu(512, 512, 1)
        self.conv10_dw = mobilenet_conv_dw_relu(512, 512, 1)
        self.conv11_dw = mobilenet_conv_dw_relu(512, 512, 1)
        self.conv12_dw = mobilenet_conv_dw_relu(512, 512, 1)
        self.conv13_dw = mobilenet_conv_dw_relu(512, 1024, 2)
        self.conv14_dw = mobilenet_conv_dw_relu(1024, 1024, 1)
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, n_classes)
        self.init_weights(pretrained=True)

    def init_weights(self, pretrained=False):
        model_checkpoint_path = os.path.expanduser('~/.torch/models/mobilenet_sgd_rmsprop_69.526.tar')
        if os.path.exists(model_checkpoint_path):
            model_checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
            pretrained_dict = model_checkpoint['state_dict']

            model_dict = self.state_dict()

            # print(model_dict.keys())
            # print(pretrained_dict.keys())
            model_dict_keys = model_dict.keys()

            new_dict = {}
            for dict_index, (k, v) in enumerate(pretrained_dict.items()):
                # print(dict_index)
                # print(k)
                new_k = model_dict_keys[dict_index]
                new_v = v
                new_dict[new_k] = new_v
            model_dict.update(new_dict)
            self.load_state_dict(model_dict)

    def forward(self, x):
        """
        :param x:
        :return:
        """
        x = self.conv1_bn(x)
        x = self.conv2_dw(x)
        x = self.conv3_dw(x)
        x = self.conv4_dw(x)
        x = self.conv5_dw(x)
        x = self.conv6_dw(x)
        x = self.conv7_dw(x)
        x = self.conv8_dw(x)
        x = self.conv9_dw(x)
        x = self.conv10_dw(x)
        x = self.conv11_dw(x)
        x = self.conv12_dw(x)
        x = self.conv13_dw(x)
        x = self.conv14_dw(x)
        x = self.avg_pool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    n_classes = 1000
    model = MobileNet(n_classes=n_classes)
    model.eval()
    input_data = misc.imread('../../../data/cat.jpg')
    # 按照imagenet的图像格式预处理
    input_data = imagenet_utils.imagenet_preprocess(input_data)

    x = Variable(torch.FloatTensor(torch.from_numpy(input_data)))
    y = Variable(torch.LongTensor(np.ones(1, dtype=np.int)))

    start = time.time()
    pred = model(x)
    end = time.time()
    print("MobileNet forward time:", end - start)

    imagenet_utils.get_imagenet_label(pred)
