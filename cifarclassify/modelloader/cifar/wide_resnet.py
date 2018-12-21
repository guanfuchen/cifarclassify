# -*- coding: utf-8 -*-
# 参考代码[resnet.py](https://github.com/szagoruyko/wide-residual-networks/blob/master/pytorch/resnet.py)
import torch
import torch.nn.functional as F

import torch
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
from functools import partial
from nested_dict import nested_dict
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

from cifarclassify.utils import imagenet_utils


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


# conv params no*ni*k*k
def conv_params(ni, no, k=1):
    return kaiming_normal_(torch.Tensor(no, ni, k, k))


def linear_params(ni, no):
    return {'weight': kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}


def bnparams(n):
    return {'weight': torch.rand(n),
            'bias': torch.zeros(n),
            'running_mean': torch.zeros(n),
            'running_var': torch.ones(n)}


def data_parallel(f, input, params, mode, device_ids, output_device=None):
    assert isinstance(device_ids, list)
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, mode)

    params_all = Broadcast.apply(device_ids, *params.values())
    params_replicas = [{k: params_all[i + j*len(params)] for i, k in enumerate(params.keys())}
                       for j in range(len(device_ids))]

    replicas = [partial(f, params=p, mode=mode)
                for p in params_replicas]
    inputs = scatter([input], device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def flatten(params):
    return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def batch_norm(x, params, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=params[base + '.running_mean'],
                        running_var=params[base + '.running_var'],
                        training=mode)


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)


def set_requires_grad_except_bn_(params):
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var'):
            # if not running_mean or running_var requires grad
            v.requires_grad = True

# def wide_resnet(depth, width, num_classes):
class wide_resnet(nn.Module):
    def __init__(self, depth, width, n_classes):
        super(wide_resnet, self).__init__()
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'  # 4 is for conv0
        self.n = (depth - 4) // 6
        widths = [int(v * width) for v in (16, 32, 64)]  # for normal resnet 16 32 64

        def gen_block_params(ni, no):
            return {
                'conv0': conv_params(ni, no, 3),
                'conv1': conv_params(no, no, 3),
                'bn0': bnparams(ni),
                'bn1': bnparams(no),
                'convdim': conv_params(ni, no, 1) if ni != no else None,
            }

        def gen_group_params(ni, no, count):
            return {'block%d' % i: gen_block_params(ni if i == 0 else no, no) for i in range(count)}

        # conv0+group0+goup1+group2
        self.flat_params = cast(flatten({
            'conv0': conv_params(3, 16, 3),  # input 2 output 16
            'group0': gen_group_params(16, widths[0], self.n),  # input 16 output widths[0]
            'group1': gen_group_params(widths[0], widths[1], self.n),
            'group2': gen_group_params(widths[1], widths[2], self.n),
            'bn': bnparams(widths[2]),
            'fc': linear_params(widths[2], n_classes),
        }))

        # except bn requires grad
        set_requires_grad_except_bn_(self.flat_params)

    def block(self, x, params, base, mode, stride):
        o1 = F.relu(batch_norm(x, params, base + '.bn0', mode), inplace=True)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(batch_norm(y, params, base + '.bn1', mode), inplace=True)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def group(self, o, params, base, mode, stride):
        for i in range(self.n):
            o = self.block(o, params, '%s.block%d' % (base, i), mode, stride if i == 0 else 1)
        return o


    def forward(self, input):
        x = F.conv2d(input, self.flat_params['conv0'], padding=1)
        g0 = self.group(x, self.flat_params, 'group0', self.training, 1)
        g1 = self.group(g0, self.flat_params, 'group1', self.training, 2)
        g2 = self.group(g1, self.flat_params, 'group2', self.training, 2)
        o = F.relu(batch_norm(g2, self.flat_params, 'bn', self.training))
        # print('o.shape:', o.shape)
        o = F.avg_pool2d(o, 8, 1, 0)
        # print('o.shape:', o.shape)
        o = o.view(o.size(0), -1)
        # print('o.shape:', o.shape)
        o = F.linear(o, self.flat_params['fc.weight'], self.flat_params['fc.bias'])
        return o
    # return f, flat_params
    # return f

def wide_resnet_28_10(n_classes):
    depth = 28
    width = 10
    n_classes = n_classes
    return wide_resnet(depth, width, n_classes)

def wide_resnet_16_8(n_classes):
    depth = 16
    width = 8
    n_classes = n_classes
    return wide_resnet(depth, width, n_classes)

if __name__ == '__main__':
    n_classes = 10
    # model = wide_resnet_28_10(n_classes=n_classes)
    model = wide_resnet_16_8(n_classes)

    x = Variable(torch.randn(1, 3, 32, 32))
    y = Variable(torch.LongTensor(np.ones(1, dtype=np.int)))
    # print(x.shape)
    start = time.time()
    pred = model(x)
    # print('pred.shape', pred.shape)
    end = time.time()
    print("AlexNet forward time:", end-start)
