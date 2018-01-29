#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import pickle
import os

from cifarclassify.utils import numpy_utils

def get_imagenet_label(pred):
    """
    :param pred: torch.Variable，(1, 1000)
    :return: 在imagenet中的标签名
    """
    pred_np = pred.data.numpy()
    print('pred_np.shape:', pred_np.shape)
    pred_np = np.squeeze(pred_np, axis=0)
    pred_np_prob = numpy_utils.softmax(pred_np)
    # argsort是从小到大
    pred_np_argmax = np.argsort(pred_np)[::-1]
    pred_np_argmax_top5 = pred_np_argmax[:5]
    pred_np_prob_top5 = pred_np_prob[pred_np_argmax_top5]
    # print('pred_np_argmax.shape:', pred_np_argmax.shape)
    print('pred_np_argmax_top5:', pred_np_argmax_top5)
    print('pred_np_prob_top5:', pred_np_prob_top5)

    # 获取可读性的标签
    imagenet_label_file_path = os.path.expanduser('~/Data/imagenet1000_clsid_to_human.pkl')
    if os.path.exists(imagenet_label_file_path):
        label_name = pickle.load(open(imagenet_label_file_path, 'r'))
        pred_np_label_name_top5 = []
        for pred_np_argmax_top5_index in pred_np_argmax_top5:
            label = label_name[pred_np_argmax_top5_index]
            pred_np_label_name_top5.append(label)
        print('pred_np_label_name_top5:', pred_np_label_name_top5)


def imagenet_preprocess(input):
    """
    :param input: input numpy shape: (height, width, channel)
    :return: output numpy shape: (batch, channel, height, width)
    """
    image_height, image_width, image_channel = (224, 224, 3)
    # crop中心
    input = numpy_utils.image_crop_resize(input, image_height, image_width)
    # 直接resize
    # input = misc.imresize(input, (image_height, image_width))
    input = input[:, :, ::-1]
    # BGR
    input = input - [103.939, 116.779, 123.68]
    input = input * 0.017
    input = np.expand_dims(input, axis=0)
    input = input.astype(np.float32)
    input = input.transpose((0, 3, 1, 2))
    print(input.shape)
    return input