#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import skimage
from scipy import misc


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def image_crop_resize(img, input_height, input_width):
    img_scaled = None
    aspect = img.shape[1] / float(img.shape[0])
    # print("Orginal aspect ratio: " + str(aspect))
    if aspect > 1:
        # landscape orientation - wide image
        res = int(aspect * input_height)
        img_scaled = misc.imresize(img, (input_width, res))
    if aspect < 1:
        # portrait orientation - tall image
        res = int(input_width / aspect)
        img_scaled = misc.imresize(img, (res, input_height))
    if aspect == 1:
        img_scaled = misc.imresize(img, (input_width, input_height))

    img_center = crop_center(img_scaled, 224, 224)
    return img_center
