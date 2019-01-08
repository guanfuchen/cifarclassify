# -*- coding: utf-8 -*-
import torch
import os
import collections
import random

import cv2
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils import data
from torchvision import transforms
import glob


class BearingLoader(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, is_augment=False):
        self.root = root
        self.split = split
        self.img_size = (224, 224)  # (h, w)
        self.is_transform = is_transform
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.n_classes = 4
        self.files = collections.defaultdict(list)
        self.joint_augment_transform = None
        self.is_augment = is_augment

        file_list = glob.glob(root + '/*.jpg')
        file_list.sort()
        file_list_len = len(file_list)
        split_index = int(file_list_len*0.7)
        if self.split == 'train':
            self.files[split] = file_list[:split_index]
        elif self.split == 'val':
            self.files[split] = file_list[split_index:]

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_file_name = img_name[img_name.rfind('/') + 1:img_name.rfind('.')]
        # img_file_name = img_name[:img_name.rfind('.')]
        # print(img_file_name)

        img = Image.open(img_name)
        img = img.resize((self.img_size[1], self.img_size[0]))
        lbl = img_file_name[:img_file_name.index('_')]
        lbl = int(lbl)
        lbl -= 1 # 1-4 to 0-3
        # print(lbl)

        if self.is_augment:
            if self.joint_augment_transform is not None:
                img, lbl = self.joint_augment_transform(img, lbl)

        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.int32)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    # 转换HWC为CHW
    def transform(self, img, lbl):
        img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= self.mean
        img = img.astype(float) / 255.0
        # HWC -> CHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


def main():
    home_path = os.path.expanduser('~')
    local_path = os.path.join(home_path, 'Data/Bearing/dataset')
    batch_size = 4
    dst = BearingLoader(local_path, is_transform=True, is_augment=False)
    trainloader = data.DataLoader(dst, batch_size=batch_size, shuffle=True)
    for i, (imgs, labels) in enumerate(trainloader):
        print(i)
        print(imgs.shape)
        print(labels.shape)
        # if i == 0:
        image_list_len = imgs.shape[0]
        for image_list in range(image_list_len):
            img = imgs[image_list, :, :, :]
            img = img.numpy()
            img = np.transpose(img, (1, 2, 0))
            plt.subplot(image_list_len, 2, 2 * image_list + 1)
            plt.imshow(img)
        plt.show()
        if i == 0:
            break


if __name__ == '__main__':
    main()

