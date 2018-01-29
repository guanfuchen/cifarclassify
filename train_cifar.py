#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os

import torch
import torchvision
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import transforms
import argparse


def train(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=os.path.expanduser('~/Data'), train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=os.path.expanduser('~/Data'), train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    from cifarclassify.modelloader.cifar.alexnet import AlexNet
    start_epoch = 0

    if args.structure == 'AlexNet':
        model = AlexNet(n_classes=32)
    else:
        print('not valid model name')
        return

    if args.resume_model_state_dict != '':
        start_epoch_id1 = args.resume_model_state_dict.rfind('_')
        start_epoch_id2 = args.resume_model_state_dict.rfind('.')
        start_epoch = int(args.resume_model_state_dict[start_epoch_id1 + 1:start_epoch_id2])
        model.load_state_dict(torch.load(args.resume_model_state_dict))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch + 1, 20000, 1):
        loss_epoch = 0
        loss_avg_epoch = 0
        data_count = 0
        for i, (imgs, labels) in enumerate(trainloader):
            data_count = i
            print(i)
            imgs, labels = Variable(imgs), Variable(labels)

            # 训练优化参数
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            # print('loss:', loss)
            loss_numpy = loss.data.numpy()
            loss_epoch += loss_numpy
            loss.backward()

            optimizer.step()

        # 输出一个周期后的loss
        loss_avg_epoch = loss_epoch / (data_count * args.batch_size * 1.0)
        print('epoch:', epoch)
        print('loss_avg_epoch:', loss_avg_epoch)

        # 存储模型
        if args.save_model and epoch%args.save_epoch==0:
            torch.save(model.state_dict(), '{}_cifar10_{}.pt'.format(args.structure, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameter setting')
    parser.add_argument('--structure', type=str, default='AlexNet', help='use the net structure to segment [ AlexNet ]')
    parser.add_argument('--resume_model', type=str, default='', help='resume model path [ AlexNet_cifar10_0.pkl ]')
    parser.add_argument('--resume_model_state_dict', type=str, default='', help='resume model state dict path [ AlexNet_cifar10_0.pt ]')
    parser.add_argument('--save_model', type=bool, default=False, help='save model [ False ]')
    parser.add_argument('--save_epoch', type=int, default=1, help='save model after epoch [ 1 ]')
    parser.add_argument('--init_vgg16', type=bool, default=False, help='init model using vgg16 weights [ False ]')
    parser.add_argument('--dataset_path', type=str, default='', help='train dataset path [ /home/cgf/Data/CamVid ]')
    parser.add_argument('--data_augment', type=bool, default=False, help='enlarge the training data [ False ]')
    parser.add_argument('--batch_size', type=int, default=128, help='train dataset batch size [ 128 ]')
    parser.add_argument('--lr', type=float, default=1e-5, help='train learning rate [ 0.01 ]')
    parser.add_argument('--vis', type=bool, default=False, help='visualize the training results [ False ]')
    args = parser.parse_args()
    print(args)
    train(args)
