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
import visdom
import numpy as np


def train(args):
    if args.vis:
        vis = visdom.Visdom()
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

    # testset = torchvision.datasets.CIFAR10(root=os.path.expanduser('~/Data'), train=False, download=True, transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

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
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # 0<epoch<step_size lr=base_lr
    # step_size<epoch<2*step_size lr=base_lr*gamma
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.1)

    for epoch in range(start_epoch, 20000, 1):
        print('epoch:', epoch)
        scheduler.step()
        # loss_epoch = 0
        # loss_avg_epoch = 0
        # data_count = 0

        if args.vis:
            win = 'lr step'
            lr = scheduler.get_lr()
            lr = np.array(lr)
            # print('lr:', lr)
            win_res = vis.line(X=np.ones(1) * epoch, Y=lr, win=win, update='append', name=win)
            if win_res != win:
                vis.line(X=np.ones(1) * epoch, Y=lr, win=win, name=win)

        for i, (imgs, labels) in enumerate(trainloader):
            # data_count = i
            # print(i)
            imgs, labels = Variable(imgs), Variable(labels)

            # 训练优化参数
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            # print('loss:', loss)
            loss_numpy = loss.data.numpy()
            loss_numpy = loss_numpy[np.newaxis]
            # print('loss_numpy.shape:', loss_numpy.shape)
            # print('loss_numpy:', loss_numpy)
            # loss_epoch += loss_numpy
            if args.vis:
                win = 'loss iterations'
                # print(trainset.__len__())
                # print(epoch * trainset.__len__() / (args.batch_size * 1.0) + i)
                win_res = vis.line(X=np.ones(1) * (epoch*trainset.__len__()/(args.batch_size*1.0) + i), Y=loss_numpy, win=win, update='append', name=win)
                if win_res != win:
                    vis.line(X=np.ones(1) * (epoch*trainset.__len__()/(args.batch_size*1.0) + i), Y=loss_numpy, win=win, name=win)
            loss.backward()

            optimizer.step()
            # if i == 10:
            #     break

        # 输出一个周期后的loss
        # loss_avg_epoch = loss_epoch / (data_count * args.batch_size * 1.0)
        # print('loss_avg_epoch:', loss_avg_epoch)

        # 存储模型
        if args.save_model and epoch%args.save_epoch==0 and epoch != 0:
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
    parser.add_argument('--lr', type=float, default=1e-1, help='train learning rate [ 0.01 ]')
    parser.add_argument('--vis', type=bool, default=False, help='visualize the training results [ False ]')
    args = parser.parse_args()
    print(args)
    train(args)
