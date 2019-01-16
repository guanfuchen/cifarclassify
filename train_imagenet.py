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

from cifarclassify.dataloader.bearing_loader import BearingLoader
from cifarclassify.dataloader.caltech101_loader import Caltech101Loader
from cifarclassify.modelloader.imagenet.alexnet import AlexNet
from cifarclassify.modelloader.imagenet.googlenet import GoogLeNet
from cifarclassify.modelloader.imagenet.peleenet import PeleeNet
from cifarclassify.modelloader.imagenet.resnet import resnet18, resnet50


def train(args):
    if args.vis:
        vis = visdom.Visdom()
        vis.close()

    local_path = os.path.expanduser(args.dataset_path)
    trainset = None
    valset = None
    if args.dataset == 'Bearing':
        trainset = BearingLoader(local_path, is_transform=True, is_augment=False, split='train')
        valset = BearingLoader(local_path, is_transform=True, is_augment=False, split='val')
    elif args.dataset == 'Caltech101':
        trainset = Caltech101Loader(local_path, is_transform=True, is_augment=False, split='train')
        valset = Caltech101Loader(local_path, is_transform=True, is_augment=False, split='val')
    else:
        print('{} dataset does not implement'.format(args.dataset))
        exit(0)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)

    start_epoch = 0

    if args.structure == 'AlexNet':
        model = AlexNet(n_classes=trainset.n_classes, pretrained=args.init_vgg16)
    elif args.structure == 'resnet18':
        model = resnet18(n_classes=trainset.n_classes, pretrained=args.init_vgg16)
    elif args.structure == 'resnet50':
        model = resnet50(n_classes=trainset.n_classes, pretrained=args.init_vgg16)
    elif args.structure == 'GoogLeNet':
        model = GoogLeNet(n_classes=trainset.n_classes)
    elif args.structure == 'PeleeNet':
        model = PeleeNet(n_classes=trainset.n_classes)
    else:
        print('not valid model name')
        exit(0)

    if args.resume_model_state_dict != '':
        start_epoch_id1 = args.resume_model_state_dict.rfind('_')
        start_epoch_id2 = args.resume_model_state_dict.rfind('.')
        start_epoch = int(args.resume_model_state_dict[start_epoch_id1 + 1:start_epoch_id2])
        model.load_state_dict(torch.load(args.resume_model_state_dict))

    if args.cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.1)

    for epoch in range(start_epoch, 20000, 1):
        print('epoch:', epoch)
        scheduler.step()
        model.train()
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
                vis.line(X=np.ones(1) * epoch, Y=lr, win=win, name=win, opts=dict(title=win, xlabel='iteration', ylabel='lr'))

        for i, (imgs, labels) in enumerate(trainloader):
            # data_count = i
            # print(i)
            imgs, labels = Variable(imgs), Variable(labels)

            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()

            # 训练优化参数
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)
            # print('loss:', loss)
            loss_numpy = loss.cpu().data.numpy()
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
                    vis.line(X=np.ones(1) * (epoch*trainset.__len__()/(args.batch_size*1.0) + i), Y=loss_numpy, win=win, name=win, opts=dict(title=win, xlabel='iteration', ylabel='loss'))
            loss.backward()

            optimizer.step()
            # if i == 10:
            #     break
            # break

        # 输出一个周期后的loss
        # loss_avg_epoch = loss_epoch / (data_count * args.batch_size * 1.0)
        # print('loss_avg_epoch:', loss_avg_epoch)

        # val result on val dataset and pick best to save
        if args.val_interval > 0  and epoch % args.val_interval == 0:
            # print('----starting val----')
            model.eval()
            val_correct = 0
            val_data_count = valset.__len__() * 1.0
            for val_i, (val_imgs, val_labels) in enumerate(valloader):
                val_imgs, val_labels = Variable(val_imgs), Variable(val_labels)

                if args.cuda:
                    val_imgs = val_imgs.cuda()
                    val_labels = val_labels.cuda()

                # print('val_imgs.shape:', val_imgs.shape)
                # print('val_labels.shape:', val_labels.shape)
                val_outputs = model(val_imgs)
                # print('val_outputs.shape:', val_outputs.shape)
                val_pred = val_outputs.cpu().data.max(1)[1].numpy()
                # print('val_pred:', val_pred)
                # print('val_pred.shape:', val_pred.shape)
                val_labels_np = val_labels.cpu().data.numpy()
                # print('val_labels_np:', val_labels_np)
                val_correct += sum(val_labels_np==val_pred)
                # print('val_correct:', val_correct)
                # break
            val_acc = val_correct * 1.0 / val_data_count
            # print('val_acc:', val_acc)
            if args.vis:
                win = 'acc epoch'
                val_acc_expand = np.expand_dims(val_acc, axis=0)
                win_res = vis.line(X=np.ones(1) * epoch * args.val_interval, Y=val_acc_expand, win=win, update='append', name=win)
                if win_res != win:
                    vis.line(X=np.ones(1) * epoch * args.val_interval, Y=val_acc_expand, win=win, name=win, opts=dict(title=win, xlabel='epoch', ylabel='acc'))
            # print('----ending   val----')
        # 存储模型
        # if args.save_model and epoch%args.save_epoch==0 and epoch != 0:
        #     torch.save(model.state_dict(), '{}_cifar10_{}.pt'.format(args.structure, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameter setting')
    parser.add_argument('--structure', type=str, default='resnet18', help='use the net structure to segment [ AlexNet ]')
    parser.add_argument('--resume_model', type=str, default='', help='resume model path [ AlexNet_cifar10_0.pkl ]')
    parser.add_argument('--resume_model_state_dict', type=str, default='', help='resume model state dict path [ AlexNet_cifar10_0.pt ]')
    parser.add_argument('--save_model', type=bool, default=False, help='save model [ False ]')
    parser.add_argument('--save_epoch', type=int, default=1, help='save model after epoch [ 1 ]')
    parser.add_argument('--init_vgg16', type=bool, default=False, help='init model using vgg16 weights [ False ]')
    parser.add_argument('--dataset', type=str, default='Caltech101', help='train dataset path [ Caltech101 Bearing ]')
    parser.add_argument('--dataset_path', type=str, default='~/Data/101_ObjectCategories', help='train dataset path [ ~/Data/101_ObjectCategories ~/Data/Bearing ]')
    parser.add_argument('--data_augment', type=bool, default=False, help='enlarge the training data [ False ]')
    parser.add_argument('--batch_size', type=int, default=128, help='train dataset batch size [ 128 ]')
    parser.add_argument('--val_interval', type=int, default=1, help='val dataset interval unit epoch [ 3 ]')
    parser.add_argument('--lr', type=float, default=1e-1, help='train learning rate [ 0.01 ]')
    parser.add_argument('--vis', type=bool, default=False, help='visualize the training results [ False ]')
    parser.add_argument('--cuda', type=bool, default=False, help='use cuda [ False ]')
    args = parser.parse_args()
    print(args)
    train(args)
