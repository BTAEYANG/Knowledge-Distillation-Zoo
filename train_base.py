from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst

from dataUtils.getData import getDataLoader
from utils import AverageMeter, accuracy, transform_time, define_tsnet
from utils import load_pretrained_model, save_checkpoint
from utils import create_exp_dir, count_parameters_in_MB

parser = argparse.ArgumentParser(description='Train base net')

# various path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='/home/lab265/lab265/datasets', help='path name of image dataset')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=100, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--split_factor', type=float, default=0.2, help='split factor for dataset produce train val test')
parser.add_argument('--gpu_dataParallel', type=bool, default=False, help='use gpu data parallel')

# net and dataset choose
parser.add_argument('--data_name', type=str, required=True, help='name of dataset')  # CIFAR10 / CIFAR100
parser.add_argument('--net_name', type=str, required=True, help='name of base net')

args, unparsed = parser.parse_known_args()

args.save_root = os.path.join(args.save_root, args.note)
create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    logging.info('----------- Network Initialization --------------')
    net = define_tsnet(name=args.net_name, num_class=args.num_class, cuda=args.cuda)
    logging.info('----------- Param size = %fMB', count_parameters_in_MB(net))

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True
    if args.gpu_dataParallel:
        net = torch.nn.DataParallel(net)

    # save initial parameters
    logging.info('Saving initial parameters......')
    save_path = os.path.join(args.save_root, 'initial_r{}.pth.tar'.format(args.net_name[6:]))
    torch.save({
        'epoch': 0,
        'net': net.state_dict(),
        'prec@1': 0.0,
        'prec@5': 0.0,
    }, save_path)

    # initialize optimizer
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # initialize scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # define loss functions
    if args.cuda:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # load data_loader
    train_loader, validation_loader, test_loader = getDataLoader(root_path=args.img_root,
                                                                 split_factor=args.split_factor, seed=args.seed,
                                                                 data_set=args.data_name)

    best_top1 = 0
    best_top5 = 0
    for epoch in range(1, args.epochs + 1):
        # adjust_lr(optimizer, epoch)
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'current_lr：{current_lr}')

        # train one epoch
        epoch_start_time = time.time()
        train(train_loader, net, optimizer, criterion, epoch)

        # evaluate on testing set
        logging.info('Validation the models......')
        val_top1, val_top5 = val(validation_loader, net, criterion)

        epoch_duration = time.time() - epoch_start_time
        logging.info('Epoch time: {}s'.format(int(epoch_duration)))

        # scheduler step
        scheduler.step()

        # save model
        is_best = False
        if val_top1 > best_top1:
            best_top1 = val_top1
            best_top5 = val_top5
            is_best = True
        logging.info('Saving models......')
        save_checkpoint({
            'epoch': epoch,
            'net': net.state_dict(),
            'prec@1': val_top1,
            'prec@5': val_top5,
        }, is_best, args.save_root)


def train(train_loader, net, optimizer, criterion, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.train()

    end = time.time()
    for i, (img, target) in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)

        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        _, _, _, _, _, out = net(img)
        loss = criterion(out, target)

        pre_1, pre_5 = accuracy(out, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(pre_1.item(), img.size(0))
        top5.update(pre_5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
                       'Time:{batch_time.val:.4f} '
                       'Data:{data_time.val:.4f}  '
                       'loss:{losses.val:.4f}(avg:{losses.avg:.4f})  '
                       'prec@1:{top1.val:.2f}(avg:{top1.avg:.2f}%)  '
                       'prec@5:{top5.val:.2f}(avg:{top5.avg:.2f}%)'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                losses=losses, top1=top1, top5=top5))
            logging.info(log_str)


def val(validation_loader, net, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()

    for i, (img, target) in enumerate(validation_loader, start=1):
        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        with torch.no_grad():
            _, _, _, _, _, out = net(img)
            loss = criterion(out, target)

        pre_1, pre_5 = accuracy(out, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(pre_1.item(), img.size(0))
        top5.update(pre_5.item(), img.size(0))

    f_l = [losses.avg, top1.avg, top5.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}%, Prec@5: {:.2f}%'.format(*f_l))

    return top1.avg, top5.avg


def test(test_loader, net, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    net.eval()

    for i, (img, target) in enumerate(test_loader, start=1):
        if args.cuda:
            img = img.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        with torch.no_grad():
            _, _, _, _, _, out = net(img)
            loss = criterion(out, target)

        pre_1, pre_5 = accuracy(out, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(pre_1.item(), img.size(0))
        top5.update(pre_5.item(), img.size(0))

    f_l = [losses.avg, top1.avg, top5.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}%, Prec@5: {:.2f}%'.format(*f_l))

    return top1.avg, top5.avg


# def adjust_lr(optimizer, epoch):
#     scale = 0.1
#     lr_list = [args.lr] * 100
#     lr_list += [args.lr * scale] * 50
#     lr_list += [args.lr * scale * scale] * 50
#
#     lr = lr_list[epoch - 1]
#     logging.info('Epoch: {}  lr: {:.3f}'.format(epoch, lr))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


if __name__ == '__main__':
    main()
