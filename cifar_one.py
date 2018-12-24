'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
import torch.utils.data
from loss import KLLoss
from torch.utils.data import ConcatDataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig,ramps
import pdb

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--consistency_rampup', '--consistency_rampup', default=80, type=float,
                    metavar='consistency_rampup', help='consistency_rampup ratio')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')

parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume',
                    default='',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
parser.add_argument('--layerensemble', type=bool, default=False, help='Using layer ensembel')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
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
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    elif args.dataset =='cifar100':
        dataloader = datasets.CIFAR100
        num_classes = 100
    elif args.dataset == 'svhn':
        dataloader = datasets.SVHN
        num_classes = 10

    if args.dataset.startswith('cifar'):

        trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    elif args.dataset == 'svhn':
        train_dataset = dataloader(root='./data', split='train', download=True, transform=transforms.Compose([
                                   # transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))]))
        train_extra_dataset = dataloader(root='./data', split='extra', download=True, transform=transforms.Compose([
                                   # transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))]))
        trainset = ConcatDataset([train_dataset, train_extra_dataset])
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        testset = dataloader(root='./data', split='test', download=True, transform=transforms.Compose([
                                   # transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
                               ]))
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    else:
        print("not support dataset")

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.endswith('resnext'):
        model = models.__dict__[args.arch](
            cardinality=args.cardinality,
            num_classes=num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            dropRate=args.drop,
        )
    elif args.arch.endswith('densenet'):

        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
            growthRate=args.growthRate,
            compressionRate=args.compressionRate,
            dropRate=args.drop,
        )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
            widen_factor=args.widen_factor,
            dropRate=args.drop,
        )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
            num_classes=num_classes,
            depth=args.depth,
        )

    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss()

    criterion_kl = KLLoss().cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cifar' + args.arch

    if args.resume or args.evaluate:
        # Load checkpoint.

        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['TAcc_1', 'VAcc_1','TAcc_2','VAcc_2','TAcc_3','VAcc_3','TAcc_e','VAcc_e'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc_1,train_acc_2,train_acc_3,train_acc_en = train(trainloader, model, criterion, criterion_kl, optimizer, epoch,
                                      use_cuda)
        test_loss, test_acc_1,test_acc_2,test_acc_3,test_acc_en = test(testloader, model, use_cuda)

        # append logger file
        logger.append([train_acc_1, test_acc_1,train_acc_2,test_acc_2,train_acc_3,test_acc_3,train_acc_en,test_acc_en])

        # save model
        is_best = test_acc_1 > best_acc
        best_acc = max(test_acc_1, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc_1,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, criterion_kl, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    losses = AverageMeter()
    losses_kl = AverageMeter()
    top1_c1 = AverageMeter()
    top5_c1 = AverageMeter()
    top1_c2 = AverageMeter()
    top5_c2 = AverageMeter()
    top1_c3 = AverageMeter()
    top5_c3 = AverageMeter()
    top1_t = AverageMeter()
    top5_t = AverageMeter()


    bar = Bar('Processing', max=len(trainloader))
    consistency_weight = get_current_consistency_weight(epoch)
    for batch_idx, (inputs, targets) in enumerate(trainloader):


        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs1, outputs2, outputs3, outputs4 = model(inputs)

        loss_cross = criterion(outputs1, targets) + criterion(outputs2, targets) + criterion(outputs3,targets) + criterion(
                 outputs4, targets)

        loss_kl = consistency_weight*(criterion_kl(outputs1, outputs4) +criterion_kl(outputs2,outputs4)+criterion_kl(outputs3,outputs4))
        prec1_t, prec5_t = accuracy(outputs4.data, targets.data, topk=(1, 5))
        prec1_c1, prec5_c1 = accuracy(outputs1.data, targets.data, topk=(1, 5))
        prec1_c2, prec5_c2 = accuracy(outputs2.data, targets.data, topk=(1, 5))
        prec1_c3, prec5_c3 = accuracy(outputs3.data, targets.data, topk=(1, 5))
        top1_c1.update(prec1_c1[0], inputs.size(0))
        top5_c1.update(prec5_c1[0], inputs.size(0))
        loss = loss_cross+loss_kl
        losses_kl.update(loss_kl.data[0], inputs.size(0))
        losses.update(loss.data[0], inputs.size(0))
        top1_c2.update(prec1_c2[0], inputs.size(0))
        top5_c2.update(prec5_c2[0], inputs.size(0))
        top1_c3.update(prec1_c3[0], inputs.size(0))
        top5_c3.update(prec5_c3[0], inputs.size(0))
        top1_t.update(prec1_t[0], inputs.size(0))
        top5_t.update(prec5_t[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar.suffix = '({batch}/{size})   || Loss: {loss:.4f} |LossKL: {losses_kl:.4f} | top1_C1: {top1_C1: .4f} | top1_C2: {top1_C2: .4f}|top1_C3: {top1_C3: .4f}| top1_t: {top1_t: .4f} '.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            loss=losses.avg,
            losses_kl=losses_kl.avg,
            top1_C1=top1_c1.avg,
            top1_C2=top1_c2.avg,
            top1_C3=top1_c3.avg,
            top1_t=top1_t.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg,top1_c1.avg,top1_c2.avg,top1_c3.avg,top1_t.avg)


def test(testloader, model, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_c1 = AverageMeter()
    top1_c2 = AverageMeter()
    top1_c3 = AverageMeter()
    top1_avg= AverageMeter()
    top1_t = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):

        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        outputs1, outputs2, outputs3, outputs4 = model(inputs)

        # measure accuracy and record loss
        prec1_c1, _ = accuracy(outputs1.data, targets.data, topk=(1, 5))
        prec1_c2, _ = accuracy(outputs2.data, targets.data, topk=(1, 5))
        prec1_c3, _ = accuracy(outputs3.data, targets.data, topk=(1, 5))
        prec1_en, _ = accuracy(outputs4.data, targets.data, topk=(1, 5))
        top1_c1.update(prec1_c1[0], inputs.size(0))
        top1_c2.update(prec1_c2[0], inputs.size(0))
        top1_c3.update(prec1_c3[0], inputs.size(0))
        top1_avg.update((prec1_c1[0]+prec1_c2[0]+prec1_c3[0])/3, inputs.size(0))
        top1_t.update(prec1_en[0], inputs.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size})| Loss: {loss: .4f} | top1_C1: {top1_C1: .4f} |top1_C2: {top1_C2: .4f}|top1_C3: {top1_C3:.4f} |top1_t: {top1_t: .4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            loss=losses.avg,
            top1_C1=top1_c1.avg,
            top1_C2=top1_c2.avg,
            top1_C3=top1_c3.avg,
            top1_t=top1_t.avg,
        )
        bar.next()

    bar.finish()
    return (losses.avg, top1_c1.avg,top1_c2.avg,top1_c3.avg,top1_t.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
