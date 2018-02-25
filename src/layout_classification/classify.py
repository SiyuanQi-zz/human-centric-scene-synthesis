import os
import shutil
import time
import argparse

import numpy as np
import scipy.misc
import scipy.io
import torch
import torch.nn
import torchvision
import cv2
import matplotlib.pyplot as plt


model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='../../tmp/dataset/', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet152',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg16)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='../../tmp/checkpoints/', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=True, action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_prec1 = 0


class SegDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.image_list = os.listdir(root_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image = scipy.misc.imread(os.path.join(self.root_dir, image_name), mode='RGB')
        image = cv2.resize(image, (244, 244), interpolation=cv2.INTER_LINEAR)
        target = torch.LongTensor([int(os.path.splitext(image_name)[0].split('_')[-1])])
        # print target
        # plt.imshow(image)
        # plt.show()
        return self.transform(image), target


class Vgg16(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(Vgg16, self).__init__()
        self.features = torchvision.models.vgg16(pretrained=False).features
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            # torch.nn.ReLU(True),
            # torch.nn.Dropout(),
            # torch.nn.Linear(4096, num_classes),
            # torch.nn.Sigmoid()
        )
        self.last_layer = torch.nn.Sequential(
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        output = self.last_layer(x)
        return x, output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Resnet152(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(Resnet152, self).__init__()
        # self.learn_modules = torch.nn.Sequential()
        # pretrained_resnet = torchvision.models.resnet152(pretrained=True)
        # for i, m in enumerate(pretrained_resnet.modules()):
        #     if isinstance(m, torch.nn.Linear):
        #         break
        #     self.learn_modules.add_module(str(i), m)
        self.learn_modules = torchvision.models.resnet152(pretrained=False)
        self.fc = torch.nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.learn_modules(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return x, output


def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    # model = Vgg16(num_classes=2)
    model = Resnet152(num_classes=2)
    # model = torchvision.models.resnet152(pretrained=True)

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(os.path.join(args.resume, 'model_best.pth')):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(os.path.join(args.resume, 'model_best.pth'))
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(os.path.join(args.resume, 'model_best.pth')))

    torch.backends.cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    testdir = os.path.join(args.data, 'test')
    # normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # normalize,
        ])

    train_dataset = SegDataset(traindir, transform)
    val_dataset = SegDataset(testdir, transform)
    test_dataset = SegDataset(valdir, transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(test_loader, model, criterion, test=True)
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Best precision: {:.03f}'.format(best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
    
    test_prec = validate(test_loader, model, criterion, test=True)
    print('Testing precision: {:.04f}'.format(test_prec))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        feature, output = model(input_var)
        loss = criterion(output, target_var.squeeze(1))

        #################################################
        ###################  feature  ###################
        #################################################
        # feature = feature.data.cpu().numpy()
        # scipy.io.savemat(os.path.join(args.resume, '../features/train/{:05d}_{}.mat'.format(i, target.cpu().numpy()[0, 0])), {'feature': feature})

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        prec1 = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))
        # top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        feature, output = model(input_var)
        loss = criterion(output, target_var.squeeze(1))

        #################################################
        ###################  feature  ###################
        #################################################
        # feature = feature.data.cpu().numpy()
        # if test:
        #     scipy.io.savemat(os.path.join(args.resume, '../features/test/{:05d}_{}.mat'.format(i, target.cpu().numpy()[0, 0])), {'feature': feature})
        # else:
        #     scipy.io.savemat(os.path.join(args.resume, '../features/val/{:05d}_{}.mat'.format(i, target.cpu().numpy()[0, 0])), {'feature': feature})

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        prec1 = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))
        # top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.4f} ({top1.avg:.4f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec@1 {top1.avg:.4f} Prec@5 {top5.avg:.4f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, os.path.join(args.resume, filename))
    if is_best:
        shutil.copyfile(os.path.join(args.resume, filename), os.path.join(args.resume, 'model_best.pth'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    output = torch.nn.Softmax()(output)
    # print output
    # print target
    correct_num = 0
    output_np = output.data.cpu().numpy()
    target_np = target.cpu().numpy()
    for batch_i in range(target.size()[0]):
        if np.argmax(output_np[batch_i, :]) == np.argmax(target_np[batch_i, :]):
            correct_num += 1
    # print correct_num
    return float(correct_num)/target.size()[0]


def accuracy_old(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()