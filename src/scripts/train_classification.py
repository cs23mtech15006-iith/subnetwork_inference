import argparse
import yaml
import os
import shutil
import time

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

from src.datasets.image_loaders import get_image_loader
from src.utils import check_dataparallel_save, instantiate_model, strip_dataparallel_save, mkdir, save_object, load_object

###################################
#  Default Pytorch Train loop with some hardcoded values for simplicity
#  Multinode features have been removed
#  Code has been changed to take any model as input
#######################################

parser = argparse.ArgumentParser(description='PyTorch Image Training')

parser.add_argument('--config', default=None, type=str,
                    help='path to YAML config file to use')
parser.add_argument('--seed', default=0, type=int,
                    help='random seed')
parser.add_argument('--dataset', type=str, default='MNIST',
                    choices=["CIFAR10", "CIFAR100", "SVHN", "MNIST", "Fashion", "EMNIST"],
                    help='dataset to train (default: MNIST)')
parser.add_argument('--data_dir', type=str, default='../data/',
                    help='directory where datasets are saved (default: ../data/)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=None, type=int,
                    help='number of total epochs to run (if None, use dataset default)')
parser.add_argument('--mcsamples', default=1, type=int,
                    help='number of MC dropout samples to use at test time (default: 1)')
parser.add_argument('-wd', '--weight_decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--save_dir', default='./results/', type=str,
                    help='path where to save checkpoints (default: ./results/)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use (default: None)')
parser.add_argument('--batch_size', default=256, type=int,
                    help='batch size to use (default: 256)')
parser.add_argument('--model', type=str, default='resnet50',
                    choices=["lenet", "resnet18", "resnet32", "resnet50", "resnet101"],
                    help='model to train (default: resnet50)')
parser.add_argument('--p_drop', type=float, default=0,
                    help='dropout probability at conv layers (default: 0)')

best_acc1 = 0
lr = 0.1
momentum = 0.9
print_freq = 40


def main(args):
    dataset = args.dataset
    workers = args.workers
    epochs = args.epochs
    weight_decay = args.weight_decay
    resume = args.resume
    MC_samples = 1 if args.p_drop == 0 else args.mcsamples 
    save_dir = args.save_dir
    gpu = args.gpu
    data_dir = args.data_dir
    batch_size = args.batch_size
    model = args.model
    p_drop = args.p_drop

    epoch_dict = {
        'Imagenet': 90,
        'SmallImagenet': 90,
        'CIFAR10': 300,
        'CIFAR100': 300,
        'SVHN': 90,
        'Fashion': 90,
        'MNIST': 90,
        'EMNIST': 90
    }

    milestone_dict = {
        'Imagenet': [30, 60],  # This is pytorch default
        'SmallImagenet': [30, 60],
        'CIFAR10': [150, 225],
        'CIFAR100': [150, 225],
        'SVHN': [50, 70],
        'Fashion': [40, 70],
        'MNIST': [40, 70],
        'EMNIST': [40, 70],
    }

    if epochs is None:
        epochs = epoch_dict[dataset]
    milestones = milestone_dict[dataset]

    # instantiate and train model
    model = instantiate_model(model, dataset, p_drop)
    train_loop(model, dname=dataset, data_dir=data_dir, epochs=epochs, workers=workers, gpu=gpu, resume=resume,
               weight_decay=weight_decay, save_dir=save_dir, milestones=milestones,
               MC_samples=MC_samples, batch_size=batch_size)


def train_loop(model, dname, data_dir, epochs=90, workers=4, gpu=None, resume='', weight_decay=1e-4,
               save_dir='./', milestones=None, MC_samples=1, batch_size=256):
    if save_dir is not None:
        mkdir(save_dir)
    global best_acc1

    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))

    if gpu is not None:  # Check for single GPU
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    elif (not resume) or (not os.path.isfile(resume)):
        # # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(reduction='mean').cuda(gpu)

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    if milestones is None:  # if milestones are not specified, set to impossible value so LR is never decayed.
        milestones = [epochs + 1]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    tr_acc1_vec = []
    tr_acc5_vec = []
    tr_loss_vec = []
    acc1_vec = []
    acc5_vec = []
    loss_vec = []

    start_epoch = 0
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            if gpu is None:
                checkpoint = torch.load(resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(gpu)
                checkpoint = torch.load(resume, map_location=loc)
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(gpu)

            model_state_dict = checkpoint['model_state'] if 'model_state' in checkpoint.keys() else checkpoint['state_dict']
            if check_dataparallel_save(model_state_dict):
                    model_state_dict = strip_dataparallel_save(model_state_dict)
            model.load_state_dict(model_state_dict)
            if gpu is None:
                 model = torch.nn.DataParallel(model).cuda()

            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

        candidate_progress_file = resume.split('/')
        candidate_progress_file = '/'.join(candidate_progress_file[:-1]) + '/stats_array.pkl'

        if os.path.isfile(candidate_progress_file):
            print("=> found progress file at '{}'".format(candidate_progress_file))
            try:
                tr_acc1_vec, tr_acc5_vec, tr_loss_vec, acc1_vec, acc5_vec, loss_vec = \
                    load_object(candidate_progress_file)
                print("=> Loaded progress file at '{}'".format(candidate_progress_file))
            except Exception:
                print("=> Unable to load progress file at '{}'".format(candidate_progress_file))
        else:
            print("=> NOT found progress file at '{}'".format(candidate_progress_file))

    cudnn.benchmark = True

    _, train_loader, val_loader, _, _, _ = \
        get_image_loader(dname, batch_size, cuda=True, workers=workers, distributed=False, data_dir=data_dir)

    for epoch in range(start_epoch, epochs):

        # train for one epoch and update lr scheduler setting
        tr_acc1, tr_acc5, tr_loss = train(train_loader, model, criterion, optimizer, epoch, gpu)
        print('used lr: %f' % optimizer.param_groups[0]["lr"])
        scheduler.step()

        tr_acc1_vec.append(tr_acc1)
        tr_acc5_vec.append(tr_acc5)
        tr_loss_vec.append(tr_loss)

        # evaluate on validation set
        acc1, acc5, loss = validate(val_loader, model, criterion, gpu, MC_samples=MC_samples)

        acc1_vec.append(acc1)
        acc5_vec.append(acc5)
        loss_vec.append(loss)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if save_dir is not None:
            if isinstance(model, nn.DataParallel):
                model_save = model.module
            else:
                model_save = model
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state': model_save.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, save_dir=save_dir)

            all_results = [tr_acc1_vec, tr_acc5_vec, tr_loss_vec, acc1_vec, acc5_vec, loss_vec]
            save_object(all_results, os.path.join(save_dir, 'stats_array.pkl'))


def train(train_loader, model, criterion, optimizer, epoch, gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if gpu is not None:
            images = images.cuda(gpu, non_blocking=True)
        target = target.cuda(gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)
    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, gpu, MC_samples=1):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

            # compute output with option for MC_dropout predictions
            if MC_samples == 1:
                output = model(images)
                loss = criterion(output, target)
            else:
                pred_samples = []
                for sample in range(MC_samples):
                    output = model(images)
                    pred_samples.append(F.log_softmax(output, dim=1))
                pred_samples = torch.stack(pred_samples, dim=0)
                mean_log_preds = torch.logsumexp(pred_samples, dim=0) - np.log(pred_samples.shape[0])
                output = mean_log_preds
                loss = F.nll_loss(output, target, reduction='mean')

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    print('Saving to %s' % os.path.join(save_dir, 'checkpoint.pth.tar'))
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        print("Saving best")
        shutil.copyfile(os.path.join(save_dir, filename), os.path.join(save_dir, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    args_dict = vars(args)
    
    # load YAML config file
    if args.config is not None:
        with open(args.config) as f:
            config = yaml.full_load(f)
            args_dict.update(config)

    # print arguments
    for key, val in args_dict.items():
        print(f'{key}: {val}')
    print()

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    main(args)
