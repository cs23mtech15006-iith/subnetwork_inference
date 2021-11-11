import time
import copy
from pathlib import Path
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import load_object, save_object, mkdir, check_dataparallel_save, strip_dataparallel_save, model_to_device
import numpy as np
import shutil
import os

from src.models.MLPs import res_MLP
from src.datasets.UCI_gap_loader import train_val_test_UCI
from src.utils import homo_Gauss_mloglike

import argparse

parser = argparse.ArgumentParser(description='UCI dataset running script')


parser.add_argument('--dataset', help='Toggles which dataset to optimize for.', choices=['boston_gap','concrete_gap', 'energy_gap', 'power_gap', 'wine_gap', 'yacht_gap',
                    'kin8nm_gap','naval_gap', 'protein_gap', 'boston', 'concrete','energy', 'power', 'wine', 'yacht','kin8nm', 'naval', 'protein'], default=None)

parser.add_argument('--N_split', type=int, help='split to run.', default=0)

parser.add_argument('--early_stop_patience', type=int, help='early stopping patience in epochs. 0 to disable. -1 to a tenth of training time.', default=-1)

parser.add_argument('--data_dir', type=str, help='Where to save dataset. Default ./data/',
                    default='./data/')

parser.add_argument('--save_dir', type=str, help='Where to save results. Default ./saves/',
                    default='./UCI_results/')


parser.add_argument('--BN', dest='BN', action='store_true',
                    help='Use Batchnorm, default: Not using it',
                    default=False)


# parser.add_argument('--activation', type=str,
#                     help='Type of activation function to use. Options: ReLU, Softplus, Tanh. Default: ReLU',
#                     choices=['ReLU', 'Tanh', 'Softplus'],
#                     default='ReLU')

parser.add_argument('--num_workers', type=int, help='Number of parallel workers for dataloading. Default: 1', default=1)
parser.add_argument('--N_layers', type=int, help='Number of hidden layers to use.', default=2)
parser.add_argument('--width', type=int, help='Number of hidden units to use.', default=50)
parser.add_argument('--batch_size', type=int, help='Default: 512.', default=512)
parser.add_argument('--N_epochs', type=int, help='Default: 1000.', default=1000)

parser.add_argument('--save_freq', type=int, help='Number of epochs between saves. Default: 20.', default=20)


parser.add_argument('--device', type=str, help='Which GPU to run on. default=cpu', default='cpu')



parser.add_argument('--lr', type=float, help='learning rate, default=1e-2', default=1e-2)
parser.add_argument('--wd', type=float, help='weight_decay, default=1e-4', default=1e-4)

parser.add_argument('--seed', type=float, help='seed, default=0', default=0)



def main(args):
    prop_val = 0.15
    momentum = 0.9

    if args.device != 'cpu':
         args.device = int( args.device)

    trainset, valset, testset, N_train, input_dim, output_dim, y_means, y_stds = \
        train_val_test_UCI(args.dataset, base_dir=args.data_dir, prop_val=prop_val, n_split=args.N_split)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=(args.device != 'cpu'),
                                                num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, pin_memory=(args.device != 'cpu'),
                                            num_workers=args.num_workers)

    model = res_MLP(input_dim=input_dim, output_dim=output_dim, width=args.width, n_layers=args.N_layers, BN=args.BN, act=nn.ReLU)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=momentum, weight_decay=args.wd)

    nll_func = homo_Gauss_mloglike()

    optimizer.add_param_group({"params": nll_func.parameters()})

    if args.early_stop_patience > 0:
        early_stop_patience = args.early_stop_patience

    elif args.early_stop_patience == -1:
        early_stop_patience = int(args.N_epochs / 10)

    else:
        early_stop_patience = None

    milestones = [int(args.N_epochs * 2 / 3)]

    train_loop(model, trainloader, valloader, optimizer, args.N_epochs, nll_func, device=args.device, resume='',
               savedir=args.save_dir, milestones=milestones, save_freq=args.save_freq, patience=early_stop_patience)

def train_loop(model, train_loader, val_loader, optimizer, epochs, nll_func, device='cpu', resume='',
               savedir='./', milestones=None, save_freq=10, patience=None):

    # Device can be 'cpu', 'cuda:x' or None for using all available GPUs

    if savedir is not None:
        mkdir(savedir)

    if device != 'cpu':
        if device is not None: # check for single GPU
            gpu = int(device.replace('cuda:', ''))
            print("Use GPU: {} for training".format(gpu))
            torch.cuda.set_device(gpu)
            model = model.cuda(gpu)
        elif not resume:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()

        # define loss function  and optimizer
        nll_func = nll_func.cuda()

    # # add optimizer parameters in case that they are needed
    # optimizer.add_param_group({"params": nll_func.parameters()})

    # if milestones are not specified, set to impossible value so LR is never decayed.
    if milestones is None:
        milestones = [epochs + 1]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    tr_err_vec = []
    tr_loss_vec = []
    err_vec = []
    loss_vec = []
    best_loss = np.inf
    is_best = False
    best_epoch = 0

    start_epoch = 0
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            if device == 'cpu':
                checkpoint = torch.load(resume, lambda storage, loc: storage)
            elif device is None:
                checkpoint = torch.load(resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(gpu)
                checkpoint = torch.load(resume, map_location=loc)
            start_epoch = checkpoint['epoch']

            model_state_dict = checkpoint['model_state'] if 'model_state' in checkpoint.keys() else checkpoint['state_dict']
            if check_dataparallel_save(model_state_dict):
                    model_state_dict = strip_dataparallel_save(model_state_dict)
            model.load_state_dict(model_state_dict)
            if device is None:
                 model = torch.nn.DataParallel(model).cuda()

            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            nll_func.load_state_dict(checkpoint['nll_func'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

        candidate_progress_file = resume.split('/')
        candidate_progress_file = '/'.join(candidate_progress_file[:-1]) + '/stats_array.pkl'

        if os.path.isfile(candidate_progress_file):
            print("=> found progress file at '{}'".format(candidate_progress_file))
            try:
                tr_err_vec, tr_loss_vec, err_vec, loss_vec, best_loss = \
                    load_object(candidate_progress_file)
                print("=> Loaded progress file at '{}'".format(candidate_progress_file))
            except Exception:
                print("=> Unable to load progress file at '{}'".format(candidate_progress_file))
        else:
            print("=> NOT found progress file at '{}'".format(candidate_progress_file))

    cudnn.benchmark = True

    for epoch in range(start_epoch, epochs):

        # train for one epoch and update lr scheduler setting
        tr_loss, tr_err = train(train_loader, model, nll_func, optimizer, device)
        # if milestones:
            # print('used lr: %f' % optimizer.param_groups[0]["lr"])
        scheduler.step()

        tr_err_vec.append(tr_err)
        tr_loss_vec.append(tr_loss)

        print('Epoch %d, train err %.5f, train loss %.5f' % (epoch, tr_err, tr_loss))

        # evaluate on validation set
        if val_loader is not None:
            loss, err = validate(val_loader, model, nll_func, device)

            err_vec.append(err)
            loss_vec.append(loss)

            # remember best acc@1 and save checkpoint
            is_best = loss < best_loss
            if is_best:
                best_epoch = epoch
            elif patience is not None and epoch > best_epoch + patience:
                print(f'No improvement for {patience} epochs, early stopping.')
                break

            best_loss = min(loss, best_loss)

            print('-- val err %.5f, val loss %.5f, Best:%s' % (err, loss, str(is_best)))

        if savedir is not None and (is_best or epoch % save_freq == 0):
            if isinstance(model, nn.DataParallel):
                model_save = model.module
            else:
                model_save = model
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state': model_save.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'nll_func': nll_func.state_dict(),
            }, is_best, savedir=savedir)

            all_results = [tr_err_vec, tr_loss_vec, err_vec, loss_vec, best_loss]
            save_object(all_results, os.path.join(savedir, 'stats_array.pkl'))


        

def train(train_loader, model, criterion, optimizer, device):

    # switch to train mode
    model.train()

    end = time.time()

    total_loss = 0
    total_mse = 0
    for i, (inputs, targets) in enumerate(train_loader):

        if device != 'cpu':
            if device is not None:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device , non_blocking=True)
            else:
                targets = targets.to('cuda:0', non_blocking=True)

        # compute output
        output = model(inputs)
        loss = criterion(output, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        batch_size = inputs.shape[0]
        total_loss += loss.item() * batch_size
        total_mse += F.mse_loss(output, targets, size_average='sum').item()

        end = time.time()

    return total_loss / len(train_loader.dataset), total_mse / len(train_loader.dataset)


def validate(val_loader, model, criterion, device):

    # switch to evaluate mode
    model.eval()

    total_loss = 0
    total_mse = 0
    with torch.no_grad():
        end = time.time()
        for i, (inputs, targets) in enumerate(val_loader):
            
            if device != 'cpu':
                if device is not None:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                else:
                    targets = targets.to('cuda:0', non_blocking=True)

            # compute output
            output = model(inputs)
            loss = criterion(output, targets)

            # measure accuracy and record loss
            batch_size = inputs.shape[0]
            total_loss += loss.item() * batch_size
            
            total_mse += F.mse_loss(output, targets, size_average='sum').item()

            # measure elapsed time
            end = time.time()

    return total_loss / len(val_loader.dataset), total_mse / len(val_loader.dataset)


def save_checkpoint(state, is_best, savedir, filename='checkpoint.pth.tar'):
    print('Saving to %s' % os.path.join(savedir, 'checkpoint.pth.tar'))
    torch.save(state, os.path.join(savedir, filename))
    if is_best:
        print("Saving best")
        shutil.copyfile(os.path.join(savedir, filename), os.path.join(savedir, 'model_best.pth.tar'))


if __name__ == '__main__':
    args = parser.parse_args()

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != 'cpu':
        torch.backends.cudnn.benchmark = True

    main(args)
