import torch
import torch.nn as nn
import os
import pickle
from torch.nn import DataParallel
import torch.utils.data as data
from torch.distributions import Normal
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from PIL import Image
import csv
from torchvision import datasets

from src.models.img_resnets import resnet18, resnet34, resnet50, resnet101


def model_to_device(model, device, default='cuda:0'):
    if device is not None:
        model = model.to(device=device)
    else:
        model = model.to(device=default)
    return model


def get_last_layer(model):
    layer = list(model.children())[-1]
    
    while not isinstance(layer, nn.Linear):
        layer = layer[-1]
    
    return layer
    

def flatten_nn(model, batchnorm_layers):
    weights = []
    for name, param in model.named_parameters():
        name = name.replace('module.', '')
        if 'weight' in name and name not in batchnorm_layers:
            weights.append(param.detach().flatten())
    return torch.cat(weights, dim=0)


def unflatten_nn(model, batchnorm_layers, weights):

    w_pointer = 0
    weight_dict = {}
    for name, param in model.named_parameters():
        name = name.replace('module.', '')
        if 'weight' in name and name not in batchnorm_layers:
            len_w = param.data.numel()
            weight_dict[name] = weights[w_pointer:w_pointer +
                                        len_w].view(param.shape).float().to(weights.device, non_blocking=True)
            w_pointer += len_w

    return weight_dict


def get_n_params(model, batchnorm_layers, mask=None):
    D = 0
    for name, param in model.named_parameters():
        name = name.replace('module.', '')
        if 'weight' in name and name not in batchnorm_layers:
            if mask is not None:
                D += param[mask[name].nonzero(as_tuple=True)].numel()
            else:
                D += param.numel()
        
    return D
        

def list_batchnorm_layers(model):
    """ compute list of names of all BatchNorm layers in the model """

    batchnorm_layers = []
    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if type(module) is torch.nn.modules.batchnorm.BatchNorm2d or type(module) is torch.nn.modules.batchnorm.BatchNorm1d:
            batchnorm_layers.append(name + ".weight")

    return batchnorm_layers

def print_nonzeros(mask):
    """ Print table of zeros and non-zeros count """

    remain = total = 0
    for name in mask:
        nz_params = len(mask[name].nonzero(as_tuple=False))
        total_params = mask[name].numel()
        remain += nz_params
        total += total_params
        remaining = f"{nz_params:7} / {total_params:7} ({100 * nz_params / total_params:6.2f}%)"
        print(
            f"{name:35s} | remaining = {remaining} | pruned = {total_params - nz_params:7d} | shape = {mask[name].size()}")
    compr_rate = f"{total/remain:10.2f}x  ({100 * (total-remain) / total:6.2f}% pruned)"
    print("====================================================================================================")
    print(
        f"remaining: {remain}, pruned: {total - remain}, total: {total}, compression rate: {compr_rate}")


class Datafeed(data.Dataset):

    def __init__(self, x_train, y_train=None, transform=None):
        self.data = x_train
        self.targets = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.targets is not None:
            return img, self.targets[index]
        else:
            return img

    def __len__(self):
        return len(self.data)


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as f:
        try:
            return pickle.load(f)
        except Exception:
            return pickle.load(f, encoding="latin1")

def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path, mode=0o777)


def check_dataparallel_save(state_dict):
    key = list(state_dict.keys())[0]
    if key[:7] == 'module.':
        return True
    else:
        return False

def strip_dataparallel_save(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove module.
        new_state_dict[name] = v
    return new_state_dict
        

def load_model(model, savefile, cuda_enabled, gpu=None):
    if cuda_enabled:
        assert torch.cuda.is_available()
        if gpu is None:
            if not isinstance(model, nn.DataParallel):
                model = torch.nn.DataParallel(model)
            model = model.cuda()
            checkpoint = torch.load(savefile)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # torch.cuda.set_device(gpu)
            loc = 'cuda:{}'.format(gpu)
            model = model.to(device=loc)
            checkpoint = torch.load(savefile, map_location=loc)
            state_dict = checkpoint['state_dict']
            if check_dataparallel_save(state_dict):
                state_dict = strip_dataparallel_save(state_dict)

            model.load_state_dict(state_dict)
    else:
        checkpoint = torch.load(savefile, lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

    model = model.eval()
    return model


def rms(x, y):
    return F.mse_loss(x, y, reduction='mean').sqrt()


def get_rms(mu, y, y_means, y_stds):
    x_un = mu * y_stds + y_means
    y_un = y * y_stds + y_means
    assert x_un.shape[1] == 1
    assert y_un.shape[1] == 1
    return rms(x_un, y_un).item()


def get_gauss_loglike(mu, sigma, y, y_means, y_stds):
    mu_un = mu * y_stds + y_means
    y_un = y * y_stds + y_means
    sigma_un = sigma * y_stds
    assert mu_un.shape[1] == 1
    assert y_un.shape[1] == 1
    assert sigma_un.shape[1] == 1
    if torch.isnan(sigma_un).any() or torch.isnan(mu_un).any():
        return None
    dist = Normal(mu_un, sigma_un)
    return dist.log_prob(y_un).mean(axis=0).item()  # mean over datapoints

class homo_Gauss_mloglike(nn.Module):
    def __init__(self, Ndims=1, sig=None):
        super().__init__()
        if sig is None:
            self.log_std = nn.Parameter(torch.zeros(Ndims))
        else:
            self.log_std = nn.Parameter(torch.ones(
                Ndims) * np.log(sig), requires_grad=False)

    def forward(self, mu, y, model_std=None):
        sig = self.log_std.exp().clamp(min=1e-4)
        if model_std is not None:
            sig = (sig**2 + model_std**2)**0.5

        dist = Normal(mu, sig)
        return -dist.log_prob(y).mean(dim=0)


def torch_onehot(y, Nclass):
    if y.is_cuda:
        y = y.type(torch.cuda.LongTensor)
    else:
        y = y.type(torch.LongTensor)
    y_onehot = torch.zeros((y.shape[0], Nclass)).type(y.type())
    # In your for loop
    y_onehot.scatter_(1, y.unsqueeze(1), 1)
    return y_onehot


def np_get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


class DatafeedImage(data.Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.x_train[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y_train[index]

    def __len__(self):
        return len(self.x_train)


def get_loss_func(loss_func_name):
    """ return the desired loss function """

    if loss_func_name == "cross_entropy":
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    elif loss_func_name == "binary_cross_entropy":
        loss_func = nn.BCELoss(reduction='mean')
    elif loss_func_name == "mse":
        loss_func = nn.MSELoss(reduction="mean")
    elif loss_func_name == "gaussian":
        loss_func = homo_Gauss_mloglike()
    else:
        raise RuntimeError("Invalid loss function!")

    return loss_func


def get_num_classes(dataset):
    """ return the number of classes of the dataset """

    num_classes = {
        "Imagenet": 1000,
        "SmallImagenet": 1000,
        "CIFAR100": 100,
        "CIFAR10": 10,
        "SVHN": 10,
        "MNIST": 10,
        "Fashion": 10,
        "EMNIST": 47
    }

    return num_classes[dataset]


def instantiate_model(model, dataset, p_drop):
    """ Instantiate the model """

    initial_conv = '3x3' if dataset in ['Imagenet', 'SmallImagenet'] else '1x3'
    input_chanels = 1 if dataset in ['MNIST', 'Fashion', 'EMNIST'] else 3
    num_classes = get_num_classes(dataset)

    if model == 'resnet18':
        model_class = resnet18
    elif model == 'resnet18':
        model_class = resnet34
    elif model == 'resnet50':
        model_class = resnet50
    elif model == 'resnet101':
        model_class = resnet101
    else:
        raise Exception('requested model not implemented')

    model = model_class(num_classes=num_classes, zero_init_residual=True,
                        initial_conv=initial_conv, input_chanels=input_chanels, p_drop=p_drop)

    return model

def get_map_location(device):
    """ Get the map_location for loading a pre-trained model for the given device """

    # TODO: Support loading of multigpu models onto single GPU or CPU
    if device is None and torch.cuda.is_available(): # Allow loading models from multiGPU
        dev_string = 'cuda:0'
    elif device == 'cpu' or device == None:
        dev_string = device
    else: # Map model to be loaded to specified single gpu.
        if isinstance(device, str):
            device = device.replace('cuda:', '')
        else:
            device = device
        dev_string = 'cuda:{}'.format(device)

    return torch.device(dev_string)


def load_state_dict(model, checkpoint):
    """ Load the state_dict stored in the checkpoint into the model; 
        supports both models trained with the new training script,
        as well as the ImageNet ResNet-50 models trained with an old script """

    if 'model_state' in checkpoint.keys():
        # model_state_dict = {n: checkpoint['model_state'][n] for n in checkpoint['model_state'] if n in self.model.state_dict().keys()}
        model.load_state_dict(checkpoint['model_state'])

    elif 'state_dict' in checkpoint.keys(): # support for loading models trained with old script
        state_dict = checkpoint['state_dict']
        if check_dataparallel_save(state_dict):
            state_dict = strip_dataparallel_save(state_dict)
        model.load_state_dict(state_dict)


def parse_model_path(path):
    config = path.parts[-2].split("_")

    dataset = config[0]
    model = config[1]
    mode = config[2]
    num = int(config[-1])

    kwargs = [part.split(":") for part in config[3:-1]]
    kwargs = {kwarg[0]: kwarg[1] for kwarg in kwargs}
    kwargs = {key:
              int(val) if val.isdigit() else
              float(val) if val.replace('.', '', 1).isdigit() else
              None if val == 'None' else
              bool(val) if val in ["True", "False"] else
              val  # strings
              for key, val in kwargs.items()
              }

    rename_dict = {
        "bs": "batch_size", "e": "epochs", "wd": "weight_decay",
        "lr": "learning_rate", "m": "momentum", "es": "early_stop",
        "uv": "use_val", "pm": "prune_method", "pp": "prune_percent",
        "pl": "prune_locally", "npi": "n_prune_iters",
        "plam": "prune_lamb", "hm": "hessian_method",
        "si": "subset_idx"
    }

    options_dict = {}
    for key, val in kwargs.items():
        options_dict[rename_dict[key]] = val

    return dataset, model, mode, num, options_dict


def dict2csv(filename, dict):
    w = csv.writer(open(filename, "w"))
    for key, val in dict.items():
        w.writerow([key, val])


def plot_results_1d_regression(x_view, X_train, y_train, mean, std, dpi=120):
    import matplotlib.pyplot as plt
    """ Plot results for 1D regression benchmark """

    c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    subsample = 1

    plt.figure(dpi=dpi)
    plt.scatter(X_train[::subsample], y_train[::subsample], s=5, alpha=0.5, c=c[0])
    plt.plot(x_view, mean, c=c[3])


    plt.fill_between(x_view[:, 0],
                    (mean[:, 0] + std[:, 0]),
                    (mean[:, 0] - std[:, 0]), color=c[3], alpha=0.3)

    ylim = [min(min(mean[:, 0] - std[:, 0]) - 0.5, -4), max(max(mean[:, 0] + std[:, 0]) + 0.5, 4)]
    ylim = [max(ylim[0], -5), min(ylim[1], 5)]
    plt.ylim(ylim)
    plt.xlim([min(x_view), max(x_view)])
    plt.tight_layout()
    plt.show()
    

def get_n_data(dataloader):
    """ get number of data points in the data loader """

    if isinstance(dataloader.dataset, datasets.ImageFolder):
        return len(dataloader.dataset.imgs)
    else:
        return dataloader.dataset.data.shape[0]
