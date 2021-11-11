import torch
import yaml
import argparse
import traceback

from pathlib import Path
import numpy as np

from src.laplace.laplace import Laplace
from src.masking.masking import random_mask, wasserstein_mask, smallest_magnitude_mask
from src.utils import get_num_classes, instantiate_model, print_nonzeros, list_batchnorm_layers, \
            get_map_location, load_state_dict, model_to_device
from src.datasets.image_loaders import get_image_loader


parser = argparse.ArgumentParser(description='Subnetwork Inference')

# general parameters
parser.add_argument('--config', default=None, type=str,
                    help='path to YAML config file to use')
parser.add_argument('--seed', default=0, type=int,
                    help='random seed')
parser.add_argument('--model_path', default=None, type=str,
                    help='path to the pre-trained model')
parser.add_argument('--save_dir', default=None, type=str,
                    help='path where to save the results')
parser.add_argument('--model', type=str, default='resnet50',
                    choices=["resnet18", "resnet32", "resnet50", "resnet101"],
                    help='model to load (default: resnet50)')
parser.add_argument('--dataset', type=str, default='MNIST',
                    choices=["CIFAR10", "CIFAR100", "SVHN", "MNIST", "Fashion", "EMNIST"],
                    help='dataset to use (default: MNIST)')
parser.add_argument('--p_drop', type=float, default=0,
                    help='dropout probability at conv layers (default: 0)')
parser.add_argument('--data_dir', type=str, default='../data/',
                    help='directory where datasets are saved (default: ../data/)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--device', default=None, type=int,
                    help='device to use; None means multi-GPU (default: None)')

# subnetwork selection parameters
parser.add_argument('--subnet_selection_method', default='random', type=str,
                    choices=['random', 'magnitude', 'min-wass', 'top-k-leverage'],
                    help='subnetwork selection method (default: random)')
parser.add_argument('--n_weights_subnet', default=None, type=int,
                    help='number of weights in the subnet; None to keep all weights (default: None)')
parser.add_argument('--swag_batch_size', default=1024, type=int,
                    help='SWAG batch size for min-wass subnet selection (default: 1024)')
parser.add_argument('--swag_n_snapshots', default=256, type=int,
                    help='number of SWAG snapshots for min-wass subnet selection (default: 256)')
parser.add_argument('--swag_lr', default=1e-2, type=float,
                    help='SWAG learning rate for min-wass subnet selection (default: 1e-2)')
parser.add_argument('--swag_c_epochs', default=1, type=int,
                    help='number of epochs between SWAG snapshots for min-wass subnet selection (default: 1)')
parser.add_argument('--swag_c_batches', default=None, type=int,
                    help='number of batches between SWAG snapshots for min-wass subnet selection (default: None)')
parser.add_argument('--swag_parallel', action='store_true',
                    help='should SWAG be run on multi-GPU for min-wass subnet selection? (default: False)')

# Laplace parameters
parser.add_argument('--laplace_batch_size', default=128, type=int,
                    help='batch size to use for Jacobian computations (default: 128)')
parser.add_argument('--last_layer', action='store_true',
                    help='do Laplace inference over the last layer only? (default: False)')
parser.add_argument('--n_train_data', default=None, type=int,
                    help='number of datapoints for the train data. If None, use full dataset. (default: None)')
parser.add_argument('--train_subset_idx', default=-1, type=int,
                    help='index of data subset to use. If -1, subsample randomly. (default: -1)')


def main(args):
    # instantiate model
    model = instantiate_model(args.model, args.dataset, args.p_drop)

    map_location = get_map_location(args.device)
    checkpoint = torch.load(args.model_path, map_location=map_location)
    load_state_dict(model, checkpoint)
    model_to_device(model, args.device)
    del checkpoint

    # load dataset
    train_loader = get_image_loader(args.dataset, args.laplace_batch_size, cuda=True,
                                    workers=args.workers, distributed=False,
                                    data_dir=args.data_dir, n_train_data=args.n_train_data,
                                    train_subset_idx=args.train_subset_idx)[1]

    # compute and print subnetwork mask
    loss = 'cross_entropy'
    print("device", args.device)
    print(f"Selecting subnetwork via {args.subnet_selection_method}...")
    mask, index_mask, scores = compute_subnetwork_mask(model, args, train_loader, loss, args.last_layer)
    if mask is not None:
        print_nonzeros(mask)

    # fit Laplace approximation over subnetwork 
    print(f"Fitting Laplace approximation...")
    laplace_dir = Path(args.save_dir) / f'laplace.pth.tar'

    assert str(laplace_dir) not in args.model_path, "The save dir has been overwriten by the model path, job would not save properly"

    # instantiate Laplace model
    laplace_model = Laplace(model, mask=mask, index_mask=index_mask,
                            save_path=laplace_dir, device=args.device,
                            loss=loss, n_out=get_num_classes(args.dataset),
                            subnetwork_weight_scores=scores)

    # load MAP weights
    laplace_model.load(save_path=args.model_path)

    torch.cuda.empty_cache()

    # fit Laplace approximation
    laplace_model.fit_laplace(train_loader)


def compute_subnetwork_mask(model, args, loss, last_layer):
    """ compute subnetwork mask """

    # compute batchnorm layers
    bn_layers = list_batchnorm_layers(model)

    if args.n_weights_subnet == None:
        mask, index_mask, scores = None, None, None

    elif args.subnet_selection_method == "random":
        mask, index_mask, scores = random_mask(model, bn_layers, args.n_weights_subnet, device=args.device, last_layer=last_layer)

    elif args.subnet_selection_method == "magnitude":
        mask, index_mask, scores = smallest_magnitude_mask(model, bn_layers, args.n_weights_subnet, last_layer=last_layer)

    elif args.subnet_selection_method == "min-wass":
        swag_train_loader = get_image_loader(args.dataset, batch_size=args.swag_batch_size, cuda=True, workers=args.workers, distributed=False, data_dir=args.data_dir)[1]
        mask, index_mask, scores = wasserstein_mask(model, bn_layers, args.n_weights_subnet, swag_train_loader, args.device, loss=loss,
            n_snapshots=args.swag_n_snapshots, swag_lr=args.swag_lr, swag_c_epochs=args.swag_c_epochs, swag_c_batches=args.swag_c_batches, parallel=args.swag_parallel, last_layer=last_layer)

    return mask, index_mask, scores


if __name__ == '__main__':
    # check PyTorch version
    assert "1.8" in torch.__version__, "PyTorch 1.8 is required to run this script!"

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

    try:
        main(args)
    except RuntimeError as E:
        print(E)
        if 'CUDA out of memory' in str(E):
            print('Caught out of memory error, returning 0')
            traceback.format_exc()
            import sys
            sys.exit(0)
