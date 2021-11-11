import torch
import yaml
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.laplace.laplace import Laplace
from src.utils import get_num_classes, instantiate_model
from src.evaluation.evaluate_laplace import evaluate_laplace, aggregate_results

parser = argparse.ArgumentParser(description='Evaluate Methods')

# general parameters
parser.add_argument('--config', default=None, type=str,
                    help='path to YAML config file to use')
parser.add_argument('--seed', default=0, type=int,
                    help='random seed')
parser.add_argument('--si_model_path', default=None, type=str,
                    help='path to the Laplace model to evaluate')
parser.add_argument('--res_save_dir', default=None, type=str,
                    help='directory to save results in')
parser.add_argument('--model', type=str, default='resnet50',
                    choices=["resnet18", "resnet32", "resnet50", "resnet101", "lenet"],
                    help='model to load (default: resnet50)')
parser.add_argument('--dataset', type=str, default='MNIST',
                    choices=["CIFAR10", "CIFAR100", "SVHN", "MNIST", "Fashion", "EMNIST"],
                    help='dataset to use (default: MNIST)')
parser.add_argument('--test_batch_size', type=int, default=256,
                    help='the batch size to use for testing (default: 256)')
parser.add_argument('--device', default=None, type=int,
                    help='device to use; None means multi-GPU (default: None)')
parser.add_argument('--workers', default=4, type=int,
                    help='number of dataloader workers to use (default: 4)')
parser.add_argument('--data_dir', type=str, default='./data/',
                    help="where to find the data (default: ./data/)")
parser.add_argument('--p_drop', type=float, default=0,
                    help='dropout probability at conv layers (default: 0)')

# prediction parameters
parser.add_argument('--mode', type=str, choices=['linearized', 'forward_linearized'],
                    help='prediction method to run')
parser.add_argument('--lambd', default=1.0, type=float,
                    help='lambda for making predictions')

# parameters for running only certain corruptions/rotations – useful for parallel eval
parser.add_argument('--corruption', type=int, default=None,
                    choices=[0, 1, 2, 3, 4, 5],
                    help='0 corresponds to the test set. The default value of None means that all corruptions will be run.')
parser.add_argument('--rotation', type=int, default=None,
                    choices=[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180],
                    help='0 corresponds to the test set. The default value of None means that all rotations will be run.')
parser.add_argument('--skip_ood_eval', action='store_true',
                    help='whether to skip the OOD evaluations (default: False)')
parser.add_argument('--n_test_data', type=int, default=None,
                    help='Number of datapoints for the test (validation/corrupted/rotated/OoD) data. None means the full dataset (default: None)')
parser.add_argument('--test_subset_idx', default=-1, type=int,
                    help='Index of data subset to use. If -1, subsample randomly. (default: -1)')
parser.add_argument('--eval_on_test_subset', action='store_true',
                    help='Should we compute evaluation metrics on the test data subset provided? (default: False)')
parser.add_argument('--aggregate_subset_indices', default=None, type=int,
                    help='Number of indices of data subsets to aggregate. If None, ignore. (default: None)')

# training parameters (required for logging in the dataframe)
parser.add_argument('--train_epochs', default=None, type=int,
                    help='number of total epochs to run (if None, use dataset default)')
parser.add_argument('--mcsamples', default=1, type=int,
                    help='number of MC dropout samples to use at test time (default: 1)')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--train_batch_size', type=int, default=256,
                    help='the batch size used for training (default: 256)')
parser.add_argument('--noise_log_std', default=None, type=float,
                    help='log standard deviation of the noise for regression (default: None)')

# subnetwork selection parameters (required for logging in the dataframe)
parser.add_argument('--subnet_selection_method', default='random', type=str,
                    choices=['random', 'magnitude', 'min-wass', 'top-k-leverage', 'full_network'],
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

# Laplace parameters (required for logging in the dataframe)
parser.add_argument('--laplace_batch_size', default=128, type=int,
                    help='batch size to use for Jacobian computations (default: 128)')
parser.add_argument('--last_layer', action='store_true',
                    help='do Laplace inference over the last layer only? (default: False)')

# define default OOD datasets (used if args.skip_ood_eval is False)
target_datasets = {
    "MNIST": [(0, None, "Fashion")],
    "Fashion": [(None, None, "MNIST")],
    "CIFAR10": [(None, 0, "SVHN")],
    "CIFAR100": [(None, 0, "SVHN")],
    "SVHN": [(None, None, "CIFAR10")],
    "Imagenet": [(None, None, None)],
    "EMNIST": [(0, None, "Fashion")],
}

# define default corruptions (used if args.corruption is None)
corruptions = {
    "CIFAR10": [(None, cor, None) for cor in range(1, 6)],
    "CIFAR100": [(None, cor, None) for cor in range(1, 6)],
    "MNIST": [],
    "Fashion": [],
    "SVHN": [],
    "Imagenet": [(None, cor, None) for cor in range(0, 6)],
    "EMNIST": [],
}

# define default rotations (used if args.rotation is None)
rotations = {
    "CIFAR10": [],
    "CIFAR100": [],
    "MNIST": [(rot, None, None) for rot in range(15, 181, 15)],
    "Fashion": [],
    "SVHN": [],
    "Imagenet": [],
    "EMNIST": [(rot, None, None) for rot in range(15, 181, 15)],
}


def main(args):

    # load or create result dataframe
    corruption_str = "" if args.corruption is None else f"_{args.corruption}cor"
    data_subset_str = "" if args.n_test_data is None or args.eval_on_test_subset else f"_{args.test_subset_idx}test_subset_idx"
    df_path = Path(args.res_save_dir) / f"results{corruption_str}{data_subset_str}.csv"
    df_exists = df_path.exists()
    if df_exists:
        try:
            print("Loading existing results dataframe...")
            df = pd.read_csv(df_path)
        except Exception as E:
            print(E)
            df_exists = False

    if not df_exists:
        print("Creating new results dataframe...")
        if not df_path.parent.exists():
            df_path.parent.mkdir(parents=True, exist_ok=True)



        # define columns of result dataframe
        dtypes = np.dtype([
            # general
            ('seed', int),
            ('model', str),
            ('dataset', str),
            ('p_drop', float),
            # prediction
            ('mode', str),
            ('si_model_path', str),
            ('lambd', float),
            ('corruption', int),
            ('rotation', int),
            ('target_dataset', str),
            ('skip_ood_eval', bool),
            # training
            ('train_epochs', int),
            ('mcsamples', int),
            ('weight_decay', float),
            ('train_batch_size', int),
            ('noise_log_std', float),
            # subnet selection
            ('subnet_selection_method', str),
            ('n_weights_subnet', int),
            ('swag_batch_size', int),
            ('swag_n_snapshots', int),
            ('swag_lr', float),
            ('swag_c_epochs', int),
            ('swag_c_batches', int),
            ('swag_parallel', bool),
            ('last_layer', bool),
            # inference
            ('laplace_batch_size', int),
            # results
            ("ll", float),
            ("err", float),
            ("ece", float),
            ("brier", float),
            ("roc_auc", float),
            ("cr_err", float),
            ("fpr", float),
            ("tpr", float),
        ])
        data = np.empty(0, dtype=dtypes)
        df = pd.DataFrame(data)

    # define script arguments that should not be put into the dataframe (as they don't identify the experiment instance)
    args_to_skip = ["config", "model_path", "res_save_dir", "test_batch_size",
     "device", "workers", "data_dir", "save_dir", 'resume', 
     'epochs', 'batch_size', 'delete_old_saves', 'save_freq',  'gpu',
     'train_subset_idx', 'n_train_data', 'test_subset_idx', 'n_test_data',
     'aggregate_subset_indices', 'eval_on_test_subset']

    # build prototype of dataframe row by inserting all relevant hyperparameters
    # NOTE: having None values in the dataframe breaks checks for equality since
    # pandas treats None == None as false (unlike base Python);
    # we need working equality to avoid duplicating work
    row_to_add_proto = {}
    for arg in vars(args):
        if arg not in args_to_skip:
            val = vars(args)[arg]
            row_to_add_proto[arg] = val if val is not None else -1

    # don't load model if we just want to aggregate results
    if args.aggregate_subset_indices is None:
        # load Laplace model
        print("Loading Laplace model...")
        model = instantiate_model(args.model, args.dataset, args.p_drop)
        laplace_model = Laplace(model, save_path=args.si_model_path, device=args.device,
                                n_out=get_num_classes(args.dataset))
        laplace_model.load()

    print('Loaded model')
    print(rotations[args.dataset] + corruptions[args.dataset] + target_datasets[args.dataset])
    # run evaluations on model
    for (rot, cor, target_dataset) in rotations[args.dataset] + corruptions[args.dataset] + target_datasets[args.dataset]:
        if rot is None and cor is None and target_dataset is None:
            continue

        if args.corruption is not None:
            # if a corruption has been specified, skip all other corruptions
            if args.corruption != (cor if cor is not None else 0): continue

        if args.rotation is not None:
            # if a rotation has been specified, skip all other rotations
            if args.rotation != (rot if rot is not None else 0): continue

        if args.skip_ood_eval:
            if target_dataset is not None: continue

        row_to_add_proto.update({
            "rotation": rot if rot is not None else -1,
            "corruption": cor if cor is not None else -1,
            "target_dataset": target_dataset if target_dataset is not None else "None"
        })

        # skip if we already ran that experiment
        if len(df.loc[(df[list(row_to_add_proto)] == pd.Series(row_to_add_proto)).all(axis=1)]) > 0:
            print(f"Results exist for rot={rot}, cor={cor}, OOD={target_dataset} -- skiping experiment...")
            continue
        

        if args.aggregate_subset_indices is None:
            print(f"Running evaluation for rot={rot}, cor={cor}, OOD={target_dataset}...")
            results_dict = evaluate_laplace(laplace_model, args.dataset, args.data_dir, corruption=cor,
                                            rotation=rot, target_dataset=target_dataset,
                                            batch_size=args.test_batch_size, mode=args.mode,
                                            λ=args.lambd, n_test_data=args.n_test_data,
                                            test_subset_idx=args.test_subset_idx,
                                            workers=args.workers, res_path=df_path,
                                            eval_on_test_subset=args.eval_on_test_subset)
            print('results_dict', results_dict)
        else:
            print(f"Aggregating results for rot={rot}, cor={cor}, OOD={target_dataset}...")
            results_dict = aggregate_results(res_path=df_path, aggregate_indices=args.aggregate_subset_indices)

        if results_dict is None:
            continue

        # add results to dataframe
        row_to_add = row_to_add_proto.copy()

        for key in ["err", "ll", "brier", "ece"]:
            row_to_add[key] = results_dict[key]

        if target_dataset is not None:
            for key in ["roc_auc", "cr_err", "fpr", "tpr"]:
                row_to_add[key] = results_dict[key]

        # save dataframe
        print(f"Saving results to dataframe...")
        df = df.append(row_to_add, ignore_index=True)
        df.to_csv(df_path, index=False)


if __name__ == "__main__":
    # check PyTorch version
    assert "1.6" in torch.__version__, "PyTorch 1.6 is required to run this script!"

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

    if args.skip_ood_eval:
        raise ValueError('We dont currently support this option, better luck next time!')

    if args.n_test_data is not None and args.test_subset_idx == -1:
        raise ValueError('Need to specify data subset index to use!')

    main(args)
