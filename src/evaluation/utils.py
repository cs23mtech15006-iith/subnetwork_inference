import numpy as np
import copy

import torch
from torchvision import datasets, transforms

from src.utils import np_get_one_hot, DatafeedImage


def cat_callibration(probs, y_test, n_bins, top_k=None):
    all_preds = probs
    pred_class = np.argmax(all_preds, axis=1)

    pred_class_OH = np_get_one_hot(pred_class, probs.shape[1])
    targets_class_OH = np_get_one_hot(y_test.reshape(-1).astype(int), probs.shape[1])

    # indexing with top k is wrong
    if top_k is not None:
        top_k_idx = all_preds.argsort(axis=1)[:, -top_k:]

        all_preds = np.concatenate([all_preds[row_idxs, top_k_idx[row_idxs, :]]
                                    for row_idxs in range(top_k_idx.shape[0])], axis=0)
        targets_class_OH = np.concatenate([targets_class_OH[row_idxs, top_k_idx[row_idxs, :]]
                                           for row_idxs in range(top_k_idx.shape[0])], axis=0)
        pred_class_OH = np.concatenate([pred_class_OH[row_idxs, top_k_idx[row_idxs, :]]
                                        for row_idxs in range(top_k_idx.shape[0])], axis=0)

    expanded_preds = np.reshape(all_preds, -1)
    # These reshapes on the one hot vectors count every possible class as a different prediction
    pred_class_OH_expand = np.reshape(pred_class_OH, -1)
    targets_class_OH_expand = np.reshape(targets_class_OH, -1)
    correct_vec = (targets_class_OH_expand * (pred_class_OH_expand == targets_class_OH_expand)).astype(int)

    bin_limits = np.linspace(0, 1, n_bins + 1)
    bin_step = bin_limits[1] - bin_limits[0]
    bin_centers = bin_limits[:-1] + bin_step / 2

    bin_idxs = np.digitize(expanded_preds, bin_limits, right=False) - 1

    bin_counts = np.ones(n_bins)
    bin_corrects = np.zeros(n_bins)
    for nbin in range(n_bins+1):

        if nbin == n_bins:
            bin_counts[nbin-1] += np.sum((bin_idxs == nbin).astype(int))
            bin_corrects[nbin-1] += np.sum(correct_vec[bin_idxs == nbin])
        else:
            bin_counts[nbin] = np.sum((bin_idxs == nbin).astype(int))
            bin_corrects[nbin] = np.sum(correct_vec[bin_idxs == nbin])
    bin_probs = bin_corrects / (bin_counts + 1e-10)

    bin_probs[bin_counts == 0] = 0

    if top_k is not None:
        assert bin_counts.sum() == probs.shape[0] * top_k
    else:
        assert bin_counts.sum() == probs.shape[0] * probs.shape[1]

    # reference = bin_centers
    reference = np.array([expanded_preds[bin_idxs == nbin].mean() for nbin in range(n_bins)])
    reference[bin_counts == 0] = 0
    # TODO: calculate reference as average accuracy
    return bin_probs, bin_centers, bin_step, bin_counts, reference


def expected_callibration_error(bin_probs, reference, bin_counts, tail=False):
    bin_abs_error = np.abs(bin_probs - reference)
    if tail:
        tail_count = bin_counts[0] + bin_counts[-1]
        ECE = (bin_abs_error[0] * bin_counts[0] + bin_abs_error[-1] * bin_counts[-1]) / tail_count
    else:
        ECE = (bin_abs_error * bin_counts / bin_counts.sum(axis=0)).sum(axis=0)
    assert not np.isnan(ECE)
    return ECE


def load_corrupted_dataset(dname, severity, data_dir='../../data', batch_size=256, cuda=True, workers=4, n_data=None, subset_idx=-1):
    assert dname in ['CIFAR10', 'CIFAR100', 'Imagenet']

    transform_dict = {
        'CIFAR10': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]),
        'CIFAR100': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]),
        'Imagenet': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
    }

    if dname == 'CIFAR10':
        #x_file = data_dir + '/CIFAR-10-C/CIFAR10_c%d.npy' % severity
        x_file = data_dir + '/CIFAR-10-C_subsampled/CIFAR10_c%d.npy' % severity
        np_x = np.load(x_file)
        #y_file = data_dir + '/CIFAR-10-C/CIFAR10_c_labels.npy'
        y_file = data_dir + '/CIFAR-10-C_subsampled/CIFAR10_c_labels.npy'
        np_y = np.load(y_file).astype(np.int64)

        # subsample dataset if desired (either randomly or by subset index)
        if n_data is not None:
            if subset_idx == -1:
                np.random.seed(0)
                perm = np.random.permutation(np_x.shape[0])
                subset = perm[:n_data]
            else:
                subset = list(range(subset_idx * n_data, (subset_idx+1) * n_data))
            np_x = np_x[subset]
            np_y = np_y[subset]

        dataset = DatafeedImage(np_x, np_y, transform_dict[dname])

    elif dname == 'CIFAR100':
        x_file = data_dir + '/CIFAR-100-C/CIFAR100_c%d.npy' % severity
        np_x = np.load(x_file)
        y_file = data_dir + '/CIFAR-100-C/CIFAR100_c_labels.npy'
        np_y = np.load(y_file).astype(np.int64)
        dataset = DatafeedImage(np_x, np_y, transform_dict[dname])

    elif dname == 'Imagenet':
        dataset = datasets.ImageFolder(
            #data_dir + '/imagenet-c/%d' % severity,
            data_dir + '/imagenet-c-10k/%d' % severity,
            transform_dict[dname])

        # subsample dataset if desired (either randomly or by subset index)
        if n_data is not None:
            if subset_idx == -1:
                np.random.seed(0)
                perm = np.random.permutation(len(dataset.samples))
                subset = perm[:n_data]
            else:
                subset = list(range(subset_idx * n_data, (subset_idx+1) * n_data))

            dataset.samples = [dataset.samples[i] for i in subset]
            dataset.targets = [dataset.targets[i] for i in subset]

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=cuda)

    return loader


def rotate_load_dataset(dname, angle, data_dir='../../data', batch_size=256, cuda=True, workers=4, n_data=None, subset_idx=-1):
    assert dname in ['MNIST', 'Fashion', 'SVHN', 'CIFAR10', 'CIFAR100', 'EMNIST']

    transform_dict = {
        'MNIST': transforms.Compose([
            transforms.RandomRotation([angle, angle], resample=2, expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ]),
        'Fashion': transforms.Compose([
            transforms.RandomRotation([angle, angle], resample=2, expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ]),
        'SVHN': transforms.Compose([
            transforms.RandomRotation([angle, angle], resample=2, expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ]),
        'CIFAR10': transforms.Compose([
            transforms.RandomRotation([angle, angle], resample=2, expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]),
        'CIFAR100': transforms.Compose([
            transforms.RandomRotation([angle, angle], resample=2, expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]),
        'EMNIST': transforms.Compose([
            transforms.RandomRotation([angle, angle], resample=2, expand=False, center=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1751,), std=(0.3332,))
        ]),
        'SmallImagenet': None,
        'Imagenet': None,
    }

    dataset_dict = {
        'MNIST': datasets.MNIST,
        'Fashion': datasets.FashionMNIST,
        'SVHN': datasets.SVHN,
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100,
        'EMNIST': datasets.EMNIST,
        'SmallImagenet': None,
        'Imagenet': None,
    }

    dset_kwargs = {
        'root': data_dir,
        'train': False,
        'download': True,
        'transform': transform_dict[dname]
    }

    if dname == 'SVHN':
        del dset_kwargs['train']
        dset_kwargs['split'] = 'test'

    elif dname == 'EMNIST':
        dset_kwargs['split'] = 'balanced'

    source_dset = dataset_dict[dname](**dset_kwargs)

    # subsample source dataset if desired (either randomly or by subset index)
    if n_data is not None:
        if subset_idx == -1:
            np.random.seed(0)
            perm = np.random.permutation(source_dset.data.shape[0])
            subset = perm[:n_data]
        else:
            subset = list(range(subset_idx * n_data, (subset_idx+1) * n_data))
        source_dset.data = source_dset.data[subset]
        source_dset.targets = source_dset.targets[subset]

    source_loader = torch.utils.data.DataLoader(
        source_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=cuda)

    return source_loader


def cross_load_dataset(dname_source, dname_target, data_dir='../../data',
                        batch_size=256, workers=4, n_data=None, subset_idx=-1):
    assert dname_source in ['MNIST', 'Fashion', 'KMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'EMNIST']
    assert dname_target in ['MNIST', 'Fashion', 'KMNIST', 'SVHN', 'CIFAR10', 'CIFAR100', 'EMNIST']

    transform_dict = {
        'MNIST': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ]),
        'Fashion': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ]),
        'KMNIST': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1918,), std=(0.3483,))
        ]),
        'EMNIST': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1751,), std=(0.3332,))
        ]),
        'SVHN': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ]),
        'CIFAR10': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]),
        'CIFAR100': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
        ]),
        'SmallImagenet': None,
        'Imagenet': None,
    }

    dataset_dict = {'MNIST': datasets.MNIST,
                    'Fashion': datasets.FashionMNIST,
                    'KMNIST': datasets.KMNIST,
                    'EMNIST': datasets.EMNIST,
                    'SVHN': datasets.SVHN,
                    'CIFAR10': datasets.CIFAR10,
                    'CIFAR100': datasets.CIFAR100,
                    'SmallImagenet': None,
                    'Imagenet': None,
                    }

    dset_kwargs = {'root': data_dir,
                   'train': False,
                   'download': True,
                   'transform': transform_dict[dname_source]}

    source_dset_kwargs = dset_kwargs
    target_dset_kwargs = copy.copy(dset_kwargs)
    if dname_source == 'SVHN':
        del source_dset_kwargs['train']
        source_dset_kwargs['split'] = 'test'
    if dname_target == 'SVHN':
        del target_dset_kwargs['train']
        target_dset_kwargs['split'] = 'test'

    if dname_source == 'EMNIST':
        source_dset_kwargs['split'] = 'balanced'
    if dname_target == 'EMNIST':
        target_dset_kwargs['split'] = 'balanced'


    source_dset = dataset_dict[dname_source](**source_dset_kwargs)
    target_dset = dataset_dict[dname_target](**target_dset_kwargs)

    # subsample target dataset if desired (either randomly or by subset index)
    if n_data is not None:
        if subset_idx == -1:
            np.random.seed(0)
            perm = np.random.permutation(target_dset.data.shape[0])
            subset = perm[:n_data]
        else:
            subset = list(range(subset_idx * n_data, (subset_idx+1) * n_data))
        target_dset.data = target_dset.data[subset]
        if dname_target == "SVHN":
            target_dset.labels = target_dset.labels[subset]
        else:
            target_dset.targets = target_dset.targets[subset]

    source_loader = torch.utils.data.DataLoader(
        source_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    target_loader = torch.utils.data.DataLoader(
        target_dset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    return source_loader, target_loader
