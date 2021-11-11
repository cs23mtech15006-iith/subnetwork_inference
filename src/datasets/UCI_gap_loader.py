import os
from os import path
import zipfile
try:
    import urllib
    from urllib import urlretrieve
except Exception:
    import urllib.request as urllib

import numpy as np

import sys
N_up = 1
nb_dir = '/'.join(os.getcwd().split('/')[:-N_up])
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

from src.utils import Datafeed


def load_gap_UCI(base_dir, dname, n_split=0, gap=True):

    if not path.exists(base_dir + '/UCI_for_sharing'):
        urllib.urlretrieve('https://javierantoran.github.io/assets/datasets/UCI_for_sharing.zip',
                           filename=base_dir + '/UCI_for_sharing.zip')
        with zipfile.ZipFile(base_dir + '/UCI_for_sharing.zip', 'r') as zip_ref:
            zip_ref.extractall(base_dir)

    np.random.seed(1234)
    dir_load = base_dir + '/UCI_for_sharing/standard/' + dname + '/data/'

    if gap:
        dir_idx = base_dir + '/UCI_for_sharing/gap/' + dname + '/data/'
    else:
        dir_idx = base_dir + '/UCI_for_sharing/standard/' + dname + '/data/'

    data = np.loadtxt(dir_load + 'data.txt')
    feature_idx = np.loadtxt(dir_load + 'index_features.txt').astype(int)
    target_idx = np.loadtxt(dir_load + 'index_target.txt').astype(int)

    test_idx_list = []
    train_idx_list = []

    for i in range(20):
        try:
            test_idx_list.append(np.loadtxt(dir_idx + 'index_test_%d.txt' % i).astype(int))
            train_idx_list.append(np.loadtxt(dir_idx + 'index_train_%d.txt' % i).astype(int))
        except:
            pass


    data_train = data[train_idx_list[n_split], :]
    data_test = data[test_idx_list[n_split], :]

    X_train = data_train[:, feature_idx].astype(np.float32)
    X_test = data_test[:, feature_idx].astype(np.float32)
    y_train = data_train[:, target_idx].astype(np.float32)
    y_test = data_test[:, target_idx].astype(np.float32)

    x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)
    y_means, y_stds = y_train.mean(axis=0), y_train.std(axis=0)

    x_stds[x_stds < 1e-10] = 1.

    X_train = ((X_train - x_means) / x_stds)
    y_train = ((y_train - y_means) / y_stds)[:, np.newaxis]
    X_test = ((X_test - x_means) / x_stds)
    y_test = ((y_test - y_means) / y_stds)[:, np.newaxis]

    return X_train, X_test, x_means, x_stds, y_train, y_test, y_means, y_stds


def unison_shuffled_copies(a, b):
    np.random.seed(0)
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def train_val_test_UCI(dname, base_dir='nb_dir/data/', prop_val=0.10, n_split=0):

    gap = False
    if dname in ['boston', 'concrete','energy', 'power', 'wine', 'yacht','kin8nm', 'naval', 'protein']:
        pass
    elif dname in ['boston_gap','concrete_gap', 'energy_gap', 'power_gap', 'wine_gap', 'yacht_gap',
                    'kin8nm_gap','naval_gap', 'protein_gap']:
        gap = True
        dname = dname[:-4]

    X_train, X_test, x_means, x_stds, y_train, y_test, y_means, y_stds = \
        load_gap_UCI(base_dir=base_dir, dname=dname, n_split=n_split, gap=gap)


    X_train = (X_train * x_stds) + x_means
    y_train = (y_train * y_stds) + y_means

    X_test = (X_test * x_stds) + x_means
    y_test = (y_test * y_stds) + y_means

    # print(X_train.shape)
    # Shuffle train to get independent val
    X_train, y_train = unison_shuffled_copies(X_train, y_train)

    Ntrain = int(X_train.shape[0] * (1-prop_val))
    X_val = X_train[Ntrain:]
    y_val = y_train[Ntrain:]
    X_train = X_train[:Ntrain]
    y_train = y_train[:Ntrain]

    # print(X_train.shape)
    x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)
    y_means, y_stds = y_train.mean(axis=0), y_train.std(axis=0)

    x_stds[x_stds < 1e-10] = 1.

    X_train = ((X_train - x_means) / x_stds)
    y_train = ((y_train - y_means) / y_stds)

    X_val = ((X_val - x_means) / x_stds)
    y_val = ((y_val - y_means) / y_stds)

    X_test = ((X_test - x_means) / x_stds)
    y_test = ((y_test - y_means) / y_stds)

    trainset = Datafeed(X_train, y_train, transform=None)
    valset = Datafeed(X_val, y_val, transform=None)
    testset = Datafeed(X_test, y_test, transform=None)

    N_train = X_train.shape[0]
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    return trainset, valset, testset, N_train, input_dim, output_dim, y_means, y_stds



if __name__ == '__main__':
    base_dir = '/home/ja666/rds/hpc-work/bayesian-lottery-tickets/experiments/datasets'
    n_split = 0
    np.random.seed(1234)
    dname = 'power'
    gap = True

    X_train, X_test, x_means, x_stds, y_train, y_test, y_means, y_stds = \
        load_gap_UCI(base_dir, dname, n_split=0, gap=True)

    trainset, valset, testset, N_train, input_dim, output_dim = \
        train_val_test_UCI(dname, base_dir=base_dir, prop_val=0.10, n_split=n_split)

    print(X_train.shape)
    print(N_train, input_dim, output_dim)
