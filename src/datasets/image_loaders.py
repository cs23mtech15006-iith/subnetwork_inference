import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms, datasets


def get_image_loader(dname, batch_size, cuda, workers, distributed, data_dir='../../data', n_train_data=None, train_subset_idx=-1, n_val_data=None, val_subset_idx=-1, n_classes=-1, random_train=True):

    assert dname in ['EMNIST', 'MNIST', 'Fashion', 'SVHN', 'CIFAR10', 'CIFAR100', 'SmallImagenet', 'Imagenet']

    if dname == 'MNIST':
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True,
                                       transform=transform_train)
        val_dataset = datasets.MNIST(root=data_dir, train=False, download=True,
                                     transform=transform_test)
        input_channels = 1
        N_classes = 10

    if dname == 'EMNIST':
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1751,), std=(0.3332,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1751,), std=(0.3332,))
        ])

        train_dataset = datasets.EMNIST(root=data_dir, split='balanced', train=True, download=True,
                                       transform=transform_train)
        val_dataset = datasets.EMNIST(root=data_dir, split='balanced', train=False, download=True,
                                     transform=transform_test)
        input_channels = 1
        N_classes = 47

    elif dname == 'Fashion':

        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.2860,), std=(0.3530,))
        ])

        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True,
                                              transform=transform_train)
        val_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True,
                                            transform=transform_test)
        input_channels = 1
        N_classes = 10

    elif dname == 'SVHN':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])

        train_dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform_train)

        val_dataset = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform_test)
        input_channels = 3
        N_classes = 10

    elif dname == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)

        val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        input_channels = 3
        N_classes = 10

    elif dname == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)

        val_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
        input_channels = 3
        N_classes = 100

    elif dname == 'Imagenet':
        traindir = os.path.join(data_dir, 'imagenet/train')
        valdir = os.path.join(data_dir, 'imagenet/val')
        imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                imagenet_normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                imagenet_normalize,
            ]))

        input_channels = 3
        N_classes = 1000

    elif dname == 'SmallImagenet':
        traindir = os.path.join(data_dir, 'imagenet84/train')
        valdir = os.path.join(data_dir, 'imagenet84/val')
        small_imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transform=transforms.Compose([
                transforms.RandomCrop(84, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                transforms.ToTensor(),
                small_imagenet_normalize
            ])
        )
        val_dataset = datasets.ImageFolder(
            valdir, transform=transforms.Compose([
                transforms.ToTensor(),
                small_imagenet_normalize,
            ])
        )

        input_channels = 3
        N_classes = 1000

    # remove classes to just keep first n_classes
    if n_classes != -1:
        keep_idxs = train_dataset.targets < n_classes
        keep_idxs = list(np.arange(len(keep_idxs))[keep_idxs.numpy()])
        train_dataset = torch.utils.data.Subset(train_dataset, keep_idxs)

        keep_idxs_val = val_dataset.targets < n_classes
        keep_idxs_val = list(np.arange(len(keep_idxs_val))[keep_idxs_val.numpy()])
        val_dataset = torch.utils.data.Subset(val_dataset, keep_idxs_val)
    

    # subsample train and validation sets either randomly or by subset index
    train_dataset = subsample_data(train_dataset, n_train_data, train_subset_idx)
    val_dataset = subsample_data(val_dataset, n_val_data, val_subset_idx)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None) and random_train,
        num_workers=workers, pin_memory=cuda, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=cuda)

    Ntrain = len(train_dataset)
    Ntest = len(val_dataset)
    print('Ntrain: %d, Nval: %d' % (Ntrain, Ntest))

    return train_sampler, train_loader, val_loader, input_channels, N_classes, Ntrain


def subsample_data(dataset, n_data, subset_idx):
    """ subsample dataset either randomly or by subset index """

    if n_data is None:
        return dataset

    else:
        if subset_idx == -1:
            import numpy as np
            np.random.seed(0)
            subset = list(np.random.permutation(len(dataset))[:n_data])
        else:
            subset = list(range(subset_idx * n_data, (subset_idx+1) * n_data))
        return torch.utils.data.Subset(dataset, subset)