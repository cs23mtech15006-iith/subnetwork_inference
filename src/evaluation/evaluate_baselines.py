import torch
from torch import nn
from torch.nn import DataParallel

from tqdm import tqdm

from src.datasets.image_loaders import get_image_loader
from src.evaluation.utils import load_corrupted_dataset, rotate_load_dataset
from src.evaluation.metrics import get_all_classification_stats


def evaluate_map(model, dataset, data_dir, device, loss, corruption=None, rotation=None, batch_size=256,
                 workers=4, cuda=True, n_test_data=None, test_subset_idx=-1):

    assert not (corruption is not None and rotation is not None)
    if corruption is None and rotation is None:
        val_loader = get_image_loader(dataset, batch_size, cuda=cuda, workers=workers, distributed=False,
                                      data_dir=data_dir, n_val_data=n_test_data, val_subset_idx=test_subset_idx)[2]
    elif corruption is not None:
        val_loader = load_corrupted_dataset(dataset, severity=corruption, data_dir=data_dir, batch_size=batch_size,
                                            cuda=cuda, workers=workers, n_data=n_test_data, subset_idx=test_subset_idx)
    elif rotation is not None:
        val_loader = rotate_load_dataset(dataset, rotation, data_dir=data_dir, batch_size=batch_size, cuda=cuda,
                                         workers=workers, n_data=n_test_data, subset_idx=test_subset_idx)

    logprob_vec, target_vec = predict_map(model, val_loader, device, loss)

    return get_all_classification_stats(logprob_vec, target_vec)


def predict_map(model, data_loader, device, loss, noise_log_std=None):
    """ Make MAP predictions

    Args:
        data_loader: the data loader for the test/val/train set to make predictions on.

    Returns:
        a tuple of numpy arrays (preds, targets)
    """

    model.eval()

    if device is None and torch.cuda.is_available():
        model = DataParallel(model).cuda()

    preds = []
    targets = []
    with torch.no_grad():
        for inputs, tgts in tqdm(data_loader):
            if device is not None:
                inputs = inputs.to(device, non_blocking=True)

            # compute output
            preds.append(model(inputs).detach().cpu())
            targets.append(tgts.cpu())

    # define final layer activation function
    if loss == "cross_entropy":
        act = nn.LogSoftmax(dim=1)
    elif loss == "binary_cross_entropy":
        act = nn.LogSigmoid()
    elif loss == "mse" or loss == "gaussian":
        act = nn.Identity()
    else:
        raise RuntimeError("Invalid loss function!")

    preds = act(torch.cat(preds, dim=0))     # N x O
    targets = torch.cat(targets, dim=0)

    if device is None and torch.cuda.is_available():
        model = model.module

    if loss == "gaussian":
        stds = torch.ones_like(preds) * noise_log_std.exp()
        return torch.stack([preds.data, stds], dim=2), targets.data
    elif loss == "mse":
        stds = torch.ones_like(preds)
        return torch.stack([preds.data, stds], dim=2), targets.data
    else:
        return preds.data, targets.data
