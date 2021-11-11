import torch
from src.utils import unflatten_nn, flatten_nn, get_last_layer

def get_mask_from_weight_score_vec(model, weight_score_vec, n_weights_subnet, batchnorm_layers, largest, last_layer=False):
    """ compute mask based on the provided weight score vector """
    assert torch.all(weight_score_vec >= 0)

    if last_layer:
        D_last_layer = flatten_nn(get_last_layer(model), []).shape[0]
        weight_score_vec = weight_score_vec[-D_last_layer:]
        batchnorm_layers = []

    idx = torch.argsort(weight_score_vec, descending=largest)[:n_weights_subnet]
    idx = idx.sort()[0]
    mask_vec = torch.zeros_like(weight_score_vec)
    mask_vec[idx] = 1.

    # define layer-wise masks based on this threshold, to then prune weights with it
    model_ = get_last_layer(model) if last_layer else model
    mask = unflatten_nn(model_, batchnorm_layers, mask_vec)
    return mask, idx, weight_score_vec


def sample_mask_from_weight_score_vec(model, weight_score_vec, n_weights_subnet, batchnorm_layers, largest, last_layer=False):
    """ samlple mask based on the provided weight score vector """
    assert largest
    assert torch.all(weight_score_vec >= 0)

    if last_layer:
        D_last_layer = flatten_nn(get_last_layer(model), []).shape[0]
        weight_score_vec = weight_score_vec[-D_last_layer:]
        batchnorm_layers = []

    weight_probs = weight_score_vec / weight_score_vec.sum()

    idx = torch.multinomial(weight_probs, num_samples=n_weights_subnet, replacement=True)
    idx = idx.sort()[0]
    mask_vec = torch.zeros_like(weight_score_vec)
    mask_vec[idx] = 1.

    # define layer-wise masks based on this threshold, to then prune weights with it
    model_ = get_last_layer(model) if last_layer else model
    mask = unflatten_nn(model_, batchnorm_layers, mask_vec)
    return mask, idx, weight_score_vec
