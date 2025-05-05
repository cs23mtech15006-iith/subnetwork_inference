import torch
from src.utils import unflatten_nn, flatten_nn, get_last_layer
from src.baselines.swag import SWAG

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


def get_mask_from_layer_aware_weight_score_vec(model, weight_score_vec, n_weights_subnet, batchnorm_layers, largest, last_layer=False, mode='backward'):
    """ Compute mask using layer-aware reweighting of the score vector. """

    assert torch.all(weight_score_vec >= 0)

    layer_param_sizes = get_layer_param_sizes(model, batchnorm_layers)

    assert sum(layer_param_sizes) == weight_score_vec.numel(), \
        f"Mismatch: {sum(layer_param_sizes)} != {weight_score_vec.numel()}"

    # Assign layer priority weights
    n_layers = len(layer_param_sizes)
    if mode == 'backward': # highest weight to last layer
        layer_weights = [0.9 ** (n_layers - 1 - i) for i in range(n_layers)]
    else: # highest weight to 1st layer
        layer_weights = [0.9 ** i for i in range(n_layers)]

    # Build layer-aware weight vector
    layer_weight_vec = torch.cat([
        torch.full((size,), layer_weights[i], device=weight_score_vec.device)
        for i, size in enumerate(layer_param_sizes)
    ])

    adjusted_score_vec = weight_score_vec * layer_weight_vec

    # Truncate to last layer if needed
    if last_layer:
        D_last_layer = flatten_nn(get_last_layer(model), []).shape[0]
        adjusted_score_vec = adjusted_score_vec[-D_last_layer:]
        weight_score_vec = weight_score_vec[-D_last_layer:]
        batchnorm_layers = []

    # Select top-k indices
    idx = torch.argsort(adjusted_score_vec, descending=largest)[:n_weights_subnet]
    idx = idx.sort()[0]
    mask_vec = torch.zeros_like(adjusted_score_vec)
    mask_vec[idx] = 1.

    # Unflatten to layer-wise mask
    model_ = get_last_layer(model) if last_layer else model
    mask = unflatten_nn(model_, batchnorm_layers, mask_vec)

    return mask, idx, weight_score_vec


def get_layer_param_sizes(model, batchnorm_layers_set):
    sizes = []
    swag_model = None
    for name, module in model.named_modules():
        if isinstance(module, SWAG):
            swag_model = module
            break

    if swag_model:
        for mod, param_name_swag in swag_model.params:
            full_name_swag = param_name_swag.replace("-", ".")
            if 'weight' in full_name_swag and full_name_swag not in batchnorm_layers_set:
                param = mod._buffers.get(f"{param_name_swag.split('-')[-1]}_mean") # Access a buffer to get size
                if param is not None:
                    sizes.append(param.numel())
    else:
        for name, param in model.named_parameters():
            if 'weight' in name and name not in batchnorm_layers_set:
                sizes.append(param.numel())

    print(f"Total weight parameters counted (get_layer_param_sizes): {sum(sizes)}")
    return sizes