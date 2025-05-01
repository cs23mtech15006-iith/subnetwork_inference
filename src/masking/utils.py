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


def get_mask_from_layer_aware_weight_score_vec(model, weight_score_vec, n_weights_subnet, batchnorm_layers, largest, last_layer=False):
    """ Compute mask using layer-aware reweighting of the score vector.
    
    Applies layer-priority scaling to the weight score vector before selecting top-k weights.
    The most recent layer gets the highest priority (scaling factor of 1), and earlier layers are
    scaled by successive powers of 0.9.
    """
    assert torch.all(weight_score_vec >= 0)

    # Flatten the model and keep track of layer-wise parameter boundaries
    layer_param_sizes = []
    flat_params = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear) or (layer not in batchnorm_layers):
            layer_flat = flatten_nn(layer, [])
            if layer_flat.numel() > 0:
                layer_param_sizes.append(layer_flat.numel())
                flat_params.append(layer_flat)
    assert sum(layer_param_sizes) == weight_score_vec.numel()

    # Assign layer weights: newer layers get higher priority
    n_layers = len(layer_param_sizes)
    layer_weights = [0.9 ** (n_layers - 1 - i) for i in range(n_layers)]

    # Build a full weight adjustment vector based on layer-wise scaling
    layer_weight_vec = torch.cat([
        torch.full((size,), fill_value=layer_weights[i], device=weight_score_vec.device)
        for i, size in enumerate(layer_param_sizes)
    ])

    assert layer_weight_vec.shape == weight_score_vec.shape

    # Apply layer-aware weighting
    adjusted_score_vec = weight_score_vec * layer_weight_vec

    # Continue as in the original function
    if last_layer:
        D_last_layer = flatten_nn(get_last_layer(model), []).shape[0]
        adjusted_score_vec = adjusted_score_vec[-D_last_layer:]
        weight_score_vec = weight_score_vec[-D_last_layer:]  # for return
        batchnorm_layers = []

    idx = torch.argsort(adjusted_score_vec, descending=largest)[:n_weights_subnet]
    idx = idx.sort()[0]
    mask_vec = torch.zeros_like(adjusted_score_vec)
    mask_vec[idx] = 1.

    # Reconstruct layer-wise binary mask
    model_ = get_last_layer(model) if last_layer else model
    mask = unflatten_nn(model_, batchnorm_layers, mask_vec)

    return mask, idx, weight_score_vec  # return original score vec (not adjusted one)

