import copy
import torch

from src.utils import flatten_nn, get_loss_func, get_n_params
from src.masking.utils import get_mask_from_weight_score_vec, sample_mask_from_weight_score_vec, get_mask_from_layer_aware_weight_score_vec, get_layer_param_sizes
from src.baselines.swag import fit_swag


def random_mask(model, batchnorm_layers, n_weights_subnet, device='cpu', last_layer=False):
    """ Compute a subnetwork mask uniformly at random.

    Args:
        model: the model to mask.
        n_weights_subnet: number of model weights to keep for the subnetwork.
    Returns:
        corresponding mask
    """

    # define uniform scores to then feed into mask sampler
    n_params = get_n_params(model, batchnorm_layers)
    if device is None and torch.cuda.is_available():
        device = "cuda:0"
    weight_score_vec = torch.ones(n_params, device=device)
    return sample_mask_from_weight_score_vec(model, weight_score_vec, n_weights_subnet, batchnorm_layers, largest=True, last_layer=last_layer)


def smallest_magnitude_mask(model, batchnorm_layers, n_weights_subnet, last_layer=False):
    """ prune the model weights with the smallest magnitude """

    weight_score_vec = flatten_nn(model, batchnorm_layers).pow(2)
    return get_mask_from_weight_score_vec(model, weight_score_vec, n_weights_subnet, batchnorm_layers, largest=True, last_layer=last_layer)


def wasserstein_mask(model, batchnorm_layers, n_weights_subnet, train_loader, device, loss="cross_entropy", n_snapshots=256, swag_lr=1e-2, swag_c_epochs=1, swag_c_batches=None, parallel=False, last_layer=False, weight_score_vec=None, layer_weight=None):

    device = "cuda:0" if device is None else device
    if not weight_score_vec:
        weight_score_vec = compute_wasserstein_swag_weight_score_vec(model,
                                                                     train_loader,
                                                                     batchnorm_layers,
                                                                     device=device,
                                                                     loss=loss,
                                                                     n_snapshots=n_snapshots,
                                                                     swa_lr=swag_lr,
                                                                     swa_c_epochs=swag_c_epochs,
                                                                     swa_c_batches=swag_c_batches,
                                                                     parallel=parallel)
    try:
        if layer_weight:
            if layer_weight not in ['forward', 'backward']:
                print('Invalid layer_weight!! Supported options: forward/backward')
                return None, None, weight_score_vec
            mask, idx, weight_score_vec = get_mask_from_layer_aware_weight_score_vec(model,
                                                                                    weight_score_vec,
                                                                                    n_weights_subnet,
                                                                                    batchnorm_layers,
                                                                                    largest=True,
                                                                                    last_layer=last_layer,
                                                                                    mode=layer_weight)

        else:
            mask, idx, weight_score_vec = get_mask_from_weight_score_vec(model,
                                                                         weight_score_vec,
                                                                         n_weights_subnet,
                                                                         batchnorm_layers,
                                                                         largest=True,
                                                                         last_layer=last_layer)
    except Exception as e:
        print("ERROR!!", e)
        return None, None, weight_score_vec
    return mask, idx, weight_score_vec



def compute_wasserstein_swag_weight_score_vec(model, train_loader, batchnorm_layers, device, loss, n_snapshots, swa_lr, swa_c_epochs, swa_c_batches, parallel):
    """ compute weight score vector required for Wasserstein pruning strategy using SWAG for variance estimation """
    loss_func = get_loss_func(loss).to(device)
    swag_model = fit_swag(copy.deepcopy(model), device, train_loader, loss_func, diag_only=True, max_num_models=n_snapshots, swa_lr=swa_lr, swa_c_epochs=swa_c_epochs, swa_c_batches=swa_c_batches, parallel=parallel)
    return swag_model.get_variance_vector(batchnorm_layers)


def snr_mask(model, batchnorm_layers, n_weights_subnet, train_loader, device, loss="cross_entropy", n_snapshots=256, swag_lr=1e-2, swag_c_epochs=1, swag_c_batches=None, parallel=False, last_layer=False):
    device = "cuda:0" if device is None else device

    weight_score_vec = compute_snr_swag_weight_score_vec(model, train_loader, batchnorm_layers, device=device, loss=loss, n_snapshots=n_snapshots,
                                                                swa_lr=swag_lr, swa_c_epochs=swag_c_epochs, swa_c_batches=swag_c_batches, parallel=parallel)

    # Note: In case of Wassertein distance, we want the weights with the highest scores
    # However, in case of SNR, the variance is in the denominator. So the lower the score, the higher the variance is.
    # So we want the nodes with lowest scores for uncertainty calculation
    weight_score_vec = torch.max(weight_score_vec) - weight_score_vec

    return get_mask_from_weight_score_vec(model, weight_score_vec, n_weights_subnet, batchnorm_layers, largest=True, last_layer=last_layer)


def compute_snr_swag_weight_score_vec(model, train_loader, batchnorm_layers, device, loss, n_snapshots, swa_lr, swa_c_epochs, swa_c_batches, parallel, epsilon = 1e-8):
    """ compute weight score vector required for SNR pruning strategy using SWAG for mean and variance estimation """
    loss_func = get_loss_func(loss).to(device)
    swag_model = fit_swag(copy.deepcopy(model), device, train_loader, loss_func, diag_only=True, max_num_models=n_snapshots, swa_lr=swa_lr, swa_c_epochs=swa_c_epochs, swa_c_batches=swa_c_batches, parallel=parallel)

    mean_vec = swag_model.get_mean_vector(batchnorm_layers)
    var_vec = swag_model.get_variance_vector(batchnorm_layers)

    snr_vec = torch.pow(mean_vec, 2) / (var_vec + epsilon) # Adding epsilon to avoid divide by zero