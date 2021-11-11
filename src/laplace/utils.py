import torch
import torch.nn as nn

from opt_einsum import contract


def compute_jacobian_batched(xs, model, index_mask, batchnorm_layers, n_out):
    """ compute the Jacobian of the remaining model weights """

    from backpack import backpack, extend
    from backpack.extensions import BatchGrad

    # extend model (required for backpack)
    model.eval()
    model = extend(model)

    # loop over output dimensions and assemble Jacobian
    jacs = []
    for o in range(n_out):
        # do batched forward pass
        f = model(xs)

        # sum outputs over the batch (required for the backward pass) and extract the o-th output
        f_o = f.sum(dim=0)[o]

        # do batched backward pass
        with backpack(BatchGrad()):
            f_o.backward()

        # assemble batched Jacobian for output dimension o
        jacs_o = []
        for name, param in model.named_parameters():
            if "weight" in name and name not in batchnorm_layers:
                batch_size = param.grad_batch.size(0)
                jacs_o.append(param.grad_batch.reshape(batch_size, -1))
        jacs_o = torch.cat(jacs_o, dim=1)

        # apply mask to Jacobian
        if index_mask is not None:
            jacs_o = jacs_o[:, index_mask]

        jacs.append(jacs_o)   # B x D

    return torch.stack(jacs, dim=1)  # B x O x D


def compute_ggn_batched(xs, model, index_mask, H, batchnorm_layers, loss, n_out, noise_log_std=None):
    """ compute the generalized Gauss-Newton matrix of the remaining model weights in batches
        B: batch size, D_r: # of (remaining) weights, O: # of classes """

    model.eval()

    # batch-compute Jacobian of network output f w.r.t. remaining (non-zero) weights
    J_f = compute_jacobian_batched(xs, model, index_mask, batchnorm_layers, n_out)     # B x O x D

    # batch-compute Hessian H_L of loss L w.r.t. network outputs f
    H_L = compute_hessian_of_loss(
        xs, model, loss=loss, n_out=n_out, noise_log_std=noise_log_std)     # B x O x O

    # batch-compute generalized Gauss-Newton matrix G = J_f^T H_L J_f to approximate the Hessian and add it to running estimate
    with torch.no_grad():
        H.add_(contract('bod,bou,bue->de', J_f, H_L, J_f))    # D x D


def compute_hessian_of_loss(xs, model, loss="cross_entropy", n_out=10, noise_log_std=None):
    """ batch-compute Hessian H_L of loss L w.r.t. network outputs f """

    model.eval()
    with torch.no_grad():

        if loss == "cross_entropy":
            y_hat = torch.softmax(model(xs).detach(), dim=1)   # B x O
            H_L = torch.diag_embed(
                y_hat) - contract('bi,bj->bij', y_hat, y_hat)    # B x O x O

        elif loss == "binary_cross_entropy":
            y_hat = torch.sigmoid(model(xs).detach())    # B x 1
            H_L = (y_hat * (1 - y_hat)).unsqueeze(1)    # B x 1 x 1

        elif loss == "mse" or loss == "gaussian":
            H_device = 'cuda:0' if isinstance(model, nn.DataParallel) else xs.device
            H_L = torch.eye(n_out, device=H_device).unsqueeze(0).expand(
                xs.shape[0], -1, -1)    # B x O x O

            if loss == "gaussian":
                H_L = H_L * (noise_log_std * (-2)).exp()

        else:
            raise ValueError("Invalid loss function for GGN computation!")

    return H_L


def subsample_hessian(H, mask):
    """ Select the submatrix of the Hessian corresponding to the mask """

    if mask is None:
        return H

    # flatten mask, identify non-zero indices, and index Hessian
    mask_idx_subset = torch.nonzero(torch.cat([m.flatten() for m in mask.values()]), as_tuple=False).flatten().cpu()
    return H.cpu().index_select(0, mask_idx_subset).index_select(1, mask_idx_subset).to(H.device)
