import torch
import torch.nn as nn
from torch.nn import DataParallel
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from opt_einsum import contract

from src.laplace.utils import compute_jacobian_batched, compute_ggn_batched
from src.utils import get_map_location, list_batchnorm_layers, model_to_device, get_n_params, load_state_dict


class Laplace(nn.Module): 
    """Wrapper for Laplace approximating a model."""

    def __init__(self, model, save_path, noise_log_std=None, device=0,
                 loss="cross_entropy", n_out=10, mask=None, index_mask=None,
                 subnetwork_weight_scores=None):
        super().__init__()

        if loss not in ["cross_entropy", "binary_cross_entropy", "mse", "gaussian"]:
            raise RuntimeError("Invalid loss function!")

        self.device = device
        self.model = model_to_device(model, self.device)

        self.save_path = save_path
        folder = Path(self.save_path).parents[0]
        folder.mkdir(exist_ok=True, parents=True)

        self.n_out = n_out
        self.loss = loss

        # for pruned models
        self.mask = mask
        self.index_mask = index_mask
        self.subnetwork_weight_scores = subnetwork_weight_scores

        # for laplaced models
        self.μ = None
        self.H = None
        self.Σ = None

        # compute and store list of names of BatchNorm layers
        self.batchnorm_layers = list_batchnorm_layers(self.model)
        self.noise_log_std = noise_log_std


    def fit_laplace(self, train_loader, storage_device='cuda:0'):
        """ Determine the mean μ and Hessian/GGN H for the Laplace approximation.

        Args:
            train_loder: a data loader for the training set.
            storage_device: which GPU the hessian will be saved on. Only relevant if running on multiGPU, otherwise ignored.
        Returns:
            Nothing
        """

        self.model.eval()
        if self.index_mask is not None:
            D = self.index_mask.shape[0] 
        else:
            D = get_n_params(self.model, self.batchnorm_layers, None)
        
        # optionally parallelize model and store empty hessian on chosen device
        if self.device is None and torch.cuda.is_available():
            self.model = DataParallel(self.model).cuda()

            self.H = torch.zeros((D, D)).to(device=storage_device, non_blocking=True)
        else:
            self.H = torch.zeros((D, D)).to(device=self.device, non_blocking=True)

        for inputs, _ in tqdm(train_loader):
            if self.device is not None and self.device != 'cpu':
                inputs = inputs.to(device=self.device, non_blocking=True)

            compute_ggn_batched(
                inputs, self.model, self.index_mask, self.H, self.batchnorm_layers,
                loss=self.loss, n_out=self.n_out, noise_log_std=self.noise_log_std)

        if self.device is None and torch.cuda.is_available():
            self.model = self.model.module

        self.save()

    def predict(self, data_loader, λ=1., scaling=None):
        """ Make Bayesian predictions with the pruned or unpruned model.

        Args:
            data_loader: the data loader for the test/val/train set to make predictions on.
            λ: the (scalar) prior precision. (Default: 1.)
            scaling: the (scalar) scaling applied to the Hessian (Default: None)

        Returns:
            a tuple of numpy arrays (preds, targets)
        """
        self.model.eval()
        
        # compute the posterior covariance matrix Σ of the Laplace approximation, Σ = (sH + λI)^-1, with Hessian scaling 
        print("Computing covariance matrix...")
        H_scaling = λ if scaling is None else λ * scaling.pow(-1)

        # define final layer activation function
        if self.loss == "cross_entropy":
            out_dim = 1
            act = nn.LogSoftmax(dim=out_dim)
        elif self.loss == "binary_cross_entropy":
            act = nn.LogSigmoid()
        elif self.loss == "mse" or self.loss == "gaussian":
            act = nn.Identity()
        else:
            raise RuntimeError("Invalid loss function!")

        if self.Σ is None:
            self.H[np.diag_indices(self.H.size(0))] += H_scaling
            self.Σ = torch.inverse(self.H)
            self.H[np.diag_indices(self.H.size(0))] -= H_scaling

        self.model.eval()
        # optionally parallelize model
        if self.device is None and torch.cuda.is_available():
            self.model = DataParallel(self.model).cuda()

        preds = []  # p(y|x)
        targets = []

        for inputs, tgts in tqdm(data_loader):
            if not (self.device is None and torch.cuda.is_available()):
                inputs = inputs.to(device=self.device, non_blocking=True)

            # do forward pass to get mean predictions
            with torch.no_grad():
                f_xμ = self.model(inputs).detach()  # B x O

            g = compute_jacobian_batched(
                inputs, self.model, self.index_mask, self.batchnorm_layers, self.n_out)
 
            with torch.no_grad():
                d_gSg = contract('bij,jk,bik->bi', g, self.Σ, g)  # B x O
                if self.loss == "gaussian" or self.loss == "mse":
                    # batch-compute regression predictive in closed-form
                    noise_var = (2*self.noise_log_std).exp() if self.loss == "gaussian" else 1
                    covariance_diag = (d_gSg + noise_var).detach().cpu()
                    preds.append(torch.stack([f_xμ.detach().cpu(), covariance_diag.sqrt()], dim=2))
                    
                else:
                    # batch-compute linearized classification predictive via closed-form probit approximation
                    κ = 1 / torch.sqrt(1 + np.pi * 0.125 * d_gSg)  # B x O
                    py_xs = act(f_xμ * κ)   # B x O
                    preds.append(py_xs.detach().cpu())  # B x O

            targets.append(tgts.detach().cpu())

        preds = torch.cat(preds, dim=0)  # N x O
        targets = torch.cat(targets, dim=0)

        if self.device is None and torch.cuda.is_available():
            self.model = self.model.module

        return preds, targets


    def save(self, model=None):
        """ Save the model.

        Args:
            None.

        Returns:
            Nothing.
        """
        if model is None:
            model = self.model
        if isinstance(model, DataParallel):
            model = model.module
        torch.save({
            'model_state': model.state_dict(),
            'mask': self.mask,
            'index_mask': self.index_mask,
            'subnetwork_weight_scores': self.subnetwork_weight_scores,
            'H': self.H,
            'noise_log_std': self.noise_log_std,
        }, self.save_path)

    def load(self, save_path=None):
        """ Load the model.

        Args:
            save_path: tells us what file to load if it is different from the one specified when instantiating the class

        Returns:
            Nothing.
        """

        # load the pre-trained model weights
        save_path = self.save_path if save_path is None else save_path
        checkpoint = torch.load(save_path, map_location=get_map_location(self.device))
        load_state_dict(self.model, checkpoint)
        
        model_to_device(self.model, self.device)

        if "mask" in checkpoint.keys():
            self.mask = checkpoint['mask']

        if "index_mask" in checkpoint.keys():
            self.index_mask = checkpoint['index_mask']

        if "subnetwork_weight_scores" in checkpoint.keys():
            self.subnetwork_weight_scores = checkpoint['subnetwork_weight_scores']

        if "H" in checkpoint.keys():
            self.H = checkpoint['H']

        if 'noise_log_std' in checkpoint.keys():
            self.noise_log_std = checkpoint['noise_log_std']

        elif self.loss == 'gaussian' and 'nll_func' in checkpoint.keys():
            from src.utils import homo_Gauss_mloglike
            nll_func = homo_Gauss_mloglike()
            nll_func.load_state_dict(checkpoint['nll_func'])
            self.noise_log_std = nll_func.log_std.data
            self.noise_log_std = model_to_device(self.noise_log_std, self.device)
