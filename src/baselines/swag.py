"""
    implementation of SWAG, taken/adapted from:
    https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/swag.py
"""

import copy
import torch
from tqdm import tqdm
from src.baselines.utils import flatten, unflatten_like


def fit_swag(model, device, train_loader, loss_func, diag_only=True, max_num_models=20, swa_c_epochs=1, swa_c_batches=None, swa_lr=0.01, momentum=0.9, wd=3e-4, mask=None, parallel=False):
    """
    Fit SWAG model
    (adapted from https://github.com/wjmaddox/swa_gaussian/blob/master/experiments/train/run_swag.py)

    Args:
        device: the device to run the models on - 'cpu' or 'cuda'
        train_loader: Dataloader for train dataset
        loss_func: Instance of nn.CrossEntropyLoss or nn.BCELoss or nn.MSELoss
        diag_only: bool flag to only store diagonal of SWAG covariance matrix (Default: True)
        max_num_models: int for maximum number of SWAG models to save (Default: 20)
        swa_c_epochs: int for SWA model collection frequency/cycle length in epochs (Default: 1)
        swa_c_batches: int for SWA model collection frequency/cycle length in batches (Default: None)
        swa_lr: float for SWA learning rate; use 0.05 for CIFAR100 and 0.01 otherwise (Default: 0.01)
        momentum: float for SGD momentum (Default: 0.9)
        wd: float for weight decay (Default: 3e-4)
        mask: dict of subnetwork masks (Default: None)
        parallel: data parallel model switch (default: False)
    """

    if swa_c_epochs is not None and swa_c_batches is not None:
        raise RuntimeError("One of swa_c_epochs or swa_c_batches must be None!")

    if parallel:
        print("Using Data Parallel model")
        model = torch.nn.DataParallel(model).cuda(device)

    swag_model = SWAG(copy.deepcopy(model), no_cov_mat=diag_only, max_num_models=max_num_models, mask=mask).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=swa_lr, momentum=momentum, weight_decay=wd)

    model.train()
    if swa_c_epochs is not None:
        n_epochs = swa_c_epochs * max_num_models
    else:
        n_epochs = 1 + (max_num_models * swa_c_batches) // len(train_loader)
    for epoch in tqdm(range(int(n_epochs))):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            loss = loss_func(model(inputs), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if swa_c_batches is not None and (batch_idx+1) % swa_c_batches == 0:
                swag_model.collect_model(model)
                if swag_model.n_models == max_num_models:
                    break

        if swa_c_epochs is not None and epoch % swa_c_epochs == 0:
            swag_model.collect_model(model)

    return swag_model


class SWAG(torch.nn.Module):
    def __init__(self, base, no_cov_mat=True, max_num_models=0, var_clamp=1e-30, mask=None):
        super(SWAG, self).__init__()

        self.register_buffer("n_models", torch.zeros([1], dtype=torch.long))
        self.params = list()

        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models

        self.var_clamp = var_clamp
        self.mask = mask

        self.base = base
        self.init_swag_parameters(params=self.params, no_cov_mat=self.no_cov_mat, mask=self.mask)
        #self.base.apply(lambda module: swag_parameters(module=module, params=self.params, no_cov_mat=self.no_cov_mat, only_nonzero=self.only_nonzero))

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def init_swag_parameters(self, params, no_cov_mat=True, mask=None):
        for mod_name, module in self.base.named_modules():
            for name in list(module._parameters.keys()):
                if module._parameters[name] is None:
                    continue

                name_full = f"{mod_name}.{name}".replace(".", "-")
                data = module._parameters[name].data
                module._parameters.pop(name)
                module.register_buffer("%s_mean" % name_full, data.new(data.size()).zero_())
                module.register_buffer("%s_sq_mean" % name_full, data.new(data.size()).zero_())

                if no_cov_mat is False:
                    if mask and name_full.replace("-", ".") in mask:
                        data = data[mask[name_full.replace("-", ".")].nonzero(as_tuple=True)]
                    module.register_buffer("%s_cov_mat_sqrt" % name_full, data.new_empty((0, data.numel())).zero_())

                params.append((module, name_full))

    def get_mean_vector(self, batchnorm_layers, mask=None):
        mean_list = []
        for module, name in self.params:
            name_full = name.replace("-", ".")
            name_full = name_full.replace('module.', '')
            if 'weight' in name_full and name_full not in batchnorm_layers:
                mean = module.__getattr__("%s_mean" % name)
                if mask is not None:
                    mean = mean[mask[name_full].nonzero(as_tuple=True)]
                mean_list.append(mean.cpu())
        return flatten(mean_list)

    def get_variance_vector(self, batchnorm_layers, mask=None):
        mean_list = []
        sq_mean_list = []

        for module, name in self.params:
            name_full = name.replace("-", ".")
            name_full = name_full.replace('module.', '')
            if 'weight' in name_full and not any(bn in name_full for bn in batchnorm_layers):
                mean = module.__getattr__("%s_mean" % name)
                sq_mean = module.__getattr__("%s_sq_mean" % name)

                if mask is not None:
                    mean = mean[mask[name_full].nonzero(as_tuple=True)]
                    sq_mean = sq_mean[mask[name_full].nonzero(as_tuple=True)]

                mean_list.append(mean.cpu())
                sq_mean_list.append(sq_mean.cpu())

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        variances = torch.clamp(sq_mean - mean ** 2, self.var_clamp)

        return variances

    def get_covariance_matrix(self, batchnorm_layers, eps=1e-10):
        if self.no_cov_mat:
            raise RuntimeError("No covariance matrix was estimated!")

        cov_mat_sqrt_list = []
        for module, name in self.params:
            name_full = name.replace("-", ".")
            name_full = name_full.replace('module.', '')
            if 'weight' in name_full and name_full not in batchnorm_layers:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                cov_mat_sqrt_list.append(cov_mat_sqrt.cpu())

        # build low-rank covariance matrix
        cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)
        print(cov_mat_sqrt.shape)
        cov_mat = torch.matmul(cov_mat_sqrt.t(), cov_mat_sqrt)
        cov_mat /= (self.max_num_models - 1)
        print(cov_mat.shape)

        # obtain covariance matrix by adding variances (+ eps for numerical stability) to diagonal and scaling
        var = self.get_variance_vector(batchnorm_layers, mask=self.mask) + eps
        cov_mat.add_(torch.diag(var)).mul_(0.5)

        return cov_mat


    def sample(self, scale=1.0, cov=False, seed=None, block=False):
        if seed is not None:
            torch.manual_seed(seed)

        if not block:
            self.sample_fullrank(scale, cov)
        else:
            self.sample_blockwise(scale, cov)

    def sample_blockwise(self, scale, cov):
        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)

            sq_mean = module.__getattr__("%s_sq_mean" % name)
            eps = torch.randn_like(mean)

            var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)

            scaled_diag_sample = scale * torch.sqrt(var) * eps

            if cov is True:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                eps = cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0), 1)).normal_()
                cov_sample = (
                    scale / ((self.max_num_models - 1) ** 0.5)
                ) * cov_mat_sqrt.t().matmul(eps).view_as(mean)

                w = mean + scaled_diag_sample + cov_sample

            else:
                w = mean + scaled_diag_sample

            module.__setattr__(name.split("-")[-1], w)

    def sample_fullrank(self, scale, cov):
        scale_sqrt = scale ** 0.5

        mean_list = []
        sq_mean_list = []

        if cov:
            cov_mat_sqrt_list = []

        for (module, name) in self.params:
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            if cov:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                cov_mat_sqrt_list.append(cov_mat_sqrt.cpu())

            mean_list.append(mean.cpu())
            sq_mean_list.append(sq_mean.cpu())

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        # draw diagonal variance sample
        var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        # if covariance draw low rank sample
        if cov:
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)

            cov_sample = cov_mat_sqrt.t().matmul(
                cov_mat_sqrt.new_empty(
                    (cov_mat_sqrt.size(0),), requires_grad=False
                ).normal_()
            )
            cov_sample /= (self.max_num_models - 1) ** 0.5

            rand_sample = var_sample + cov_sample

        else:
            rand_sample = var_sample

        # update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)

        for (module, name), sample in zip(self.params, samples_list):
            module.__setattr__(name.split("-")[-1], sample.cuda())

    def collect_model(self, base_model):
        for (module, name), base_param in zip(self.params, base_model.parameters()):
            data = base_param.data

            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            # first moment
            mean = mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + data / (self.n_models.item() + 1.0)

            # second moment
            sq_mean = sq_mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + data ** 2 / (self.n_models.item() + 1.0)

            # square root of covariance matrix
            if self.no_cov_mat is False:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

                # block covariance matrices, store deviation from current mean
                dev = (data - mean)
                name_full = name.replace("-", ".")
                if self.mask and name_full in self.mask:
                    dev = dev[self.mask[name_full].nonzero(as_tuple=True)]
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1, 1).t()), dim=0)

                # remove first column if we have stored too many models
                if (self.n_models.item() + 1) > self.max_num_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]
                module.__setattr__("%s_cov_mat_sqrt" % name, cov_mat_sqrt)

            module.__setattr__("%s_mean" % name, mean)
            module.__setattr__("%s_sq_mean" % name, sq_mean)
        self.n_models.add_(1)

    def load_state_dict(self, state_dict, strict=False):
        if not self.no_cov_mat:
            n_models = state_dict["n_models"].item()
            rank = min(n_models, self.max_num_models)
            for module, name in self.params:
                mean = module.__getattr__("%s_mean" % name)
                module.__setattr__(
                    "%s_cov_mat_sqrt" % name,
                    mean.new_empty((rank, mean.numel())).zero_(),
                )
        super(SWAG, self).load_state_dict(state_dict, strict)