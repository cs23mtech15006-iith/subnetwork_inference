import torch.nn as nn


class res_MLPBlock(nn.Module):
    def __init__(self, width, BN=True, act=nn.ReLU):
        super(res_MLPBlock, self).__init__()
        if BN:
            self.ops = [nn.BatchNorm1d(num_features=width), nn.Linear(width, width), act()]
        else:
            self.ops = [nn.Linear(width, width), act()]
        self.ops = nn.Sequential(*self.ops)
    def forward(self, x):
        return x + self.ops(x)


class MLPBlock(nn.Module):
    def __init__(self, width, BN=True, act=nn.ReLU):
        super(MLPBlock, self).__init__()
        if BN:
            self.ops = [nn.BatchNorm1d(num_features=width), nn.Linear(width, width), act()]
        else:
            self.ops = [nn.Linear(width, width), act()]
        self.ops = nn.Sequential(*self.ops)
    def forward(self, x):
        return self.ops(x)


class res_MLP(nn.Module):
    """Class for residual MLP, optionally with batchnorm"""
    def __init__(self, input_dim, output_dim, width, n_layers, BN=False, act=nn.ReLU):
        super(res_MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.width = width
        
        self.layers = [nn.Linear(self.input_dim, width), act()]
        for i in range(self.n_layers-1):
            self.layers.append(res_MLPBlock(width, BN=BN))
        if BN:
             self.layers.append(nn.BatchNorm1d(num_features=width))
        self.layers.append(nn.Linear(width, output_dim))
        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.layers(x)
