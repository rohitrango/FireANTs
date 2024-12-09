import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.nn.parallel import DistributedDataParallel as DDP
from logging import getLogger
from time import sleep

from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss 

logger = getLogger(__name__)

class LaplaceConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        # self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size) * 2 / np.sqrt(kernel_size**3 * (in_channels + out_channels)))
        self.weight = nn.Parameter(-torch.ones(out_channels, in_channels, kernel_size, kernel_size, kernel_size) * 2 / np.sqrt(kernel_size**3 * (in_channels + out_channels)))
        self.k2 = kernel_size // 2
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._init_weight()
    
    def _init_weight(self):
        fan_in_out = self.kernel_size ** 3 * (self.in_channels + self.out_channels)
        with torch.no_grad():
            o = self.weight.shape[0]
            self.weight[:o//2].data.mul_(-1)
            self.weight.data.add_(torch.rand_like(self.weight) / np.sqrt(fan_in_out))
            # self.weight[:, :, self.k2, self.k2, self.k2] = 1

    def forward(self, x):
        weight = self.weight
        k2 = self.k2
        w0 = torch.zeros_like(weight)
        w0[:, :, k2, k2, k2] = weight.flatten(2).sum(2)
        weight = weight - w0
        y = F.conv3d(x, weight, bias=None, stride=1, padding=k2)
        return y

class ReLUResNet(nn.Module):
    def __init__(self, *mods, a=0):
        super().__init__()
        actv = 'relu' if a <= 0 else 'leaky_relu'
        layers = []
        for mod in mods:
            layers.append(mod)
            layers.append(nn.ReLU() if actv == 'relu' else nn.LeakyReLU(negative_slope=a))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return x + self.layers(x)

class LearnableLaplacianFilter(nn.Module):
    def __init__(self, 
                 dims=3,
                 image_channels=1,
                 hidden_channels=8,
                 mode='laplacian', 
                 num_layers=2,
                 kernel_size=3,
                 use_global_scaling=False,
                 **kwargs
                 ):
        super().__init__()
        padding = kernel_size//2
        # create the modules
        if dims == 2:
            raise NotImplementedError
        elif dims == 3:
            module = LaplaceConv3d if mode == 'laplacian' else nn.Conv3d
        else:
            raise ValueError(f'Invalid number of dimensions: {dims}')
        self.dims = dims

        # init modules
        mods = [ReLUResNet(module(image_channels, hidden_channels, kernel_size=kernel_size, padding=padding))]
        for _ in range(num_layers):
            mods.append(ReLUResNet(module(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding)))
        mods.append(module(hidden_channels, image_channels, kernel_size=kernel_size, padding=padding))
        # mods.append(nn.Conv3d(hidden_channels, image_channels, kernel_size=1, padding=0, bias=False))
        # stack them onto layers
        self.layers = nn.Sequential(*mods)
        if use_global_scaling:
            self.global_scaling = nn.Parameter(torch.zeros(1))
            logger.info('Using global scaling')
        else:
            self.global_scaling = None

    def forward(self, x):
        # return x + self.layers(x)
        y = self.layers(x)
        # normalize
        ymin = y.flatten(2).min(2).values
        ymax = y.flatten(2).max(2).values
        xmin = x.flatten(2).min(2).values
        xmax = x.flatten(2).max(2).values
        # scaling
        y = (y - ymin) / (ymax - ymin + 1e-6) * (xmax - xmin)
        # multiply with global scaling
        if self.global_scaling is not None:
            y = y * torch.sigmoid(self.global_scaling)
        return x + y

class LaplacianTrainer:
    def __init__(self, 
                 laplace, 
                 lr=3e-4,
                 grad_accum_steps=1,
                 loss_fn='cc',
                 **kwargs
                 ):
        self.laplace = laplace
        loss_params = dict(kwargs.get('loss_params', {}))
        dims = laplace.dims if not isinstance(laplace, DDP) else laplace.module.dims

        if loss_fn == 'cc':
            self.loss_fn = LocalNormalizedCrossCorrelationLoss(spatial_dims=dims, **loss_params)
        elif loss_fn == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            raise NotImplementedError
        # init optimizer
        self.optimizer = Adam(self.laplace.parameters(), lr=lr)
        # self.optimizer = SGD(self.laplace.parameters(), lr=lr, momentum=0.9)
        self.grad_accum_steps = grad_accum_steps
        self.collected_grads = 0
        self.optimizer.zero_grad()
    
    def __call__(self, x):
        return self.laplace(x)
    
    def update_laplacian(self, template=None, moved_images=None):
        ''' given the template image and moved images'''
        if template is None:
            # finish up the gradient accumulation
            if self.collected_grads > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.collected_grads = 0
            return
        
        # template is not None, compute loss
        template_updated = self.laplace(template.detach())
        loss = self.loss_fn(template_updated, moved_images.detach())
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.laplace.parameters(), 0.1)
        self.collected_grads += 1
        # if we have collected enough gradients, update the weights
        # and return the new template
        if self.collected_grads == self.grad_accum_steps:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.collected_grads = 0
            return self.laplace(template).detach()
        else:
            return None
        