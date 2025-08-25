# Copyright (c) 2025 Rohit Jena. All rights reserved.
# 
# This file is part of FireANTs, distributed under the terms of
# the FireANTs License version 1.0. A copy of the license can be found
# in the LICENSE file at the root of this repository.
#
# IMPORTANT: This code is part of FireANTs and its use, reproduction, or
# distribution must comply with the full license terms, including:
# - Maintaining all copyright notices and bibliography references
# - Using only approved (re)-distribution channels 
# - Proper attribution in derivative works
#
# For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE 


'''
author: rohitrango
'''
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from typing import Union
from fireants.registration.deformation.abstract import AbstractDeformation
from fireants.io.image import Image, BatchedImages
from fireants.utils.imageutils import scaling_and_squaring, _find_integrator_n
from fireants.types import devicetype
from fireants.losses.cc import gaussian_1d, separable_filtering
from fireants.utils.imageutils import jacobian
from fireants.registration.optimizers.sgd import WarpSGD
from fireants.registration.optimizers.adam import WarpAdam, _get_smoothing_wrapper
from fireants.utils.globals import MIN_IMG_SHARDED_SIZE as MIN_IMG_SIZE
from fireants.registration.distributed import parallel_state

from logging import getLogger
from copy import deepcopy

class CompositiveDistributedWarp(nn.Module, AbstractDeformation):
    '''
    Class for compositive warp function (collects gradients of dL/dp)
    The image is computed as M \circ (\phi + u)
    '''
    def __init__(self, fixed_images: BatchedImages, moving_images: BatchedImages,
                optimizer: str = 'Adam', optimizer_lr: float = 1e-2, optimizer_params: dict = {},
                init_scale: int = 1, 
                smoothing_grad_sigma: float = 0.5, smoothing_warp_sigma: float = 0.5, 
                rank: int = 0,
                dim_to_shard: int = 0,
                freeform: bool = False,
                dtype: torch.dtype = torch.float32,
                ) -> None:
        super().__init__()
        self.num_images = num_images = max(fixed_images.size(), moving_images.size())
        # full and sharded dimensions
        spatial_shard_dims = fixed_images.batch_tensor.shape[2:]
        self.n_dims = len(spatial_shard_dims)  # number of spatial dimensions
        self.freeform = freeform
        # distributed parameters
        self.rank = rank
        self.dim_to_shard = dim_to_shard
        # permute indices
        self.permute_imgtov = (0, *range(2, self.n_dims+2), 1)  # [N, HWD, dims] -> [N, HWD, dims] -> [N, dims, HWD]
        self.permute_vtoimg = (0, self.n_dims+1, *range(1, self.n_dims+1))  # [N, dims, HWD] -> [N, HWD, dims]
        self.device = fixed_images.device
        if optimizer_lr > 1:
            getLogger("CompositiveWarp").warning(f'optimizer_lr is {optimizer_lr}, which is very high. Unexpected registration may occur.')

        # define warp and register it as a parameter
        # set size
        if init_scale > 1:
            spatial_shard_dims = [max(int(s / init_scale), MIN_IMG_SIZE) for s in spatial_shard_dims]
        warp = torch.zeros([num_images, *spatial_shard_dims, self.n_dims], dtype=dtype, device=fixed_images.device)  # [N, HWD, dims]
        self.register_parameter('warp', nn.Parameter(warp))

        # attach grad hook if smooothing of the gradient is required
        self.smoothing_grad_sigma = smoothing_grad_sigma
        self.image_padding = 0
        if smoothing_grad_sigma > 0:
            self.smoothing_grad_gaussians = [gaussian_1d(s, truncated=2) for s in (torch.zeros(self.n_dims, device=fixed_images.device, dtype=dtype) + smoothing_grad_sigma)]
            self.image_padding = [(len(x)-1)//2 for x in self.smoothing_grad_gaussians][self.dim_to_shard]
        self.smoothing_wrapper = _get_smoothing_wrapper(self)
        self.attach_grad_hook()

        # if the warp is also to be smoothed, add this constraint to the optimizer (in the optimizer_params dict)
        oparams = deepcopy(optimizer_params)
        self.smoothing_warp_sigma = smoothing_warp_sigma
        if self.smoothing_warp_sigma > 0:
            smoothing_warp_gaussians = [gaussian_1d(s, truncated=2) for s in (torch.zeros(self.n_dims, device=fixed_images.device, dtype=dtype) + smoothing_warp_sigma)]
            oparams['smoothing_gaussians'] = smoothing_warp_gaussians

        if oparams.get('freeform') is None:
            oparams['freeform'] = freeform

        # distributed parameters
        oparams['rank'] = self.rank
        oparams['dim_to_shard'] = self.dim_to_shard
        # add optimizer
        optimizer = optimizer.lower()
        if optimizer == 'sgd':
            self.optimizer = WarpSGD(self.warp, lr=optimizer_lr, dtype=dtype, **oparams)
        elif optimizer == 'adam':
            self.optimizer = WarpAdam(self.warp, lr=optimizer_lr, dtype=dtype, **oparams)
        else:
            raise NotImplementedError(f'Optimizer {optimizer} not implemented')
    
    def attach_grad_hook(self):
        ''' attack the grad hook to the velocity field if needed '''
        if self.smoothing_grad_sigma > 0:
            def grad_smoothing_hook(grad, gaussians):
                return self.smoothing_wrapper(grad, gaussians, self.image_padding)
            hook = partial(grad_smoothing_hook, gaussians=self.smoothing_grad_gaussians)
            self.warp.register_hook(hook)
    
    def initialize_grid(self):
        ''' initialize grid to a size 
        Simply use the grid from the optimizer, which should be initialized to the correct size
        '''
        self.grid = self.optimizer.grid

    def set_zero_grad(self):
        ''' set the gradient to zero (or None) '''
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()

    def get_warp(self):
        ''' return warp function '''
        warp = self.warp
        return warp
    
    def get_inverse_warp(self):
        raise NotImplementedError('Inverse warp not implemented for compositive warp, use `compositive_warp_inverse` from warputils.py instead')
    
    def set_size(self, size):
        # print(f"Setting size to {size}")
        ''' size: [H, W, D] or [H, W] '''
        mode = 'bilinear' if self.n_dims == 2 else 'trilinear'
        # get new displacement field
        warp = F.interpolate(self.warp.detach().permute(*self.permute_vtoimg), size=size, mode=mode, align_corners=True, 
                    ).permute(*self.permute_imgtov)
        self.register_parameter('warp', nn.Parameter(warp))
        # set new inverse displacement field
        # if len(self.inv.shape) > 1:
        #     self.inv = F.interpolate(self.inv.permute(*self.permute_vtoimg), size=size, mode=mode, align_corners=True).permute(*self.permute_imgtov)
        self.attach_grad_hook()
        self.optimizer.set_data_and_size(self.warp, size)
        # interpolate inverse warp if it exists
        self.initialize_grid()
