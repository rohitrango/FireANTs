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


''' class for SGD for compositive warps '''
import torch
from torch.nn import functional as F
# from fireants.utils.imageutils import compute_inverse_warp_displacement
from fireants.utils.imageutils import jacobian as jacobian_fn
from fireants.losses.cc import separable_filtering
from fireants.interpolator import fireants_interpolator
from fireants.registration.distributed.utils import add_distributed_padding, crop_distributed_padding
import logging
logger = logging.getLogger(__name__)
from fireants.registration.distributed import parallel_state

import os
import logging
logger = logging.getLogger(__name__)
USE_NO_GP = bool(int(os.environ.get('USE_NO_GP', '0').lower()))

def adam_update_fused(grad, exp_avg, exp_avg_sq, beta1, beta2, eps):
    grad.copy_(exp_avg / (beta1) / (exp_avg_sq / (beta2)).sqrt().add_(eps))

try:
    import fireants_fused_ops as ffo
    adam_update_fused = ffo.adam_update_fused
except ImportError:
    logger.warning("Fused ops not found, using baseline implementation")

## Function for smoothing
def _get_smoothing_wrapper(optimizer):
    ''' 
    Get wrapper for smoothing
    '''
    gp_group = parallel_state.get_parallel_state().get_current_gp_group() if parallel_state.is_initialized() else [0]
    gp_size = len(gp_group)

    if gp_size <= 1:
        def smoothing_wrapper_nodist(tensor, kernels, padding=0):
            # tensor is a warp field
            tensor = separable_filtering(tensor.permute(*optimizer.permute_vtoimg).contiguous(), kernels).permute(*optimizer.permute_imgtov).contiguous()
            return tensor
        return smoothing_wrapper_nodist
    else:
        # write a distributed version
        def smoothing_wrapper_dist(tensor, kernels, padding=0):
            if padding > 0:
                tensor = add_distributed_padding(tensor, padding, optimizer.dim_to_shard-1) # 2 will be added to dim_to_shard anyway
            tensor = separable_filtering(tensor.permute(*optimizer.permute_vtoimg).contiguous(), kernels).permute(*optimizer.permute_imgtov).contiguous()
            if padding > 0:
                tensor = crop_distributed_padding(tensor, padding, optimizer.dim_to_shard-1)
            return tensor
        return smoothing_wrapper_dist

class WarpAdam:
    ''' at the moment we only support a single warp function 
    also supports multi-scale (by simply interpolating to the target size)
    shape of warp = [B, H, W, [D], dims]
    '''
    def __init__(self, warp, lr, 
                 beta1=0.9, beta2=0.99, weight_decay=0, eps=1e-8,
                 scaledown=False, multiply_jacobian=False,
                 smoothing_gaussians=None, 
                 grad_gaussians=None,
                 freeform=False,
                 offload=False,   # try offloading to CPU
                 reset=False,
                 # distributed params
                 rank: int = 0, 
                 dim_to_shard: int = 0,
                 dtype: torch.dtype = torch.float32):
        # init
        self.dtype = dtype
        if beta1 < 0.0 or beta1 >= 1.0:
            raise ValueError("Invalid beta1 value: {}".format(beta1))
        if beta2 < 0.0 or beta2 >= 1.0:
            raise ValueError("Invalid beta2 value: {}".format(beta2))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if lr < 0.0:
            raise ValueError("Invalid lr value: {}".format(lr))
        self.n_dims = len(warp.shape) - 2
        # get half resolutions
        self.half_resolution = 1.0/(max(warp.shape[1:-1]) - 1)
        self.warp = warp
        self.freeform = freeform
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.step_t = 0    # initialize step to 0
        self.weight_decay = weight_decay
        self.multiply_jacobian = multiply_jacobian
        self.reset = reset
        self.scaledown = scaledown   # if true, the scale the gradient even if norm is below 1
        # offload params
        self.device = warp.device
        self.offload = offload
        # warp grad params
        self.exp_avg = torch.zeros_like(warp, device=self.device if not self.offload else 'cpu')
        self.exp_avg_sq = torch.zeros_like(warp, device=self.device if not self.offload else 'cpu')
        self.permute_imgtov = (0, *range(2, self.n_dims+2), 1)  # [N, HWD, dims] -> [N, HWD, dims] -> [N, dims, HWD]
        self.permute_vtoimg = (0, self.n_dims+1, *range(1, self.n_dims+1))  # [N, dims, HWD] -> [N, HWD, dims]
        # set grid
        self.batch_size = batch_size = warp.shape[0]
        # init grid
        self.affine_init = torch.eye(self.n_dims, self.n_dims+1, device=warp.device, dtype=dtype)[None].expand(batch_size, -1, -1).contiguous()
        self.initialize_grid(warp.shape[1:-1])
        # gaussian smoothing parameters (if any)
        self.smoothing_gaussians = smoothing_gaussians
        self.grad_gaussians = grad_gaussians            # unused
        # distributed params
        self.rank = rank
        self.dim_to_shard = dim_to_shard
        # get padding lengths
        if self.smoothing_gaussians is not None:
            self.padding_smoothing = [len(x) for x in self.smoothing_gaussians] if isinstance(self.smoothing_gaussians, (list, tuple)) else [len(self.smoothing_gaussians) for _ in range(self.n_dims)]
            self.padding_smoothing = (self.padding_smoothing[self.dim_to_shard] - 1) // 2
        else:
            self.padding_smoothing = 0

        if USE_NO_GP:
            logger.warning(f"⚠️ Overriding GP with no GP (use this setting with caution)")
            self.padding_smoothing = 0
        # get wrapper around smoothing for distributed / not distributed
        self.smoothing_wrapper = _get_smoothing_wrapper(self)
    
    def cleanup(self):
        # manually clean up
        del self.exp_avg
        del self.exp_avg_sq
        del self.grid
    
    def set_data_and_size(self, warp, size, grid_copy=None):
        ''' change the optimization variables sizes '''
        self.warp = warp
        mode = 'bilinear' if self.n_dims == 2 else 'trilinear'

        # check size to upsample if not already in correct size
        if any([s != size[i] for i, s in enumerate(self.exp_avg.shape[1:-1])]):
            if self.offload:
                self.exp_avg = self.exp_avg.to(self.device)
                self.exp_avg_sq = self.exp_avg_sq.to(self.device)

            # if resetting, we dont need to interpolate at all
            if self.reset:
                logger.info("Resetting optimizer")
                warp_size = [self.batch_size, *size, self.n_dims]
                self.exp_avg = torch.zeros(warp_size, device=self.device if not self.offload else 'cpu')
                self.exp_avg_sq = torch.zeros(warp_size, device=self.device if not self.offload else 'cpu')
            else:
                self.exp_avg = F.interpolate(self.exp_avg.detach().permute(*self.permute_vtoimg), size=size, mode=mode, align_corners=True, 
                                    ).permute(*self.permute_imgtov)
                self.exp_avg_sq = F.interpolate(self.exp_avg_sq.detach().permute(*self.permute_vtoimg), size=size, mode=mode, align_corners=True, 
                                    ).permute(*self.permute_imgtov)
            
            # offload it back to CPU
            if self.offload:
                self.exp_avg = self.exp_avg.to('cpu')
                self.exp_avg_sq = self.exp_avg_sq.to('cpu')

        self.half_resolution = 1.0/(max(warp.shape[1:-1]) - 1)
        self.initialize_grid(size, grid_copy=grid_copy)
        # print(self.warp.shape, warpinv)
    
    def initialize_grid(self, size, grid_copy=None):
        ''' initialize the grid (so that we can use it independent of the grid elsewhere) '''
        if fireants_interpolator.use_ffo:
            self.grid = None
        else:
            if grid_copy is None:
                self.grid = F.affine_grid(self.affine_init, [self.batch_size, 1, *size], align_corners=True).detach()
            else:
                self.grid = grid_copy 

    def zero_grad(self):
        ''' set the gradient to none '''
        self.warp.grad = None
    
    def augment_jacobian(self, u):
        # Multiply u (which represents dL/dphi most likely) with Jacobian indexed by J[..., xyz, ..., phi]
        if fireants_interpolator.use_ffo:
            #TODO: Implement Jacobian computation for fused grid sampler
            logger.warning("Using fused grid sampler, Jacobian is not computed, returning input")
            return u
        else:
            jac = jacobian_fn(self.warp.data + self.grid, normalize=True)  # [B, dims, H, W, [D], dims]
            if self.n_dims == 2:
                ujac = torch.einsum('bxhwp,bhwp->bhwx', jac, u)
            else:
                ujac = torch.einsum('bxhwdp,bhwdp->bhwdx', jac, u)
            return ujac
    
    def step(self):
        ''' check for momentum, and other things '''
        grad = self.warp.grad.data
        if self.multiply_jacobian:
            grad = self.augment_jacobian(grad)
        # add weight decay term
        if self.weight_decay > 0:
            grad.add_(self.warp.data, alpha=self.weight_decay)
        # compute moments
        self.step_t += 1

        # we onload this to GPU
        if self.offload:
            # torch.cuda.synchronize()
            # move to device
            self.exp_avg = self.exp_avg.to(self.device)
            self.exp_avg_sq = self.exp_avg_sq.to(self.device)

        self.exp_avg.mul_(self.beta1).add_(grad, alpha=1-self.beta1)
        self.exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad.conj(), value=1-self.beta2)
        # bias correction
        beta_correction1 = 1 - self.beta1 ** self.step_t
        beta_correction2 = 1 - self.beta2 ** self.step_t

        # adam_update_fused(grad, self.exp_avg, self.exp_avg_sq, beta_correction1, beta_correction2, self.eps)
        adam_update_fused(grad, self.exp_avg, self.exp_avg_sq, beta_correction1, beta_correction2, self.eps)

        # we offload this to CPU
        if self.offload:
            # torch.cuda.synchronize()
            # move to device
            self.exp_avg = self.exp_avg.to('cpu')
            self.exp_avg_sq = self.exp_avg_sq.to('cpu')
            torch.cuda.empty_cache()

        # denom = (self.exp_avg_sq / beta_correction2).sqrt().add_(self.eps)
        # get updated gradient (this will be normalized and passed in)
        # grad = self.exp_avg / beta_correction1 / denom
        # grad.copy_(self.exp_avg).div_(beta_correction1).div_(denom)
        # del denom
        # grad.data.copy_(self.exp_avg / beta_correction1)
        # grad.data.div_(denom)
        # renormalize and update warp
        if self.freeform:
            grad.mul_(-self.lr)
            # self.warp.data.copy_(grad + self.warp.data)
            self.warp.data.add_(grad)
            if self.smoothing_gaussians is not None:
                self.warp.data = self.smoothing_wrapper(self.warp.data, self.smoothing_gaussians, self.padding_smoothing)
        else:
            # This is the diffeomorphic update
            # gradmax = self.eps + grad.reshape(grad.shape[0], -1).abs().max(1).values  # [B,]
            gradmax = self.eps + grad.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
            gradmax = gradmax.reshape(-1, *([1])*(self.n_dims+1))
            if not self.scaledown:  # if scaledown is "True", then we scale down even if the norm is below 1
                gradmax = torch.clamp(gradmax, min=1)
            # print(gradmax.abs().min(), gradmax.abs().max())
            # grad = grad / gradmax * self.half_resolution   # norm is now 0.5r
            grad.div_(gradmax).mul_(self.half_resolution)
            # multiply by learning rate
            grad.mul_(-self.lr)
            # print(grad.abs().max().item(), self.half_resolution, self.warp.shape)
            # compositional update
            # w = grad + F.grid_sample(self.warp.data.permute(*self.permute_vtoimg), self.grid + grad, mode='bilinear', align_corners=True).permute(*self.permute_imgtov)
            # w = grad + 
            ## grad = grad + warp * (x + grad)   # this is the compositional update
            grad.add_(fireants_interpolator.warp_composer(self.warp.data, affine=self.affine_init, v=grad, grid=self.grid, align_corners=True))
            ### WRONG Code below - do not uncomment (think why)
            # fireants_interpolator.warp_composer(self.warp.data, affine=self.affine_init, v=grad, grid=self.grid, align_corners=True, min_coords=None, max_coords=None, output=grad)
            # w = grad
            # smooth result if asked for
            if self.smoothing_gaussians is not None:
                grad = self.smoothing_wrapper(grad, self.smoothing_gaussians, self.padding_smoothing)
            self.warp.data.copy_(grad)
