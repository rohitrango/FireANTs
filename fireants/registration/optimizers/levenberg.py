# Copyright (c) 2026 Rohit Jena. All rights reserved.
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

''' class for Levenberg-Marquardt for compositive warps '''
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
from fireants.registration.optimizers.adam import _get_smoothing_wrapper
from typing import Optional
import numpy as np
import os
import logging
logger = logging.getLogger(__name__)
from functools import partial

USE_NO_GP = bool(int(os.environ.get('USE_NO_GP', '0').lower()))

@torch.no_grad()
def tile_size_one_torch_update(grad: torch.Tensor, lmbda: float) -> torch.Tensor:
    ''' update the gradient for a tile size of 1 '''
    scaling = grad.norm(p=2, dim=-1, keepdim=True).pow_(2).add_(lmbda)   # grad**2 + lmbda
    grad.div_(scaling)
    return grad

@torch.no_grad()
def tile_size_n_torch_update(grad: torch.Tensor, lmbda: float, tile_sizes: tuple[int, ...]= (2, 2, 2)) -> torch.Tensor:
    ''' update the gradient for a tile size of n '''
    dims = grad.shape[-1]
    jac = grad[..., None] * grad[..., None, :]  # [..., dims, dims]
    if dims == 2:
        B, H, W, _, _ = jac.shape
        jac = jac.reshape(B, H, W, -1).permute(0, 3, 1, 2)
    elif dims == 3:
        B, H, W, D, _, _ = jac.shape
        jac = jac.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3)
    else:
        raise ValueError(f"Invalid dimension: {dims}")
    # compute filters
    ones = [torch.ones(s, device=grad.device) for s in tile_sizes]
    jac = separable_filtering(jac, ones) / np.prod(tile_sizes)
    # reshape back
    if dims == 2:
        jac = jac.permute(0, 2, 3, 1).reshape(B, H, W, 2, 2)
    elif dims == 3:
        jac = jac.permute(0, 2, 3, 4, 1).reshape(B, H, W, D, 3, 3)
    else:
        raise ValueError(f"Invalid dimension: {dims}")
    # add identity matrix
    jac.add_(torch.eye(dims, device=grad.device).mul_(lmbda))
    # invert
    jac = torch.linalg.inv(jac)
    grad = torch.einsum('...ij,...j->...i', jac, grad)
    return grad

class WarpLevenbergMarquardt:
    '''
    shape of warp = [B, H, W, [D], dims]
    '''
    def __init__(self, warp, lr,
                 # parameters for lambda
                 lambda_init=1e-2,
                 lambda_increase_factor=1.5,
                 lambda_decrease_factor=0.975,
                 tile_size=1,
                 # other parameters
                 weight_decay=0, eps=1e-8,
                 scaledown=False, multiply_jacobian=False,
                 smoothing_gaussians=None,
                 grad_gaussians=None,
                 freeform=False,
                 offload=False,   # try offloading to CPU
                 reset=False,
                 # rejection parameters
                 rejection: bool = False,
                 loss_tolerance_ratio: float = 1.0,
                 lambda_max: Optional[float] = 1.0,
                # distributed params
                rank: int = 0,
                dim_to_shard: int = 0,
                dtype: torch.dtype = torch.float32):
        # init
        self.dtype = dtype
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
        # Levenberg-Marquardt parameters
        self.lambda_init = lambda_init
        # loss history for lambda adjustment and rejection
        self.prev_loss = None
        self.prev_prev_loss = None
        self.lambda_increase_factor = lambda_increase_factor
        self.lambda_decrease_factor = lambda_decrease_factor
        # rejection parameters
        self.rejection = rejection
        self.loss_tolerance_ratio = loss_tolerance_ratio
        self.lambda_max = lambda_max
        self.tile_size = tile_size if isinstance(tile_size, (float, tuple)) else [tile_size] * self.n_dims
        assert len(self.tile_size) == self.n_dims, "tile_size must be a tuple of length n_dims"

        assert self.lambda_increase_factor > 1.0, "lambda_increase_factor must be greater than 1.0"
        assert self.lambda_decrease_factor < 1.0, "lambda_decrease_factor must be less than 1.0"
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

        # update function for tile sizes
        if np.prod(self.tile_size) == 1:
            self.update_fn = tile_size_one_torch_update
        else:
            self.update_fn = partial(tile_size_n_torch_update, tile_sizes=self.tile_size)

    def cleanup(self):
        # manually clean up
        del self.grid

    def set_data_and_size(self, warp, size, grid_copy=None):
        ''' change the optimization variables sizes '''
        self.warp = warp
        # check size to upsample if not already in correct size
        if any([s != size[i] for i, s in enumerate(self.exp_avg.shape[1:-1])]):
            # if resetting, we dont need to interpolate at all
            if self.reset:
                logger.info("nothing to reset in optimizer")
        # set new half resolution
        self.half_resolution = 1.0/(max(warp.shape[1:-1]) - 1)
        self.initialize_grid(size, grid_copy=grid_copy)
        # reset loss history for new scale
        self.prev_loss = None
        self.prev_prev_loss = None

    def initialize_grid(self, size, grid_copy=None):
        ''' initialize the grid (so that we can use it independent of the grid elsewhere) '''
        if fireants_interpolator.use_ffo:
            self.grid = None
        else:
            if grid_copy is None:
                self.grid = F.affine_grid(self.affine_init, [self.batch_size, 1, *size], align_corners=True).detach()
            else:
                self.grid = grid_copy

    @property
    def requires_closure(self):
        return self.rejection

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

    def compute_lambda_init_if_auto(self, grad):
        ''' compute the lambda_init if it is auto '''
        if isinstance(self.lambda_init, (int, float)):
            return
        algo = str(self.lambda_init).lower()
        if algo == 'auto':
            self.lambda_init = (torch.linalg.norm(grad, ord=2, dim=-1, keepdim=False).pow_(2).mean()).item()
        else:
            raise ValueError(f"lambda_init must be a float or 'auto', got {algo}")
        logger.info(f"Computed lambda_init as {self.lambda_init} using algorithm '{algo}'")

    def _apply_grad_update(self):
        ''' Apply the gradient from self.warp.grad to self.warp.data using current lambda. '''
        grad = self.warp.grad.data
        if self.multiply_jacobian:
            grad = self.augment_jacobian(grad)
        # add weight decay term
        if self.weight_decay > 0:
            grad.add_(self.warp.data, alpha=self.weight_decay)
        # compute moments
        self.step_t += 1

        # compute damped hessian
        self.compute_lambda_init_if_auto(grad)

        # update the gradient
        grad = self.update_fn(grad, self.lambda_init)

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

    def _evaluate_with_rejection(self, closure, depth: int = 0, max_depth: int = 10):
        '''Evaluate closure, apply update, check post-update loss, and reject+reset if too bad.

        Steps:
          1. Call closure() to compute pre-update loss and gradients.
          2. Save current warp state.
          3. Apply gradient update via _apply_grad_update().
          4. Call closure() to measure post-update loss.
          5. Rejection condition: (post_loss - prev_loss) > loss_tolerance_ratio * |prev_prev_loss - prev_loss|
             If rejected: restore warp, decrease lambda, retry (up to max_depth times).
          6. On acceptance: update prev_loss/prev_prev_loss and return post-update loss.
        '''
        # step 1: compute gradients for the current warp
        closure()
        # step 2: save warp before update
        warp_saved = self.warp.data.clone()
        # step 3: apply the gradient update
        self._apply_grad_update()
        # step 4: evaluate post-update loss
        loss_after = closure()
        current_loss = loss_after.item()

        # step 5: check rejection criterion
        if (self.prev_loss is not None and self.prev_prev_loss is not None
                and depth < max_depth):
            delta = current_loss - self.prev_loss
            threshold = self.loss_tolerance_ratio * abs(self.prev_prev_loss - self.prev_loss)
            if delta > threshold:
                # reject: restore warp, increase lambda (more damping), cap, and retry
                self.warp.data.copy_(warp_saved)
                self.lambda_init *= self.lambda_increase_factor
                if self.lambda_max is not None:
                    self.lambda_init = min(self.lambda_init, self.lambda_max)
                return self._evaluate_with_rejection(closure, depth + 1, max_depth)

        # step 6: accepted — reward with smaller lambda, update loss history
        self.lambda_init *= self.lambda_decrease_factor
        self.prev_prev_loss = self.prev_loss
        self.prev_loss = current_loss
        return loss_after

    def step(self, loss_or_closure=None):
        ''' Perform a Levenberg-Marquardt update step.

        Accepts either a loss tensor (non-closure mode) or a callable closure.
        When rejection=True and a closure is provided, uses the rejection criterion:
        the update is applied, post-update loss is evaluated, and the warp is reset
        if the loss degraded beyond the tolerance.
        '''
        if self.rejection and callable(loss_or_closure):
            # _evaluate_with_rejection handles closure calls, update, and warp restore
            return self._evaluate_with_rejection(loss_or_closure)

        # non-rejection mode: standard lambda adjustment then apply update
        loss = loss_or_closure
        if self.prev_loss is not None:
            if loss.item() > self.prev_loss:   # loss increased, increase lambda
                self.lambda_init *= self.lambda_increase_factor
            else:
                self.lambda_init *= self.lambda_decrease_factor
        self.prev_loss = loss.item()
        self._apply_grad_update()
        return loss
