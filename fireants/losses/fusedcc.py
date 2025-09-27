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
Fused Cross correlation
'''
from time import time, sleep
import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
from torch.nn import functional as F
import sys
from typing import Union, Tuple, List, Optional, Dict, Any, Callable
from fireants.types import ItemOrList
import fireants_fused_ops as ffo
from fireants.losses.cc import LocalNormalizedCrossCorrelationLoss, gaussian_1d
from fireants.tests.cc_mem_test import fast_lncc
import itertools
import logging
logger = logging.getLogger(__name__)

MAX_INT32_NUMEL = 2**31 - 1

reduction_table = {
    'none': ffo.Reduction.NONE,
    'sum': ffo.Reduction.SUM,
    'mean': ffo.Reduction.MEAN
}

class FusedNCC3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_img, target_img, kernel_size, nr, dr, reduction, use_ants_gradient, use_separable):
        reduction = reduction_table[reduction.lower()]
        B, C, H, W, D = input_img.shape
        assert input_img.is_contiguous() and target_img.is_contiguous(), "input_img and target_img must be contiguous"
        interm = torch.zeros(B, 5 * C, H, W, D, device=input_img.device, dtype=input_img.dtype)
        # interm[:, :C, :, :, :] = input_img
        # interm[:, C:2*C, :, :, :] = target_img
        # interm[:, 2*C:3*C, :, :, :] = input_img * input_img
        # interm[:, 3*C:4*C, :, :, :] = target_img * target_img
        # interm[:, 4*C:, :, :, :] = input_img * target_img  # [B, 5C, H, W, D]
        ffo.create_intermediates(input_img, target_img, interm)
        # compute kernel 
        kernel_vol = kernel_size ** 3

        if use_separable:
            if interm.numel() >= MAX_INT32_NUMEL:
                # numel is too large for group convolution, fallback to singular
                filt1 = torch.ones(1, 1, kernel_size, 1, 1, device=input_img.device, dtype=input_img.dtype) / kernel_size
                filt2 = torch.ones(1, 1, 1, kernel_size, 1, device=input_img.device, dtype=input_img.dtype) / kernel_size
                filt3 = torch.ones(1, 1, 1, 1, kernel_size, device=input_img.device, dtype=input_img.dtype) / kernel_size
                for c in range(5*C):
                    interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt1, padding='same', stride=1, groups=1)
                    interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt2, padding='same', stride=1, groups=1)
                    interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt3, padding='same', stride=1, groups=1)
            else:
                filt1 = torch.ones(5*C, 1, kernel_size, 1, 1, device=input_img.device, dtype=input_img.dtype) / kernel_size
                filt2 = torch.ones(5*C, 1, 1, kernel_size, 1, device=input_img.device, dtype=input_img.dtype) / kernel_size
                filt3 = torch.ones(5*C, 1, 1, 1, kernel_size, device=input_img.device, dtype=input_img.dtype) / kernel_size
                interm = F.conv3d(interm, filt1, padding='same', stride=1, groups=interm.shape[1])
                interm = F.conv3d(interm, filt2, padding='same', stride=1, groups=interm.shape[1])
                interm = F.conv3d(interm, filt3, padding='same', stride=1, groups=interm.shape[1])
        else:
            if interm.numel() >= MAX_INT32_NUMEL:
                # numel is too large for group convolution, fallback to singular
                avg_filt = torch.ones(1, 1, kernel_size, kernel_size, kernel_size, device=input_img.device, dtype=input_img.dtype) / kernel_vol
                padding = (kernel_size - 1) // 2
                for c in range(5*C):
                    interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], avg_filt, padding=padding, stride=1, groups=1)
            else:
                avg_filt = torch.ones(5*C, 1, kernel_size, kernel_size, kernel_size, device=input_img.device, dtype=input_img.dtype) / kernel_vol
                padding = (kernel_size - 1) // 2
                interm = F.conv3d(interm, avg_filt, padding=padding, stride=1, groups=interm.shape[1])

        out = ffo.cc3d_fwd_interm_v1(interm, int(kernel_vol), reduction, nr, dr)
        ctx.save_for_backward(interm, input_img, target_img, out)
        ctx.kernel_size = kernel_size
        ctx.nr = nr
        ctx.dr = dr 
        ctx.reduction = reduction
        ctx.use_ants_gradient = use_ants_gradient
        ctx.use_separable = use_separable
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # retrieve saved tensors
        interm, input_img, target_img, out = ctx.saved_tensors
        kernel_size, nr, dr, reduction, use_ants_gradient, use_separable = ctx.kernel_size, ctx.nr, ctx.dr, ctx.reduction, ctx.use_ants_gradient, ctx.use_separable
        B, C, H, W, D = input_img.shape

        # add reduction
        # if reduction == ffo.Reduction.MEAN:
        #     grad_output /= (H * W * D)

        input_too_large = interm.numel() >= MAX_INT32_NUMEL
        inp_size = 5*C if not input_too_large else 1

        # initialize filters
        if use_separable:
            filt1 = torch.ones(inp_size, 1, kernel_size, 1, 1, device=input_img.device, dtype=input_img.dtype) / kernel_size
            filt2 = torch.ones(inp_size, 1, 1, kernel_size, 1, device=input_img.device, dtype=input_img.dtype) / kernel_size
            filt3 = torch.ones(inp_size, 1, 1, 1, kernel_size, device=input_img.device, dtype=input_img.dtype) / kernel_size
            pad1, pad2, pad3 = ((kernel_size - 1) // 2, 0, 0), (0, 0, (kernel_size - 1) // 2), (0, 0, (kernel_size - 1) // 2)
        else:
            avg_filt = torch.ones(inp_size, 1, kernel_size, kernel_size, kernel_size, device=input_img.device, dtype=input_img.dtype) / kernel_vol
            padding = (kernel_size - 1) // 2

        # initialize gradients
        grad_input_img = None
        grad_target_img = None
        # if backward is called, first tensor will always require grad
        # second may or may not require grad
        if input_img.requires_grad:
            grad_input_img = torch.zeros(B, C, H, W, D, device=input_img.device, dtype=input_img.dtype)
        if target_img.requires_grad:
            grad_target_img = torch.zeros(B, C, H, W, D, device=input_img.device, dtype=input_img.dtype)
        # compute correct gradients (maybe slightly slower)
        ffo.cc3d_bwd_modify_interm_v1(interm, input_img, target_img, grad_output, grad_input_img, grad_target_img, kernel_size, nr, dr, reduction)
        # convolve with average filter depending on whether grad_target_img is None
        # if using ants_gradient, the convolution is skipped (ignore interactions from other neighboring pixels)
        if not use_ants_gradient:
            padding = (kernel_size - 1) // 2
            if grad_target_img is not None:
                # convolve with "one" filter depending on whether separable or not
                if use_separable:
                    if input_too_large:
                        for c in range(5*C):
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt1, padding='same', stride=1, groups=1)
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt2, padding='same', stride=1, groups=1)
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt3, padding='same', stride=1, groups=1)
                    else:
                        interm = F.conv3d(interm, filt1, padding='same', stride=1, groups=interm.shape[1])
                        interm = F.conv3d(interm, filt2, padding='same', stride=1, groups=interm.shape[1])
                        interm = F.conv3d(interm, filt3, padding='same', stride=1, groups=interm.shape[1])
                else:
                    if input_too_large:
                        for c in range(5*C):
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], avg_filt, padding=padding, stride=1, groups=1)
                    else:
                        interm = F.conv3d(interm, avg_filt, padding=padding, stride=1, groups=interm.shape[1])
            else:
                if use_separable:
                    if input_too_large:
                        for c in range(3*C):
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt1, padding='same', stride=1, groups=1)
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt2, padding='same', stride=1, groups=1)
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt3, padding='same', stride=1, groups=1)
                    else:
                        filt1 = filt1[:3*C]
                        filt2 = filt2[:3*C]
                        filt3 = filt3[:3*C]
                        interm[:, :3*C, :, :, :] = F.conv3d(interm[:, :3*C, :, :, :], filt1, padding='same', stride=1, groups=3*C)
                        interm[:, :3*C, :, :, :] = F.conv3d(interm[:, :3*C, :, :, :], filt2, padding='same', stride=1, groups=3*C)
                        interm[:, :3*C, :, :, :] = F.conv3d(interm[:, :3*C, :, :, :], filt3, padding='same', stride=1, groups=3*C)
                else:
                    if input_too_large:
                        for c in range(3*C):
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], avg_filt, padding=padding, stride=1, groups=1)
                    else:
                        avg_filt = avg_filt[:3*C]
                        interm[:, :3*C, :, :, :] = F.conv3d(interm[:, :3*C, :, :, :], avg_filt, padding=padding, stride=1, groups=3*C)
        # solve for grad_input_img and grad_target_img
        ffo.cc3d_bwd_compute_grads(interm, input_img, target_img, grad_input_img, grad_target_img)
        return grad_input_img, grad_target_img, None, None, None, None, None, None


class FusedNCC3dGaussian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_img, target_img, kernel_size, sigma, nr, dr, reduction, use_ants_gradient, use_separable):
        reduction = reduction_table[reduction.lower()]
        B, C, H, W, D = input_img.shape
        assert input_img.is_contiguous() and target_img.is_contiguous(), "input_img and target_img must be contiguous"
        interm = torch.zeros(B, 5 * C, H, W, D, device=input_img.device, dtype=input_img.dtype)
        ffo.create_intermediates(input_img, target_img, interm)
        # compute kernel 
        kernel_vol = kernel_size ** 3
        # get truncated to match kernel size
        truncated = (kernel_size//2 - 0.5)/sigma
        gauss_filt = gaussian_1d(torch.tensor(sigma, device=input_img.device, dtype=input_img.dtype), truncated=truncated, approx="sampled")  # kernel size
        assert gauss_filt.numel() == kernel_size, "kernel size does not match"

        if use_separable:
            if interm.numel() >= MAX_INT32_NUMEL:
                # numel is too large for group convolution, fallback to singular
                filt1 = gauss_filt[None, None, :, None, None]
                filt2 = gauss_filt[None, None, None, :, None]
                filt3 = gauss_filt[None, None, None, None, :]
                for c in range(5*C):
                    interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt1, padding='same', stride=1, groups=1)
                    interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt2, padding='same', stride=1, groups=1)
                    interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt3, padding='same', stride=1, groups=1)
            else:
                filt1 = gauss_filt[None, None, :, None, None].expand(5*C, -1, -1, -1, -1)
                filt2 = gauss_filt[None, None, None, :, None].expand(5*C, -1, -1, -1, -1)
                filt3 = gauss_filt[None, None, None, None, :].expand(5*C, -1, -1, -1, -1)
                interm = F.conv3d(interm, filt1, padding='same', stride=1, groups=interm.shape[1])
                interm = F.conv3d(interm, filt2, padding='same', stride=1, groups=interm.shape[1])
                interm = F.conv3d(interm, filt3, padding='same', stride=1, groups=interm.shape[1])
        else:
            if interm.numel() >= MAX_INT32_NUMEL:
                # numel is too large for group convolution, fallback to singular
                avg_filt = gauss_filt[None, None, :, None, None] * gauss_filt[None, None, None, :, None] * gauss_filt[None, None, None, None, :]
                padding = (kernel_size - 1) // 2
                for c in range(5*C):
                    interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], avg_filt, padding=padding, stride=1, groups=1)
            else:
                avg_filt = gauss_filt[None, None, :, None, None] * gauss_filt[None, None, None, :, None] * gauss_filt[None, None, None, None, :]
                avg_filt = avg_filt.expand(5*C, -1, -1, -1, -1)
                padding = (kernel_size - 1) // 2
                interm = F.conv3d(interm, avg_filt, padding=padding, stride=1, groups=interm.shape[1])

        out = ffo.cc3d_fwd_interm_v1(interm, int(kernel_vol), reduction, nr, dr)
        ctx.save_for_backward(interm, input_img, target_img, out)
        ctx.kernel_size = kernel_size
        ctx.sigma = sigma
        ctx.truncated = truncated
        ctx.nr = nr
        ctx.dr = dr 
        ctx.reduction = reduction
        ctx.use_ants_gradient = use_ants_gradient
        ctx.use_separable = use_separable
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        # retrieve saved tensors
        interm, input_img, target_img, out = ctx.saved_tensors
        kernel_size, sigma, truncated, nr, dr, reduction, use_ants_gradient, use_separable = ctx.kernel_size, ctx.sigma, ctx.truncated, ctx.nr, ctx.dr, ctx.reduction, ctx.use_ants_gradient, ctx.use_separable
        B, C, H, W, D = input_img.shape

        input_too_large = interm.numel() >= MAX_INT32_NUMEL
        inp_size = 5*C if not input_too_large else 1
    
        gauss_filt = gaussian_1d(torch.tensor(sigma, device=input_img.device, dtype=input_img.dtype), truncated=truncated, approx="sampled")  # kernel size

        if reduction == ffo.Reduction.MEAN:
            grad_output = grad_output / (B * C * H * W * D)

        # initialize filters
        if use_separable:
            filt1 = gauss_filt[None, None, :, None, None].expand(inp_size, -1, -1, -1, -1)
            filt2 = gauss_filt[None, None, None, :, None].expand(inp_size, -1, -1, -1, -1)
            filt3 = gauss_filt[None, None, None, None, :].expand(inp_size, -1, -1, -1, -1)
            pad1, pad2, pad3 = ((kernel_size - 1) // 2, 0, 0), (0, 0, (kernel_size - 1) // 2), (0, 0, (kernel_size - 1) // 2)
        else:
            avg_filt = gauss_filt[None, None, :, None, None] * gauss_filt[None, None, None, :, None] * gauss_filt[None, None, None, None, :]
            avg_filt = avg_filt.expand(inp_size, -1, -1, -1, -1)
            padding = (kernel_size - 1) // 2

        # initialize gradients
        grad_input_img = None
        grad_target_img = None
        # if backward is called, first tensor will always require grad
        # second may or may not require grad
        if input_img.requires_grad:
            grad_input_img = torch.zeros(B, C, H, W, D, device=input_img.device, dtype=input_img.dtype)
        if target_img.requires_grad:
            grad_target_img = torch.zeros(B, C, H, W, D, device=input_img.device, dtype=input_img.dtype)
        # compute correct gradients (maybe slightly slower)
        ffo.cc3d_bwd_modify_interm_v1(interm, input_img, target_img, grad_output, grad_input_img, grad_target_img, kernel_size, nr, dr, reduction)
        # convolve with average filter depending on whether grad_target_img is None
        # if using ants_gradient, the convolution is skipped (ignore interactions from other neighboring pixels)
        if not use_ants_gradient:
            padding = (kernel_size - 1) // 2
            if grad_target_img is not None:
                # convolve with "one" filter depending on whether separable or not
                if use_separable:
                    if input_too_large:
                        for c in range(5*C):
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt1, padding='same', stride=1, groups=1)
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt2, padding='same', stride=1, groups=1)
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt3, padding='same', stride=1, groups=1)
                    else:
                        interm = F.conv3d(interm, filt1, padding='same', stride=1, groups=interm.shape[1])
                        interm = F.conv3d(interm, filt2, padding='same', stride=1, groups=interm.shape[1])
                        interm = F.conv3d(interm, filt3, padding='same', stride=1, groups=interm.shape[1])
                else:
                    if input_too_large:
                        for c in range(5*C):
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], avg_filt, padding=padding, stride=1, groups=1)
                    else:
                        interm = F.conv3d(interm, avg_filt, padding=padding, stride=1, groups=interm.shape[1])
            else:
                if use_separable:
                    if input_too_large:
                        for c in range(3*C):
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt1, padding='same', stride=1, groups=1)
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt2, padding='same', stride=1, groups=1)
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], filt3, padding='same', stride=1, groups=1)
                    else:
                        filt1 = filt1[:3*C]
                        filt2 = filt2[:3*C]
                        filt3 = filt3[:3*C]
                        interm[:, :3*C, :, :, :] = F.conv3d(interm[:, :3*C, :, :, :], filt1, padding='same', stride=1, groups=3*C)
                        interm[:, :3*C, :, :, :] = F.conv3d(interm[:, :3*C, :, :, :], filt2, padding='same', stride=1, groups=3*C)
                        interm[:, :3*C, :, :, :] = F.conv3d(interm[:, :3*C, :, :, :], filt3, padding='same', stride=1, groups=3*C)
                else:
                    if input_too_large:
                        for c in range(3*C):
                            interm[:, c:c+1] = F.conv3d(interm[:, c:c+1], avg_filt, padding=padding, stride=1, groups=1)
                    else:
                        avg_filt = avg_filt[:3*C]
                        interm[:, :3*C, :, :, :] = F.conv3d(interm[:, :3*C, :, :, :], avg_filt, padding=padding, stride=1, groups=3*C)
        # solve for grad_input_img and grad_target_img
        ffo.cc3d_bwd_compute_grads(interm, input_img, target_img, grad_input_img, grad_target_img)
        return grad_input_img, grad_target_img, None, None, None, None, None, None, None


class FusedLocalNormalizedCrossCorrelationLoss(nn.Module):
    """
    Local squared zero-normalized cross-correlation.
    The loss is based on a moving kernel/window over the y_true/y_pred,
    within the window the square of zncc is calculated.
    The kernel can be a rectangular / triangular / gaussian window.
    The final loss is the averaged loss over all windows.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        kernel_size: Union[int, List[int]] = 3,
        reduction: str = "mean",
        smooth_nr: float = 0,
        smooth_dr: float = 1e-5,
        use_ants_gradient: bool = True,
        use_separable: bool = True,
        kernel_type: str = "rectangular",
        sigma: float = 1.5,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, {``1``, ``2``, ``3``}. Defaults to 3.
            kernel_size: kernel spatial size, must be odd.
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid nan.
            smooth_dr: a small constant added to the denominator to avoid nan.
        """
        super().__init__()
        self.ndim = spatial_dims
        if self.ndim not in {1, 2, 3}:
            raise ValueError(f"Unsupported ndim: {self.ndim}-d, only 1-d, 2-d, and 3-d inputs are supported")
        self.reduction = reduction
        # gaussian kernel parameters
        self.kernel_type = kernel_type
        self.sigma = sigma

        # keep list if kernel_size is list, else empty list
        self.kernel_size_list = kernel_size if isinstance(kernel_size, (list, tuple)) else None
        self.kernel_size = kernel_size[0] if isinstance(kernel_size, (list, tuple)) else kernel_size
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")

        # _kernel = look_up_option(kernel_type, kernel_dict)
        kernel_vol = self.kernel_size ** self.ndim 
        self.smooth_nr = float(smooth_nr) * kernel_vol
        self.smooth_dr = float(smooth_dr) * kernel_vol
        self.use_ants_gradient = use_ants_gradient
        self.use_separable = use_separable
    
    def set_scales(self, scales):
        ''' function is called at initialization of abstract registration '''
        self.scales = scales
        if self.kernel_size_list:
            assert len(self.kernel_size_list) == len(self.scales), "kernel_size must be a list of the same length as scales"
    
    def set_iterations(self, iterations):
        ''' function is called at initialization of abstract registration '''
        self.iterations = iterations
    
    def set_current_scale_and_iterations(self, scale, iters):
        if self.kernel_size_list:
            idx = self.scales.index(scale)
            self.kernel_size = self.kernel_size_list[idx]
    
    def get_image_padding(self) -> int:
        return (self.kernel_size - 1) // 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if pred.ndim - 2 != self.ndim:
            raise ValueError(f"expecting pred with {self.ndim} spatial dimensions, got pred of shape {pred.shape}")
        if target.shape != pred.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})")
        # check if both pred and target require grad
        pgrad, tgrad = pred.requires_grad, target.requires_grad
        # if both or neither require grad, dont shuffle the order 
        # the first tensor will always require grad (or neither does)
        if tgrad and not pgrad:
            pred, target = target, pred
        
        pred, target = pred.contiguous(), target.contiguous()
        
        if self.kernel_type == "gaussian":
            return -FusedNCC3dGaussian.apply(pred, target, self.kernel_size, self.sigma, self.smooth_nr, self.smooth_dr, self.reduction, self.use_ants_gradient, self.use_separable)
        elif self.kernel_type == "rectangular":
            return -FusedNCC3d.apply(pred, target, self.kernel_size, self.smooth_nr, self.smooth_dr, self.reduction, self.use_ants_gradient, self.use_separable)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
