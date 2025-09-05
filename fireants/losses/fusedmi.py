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


from __future__ import annotations

from time import time, sleep
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
import os
import fireants_fused_ops as ffo
from fireants.losses.mi import allgather_mi
import logging
logger = logging.getLogger(__name__)
from fireants.registration.distributed import parallel_state

kernel_type_dict = {
    "gaussian": ffo.KernelType.GAUSSIAN,
    "b-spline": ffo.KernelType.BSPLINE,
    "b_spline": ffo.KernelType.BSPLINE,
    "delta": ffo.KernelType.DELTA,
}

class MI_histogram_kernel(torch.autograd.Function):
    ''' custom op to compute kernel without creating parzen window table '''
    @staticmethod
    def forward(ctx, input_img, target_img, num_bins, kernel_type, minval, maxval, sigma_ratio, approximate_reduction):
        # compute histograms
        pab, pa, pb = ffo.mutual_information_histogram_fwd(input_img, target_img, num_bins, kernel_type, minval, maxval, sigma_ratio, approximate_reduction)
        ctx.num_bins = num_bins
        ctx.kernel_type = kernel_type
        # save
        ctx.save_for_backward(input_img, target_img) 
        ctx.minval = minval
        ctx.maxval = maxval
        ctx.sigma_ratio = sigma_ratio
        return pab, pa, pb
    
    @staticmethod
    def backward(ctx, grad_pab, grad_pa, grad_pb):
        input_img, target_img = ctx.saved_tensors  #
        num_bins = ctx.num_bins
        kernel_type = ctx.kernel_type
        grad_input = None
        grad_target = None
        minval = ctx.minval
        maxval = ctx.maxval
        sigma_ratio = ctx.sigma_ratio

        if input_img.requires_grad:
            grad_input = torch.zeros_like(input_img)
        if target_img.requires_grad:
            grad_target = torch.zeros_like(target_img)
        ffo.mutual_information_histogram_bwd(input_img, target_img, grad_pab, grad_pa, grad_pb, num_bins, grad_input, grad_target, kernel_type, minval, maxval, sigma_ratio)
        # sample_size = input_img.flatten(2).shape[2]
        # if grad_input is not None:
        #     grad_input.mul_(sample_size)
        # if grad_target is not None:
        #     grad_target.mul_(sample_size)
        return grad_input, grad_target, None, None, None, None, None, None


class FusedGlobalMutualInformationLoss(nn.Module):
    """
    Differentiable global mutual information loss via Parzen windowing method.
    Reference:
        https://dspace.mit.edu/handle/1721.1/123142, Section 3.1, equation 3.1-3.5, Algorithm 1
    """
    def __init__(
        self,
        kernel_type: str = "gaussian",
        num_bins: int = 32,
        reduction: str = "mean", 
        normalize_image_if_required: bool = True,
        smooth_nr: float = 1e-7,
        smooth_dr: float = 1e-7,
        sigma_ratio: float = 1.0,
        approximate_reduction: bool = True,
    ) -> None:
        """
        Args:
            kernel_type: {``"gaussian"``, ``"b-spline"``, ``"delta"``}
                - custom implementation
            num_bins: number of bins for intensity
            sigma_ratio: a hyper param for gaussian function
            reduction: {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
            smooth_nr: a small constant added to the numerator to avoid nan.
            smooth_dr: a small constant added to the denominator to avoid nan.
        """
        super().__init__()
        logger.info(f"Initializing FusedGlobalMutualInformationLoss with kernel_type={kernel_type}, num_bins={num_bins}, reduction={reduction}, normalize_image_if_required={normalize_image_if_required}, smooth_nr={smooth_nr}, smooth_dr={smooth_dr}")
        if num_bins <= 0:
            raise ValueError("num_bins must > 0, got {num_bins}")
        self.kernel_type = kernel_type_dict[kernel_type]
        self.num_bins = num_bins
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)
        self.reduction = reduction 
        self.normalize_image_if_required = normalize_image_if_required
        self.warned = False
        self.sigma_ratio = sigma_ratio
        self.approximate_reduction = approximate_reduction


    def get_image_padding(self) -> int:
        return 0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: the shape should be B[NDHW].
            target: the shape should be same as the pred shape.
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        minval = min(pred.min(), target.min()).detach().item()
        maxval = max(pred.max(), target.max()).detach().item()
        normalize = False
        if maxval > 1:
            if not self.warned:
                logger.warn("Image values are expected to be in the range [0, 1] - normalizing the images")
                self.warned = True
            normalize = True
        if minval < 0:
            if not self.warned:
                logger.warn("Image values are expected to be in the range [0, 1] - normalizing the images")
                self.warned = True
            normalize = True
        # check if we should normalize
        if normalize: 
            if self.normalize_image_if_required:
                # pred = (pred - minval) / (maxval - minval)
                # target = (target - minval) / (maxval - minval)
                pass  # we will pass these values in the kernel
            else:
                raise ValueError("Image values are expected to be in the range [0, 1] - please set normalize_image_if_required to True or scale the images to [0, 1] before feeding them to the loss")
        else:
            minval, maxval = 0.0, 1.0 # no need to normalize

        # check if shapes are same
        if target.shape != pred.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})")
        
        # flip order
        if not pred.requires_grad and target.requires_grad:
            pred, target = target, pred
        
        # get histograms
        pab, pa, pb = MI_histogram_kernel.apply(pred, target, self.num_bins, self.kernel_type, minval, maxval, self.sigma_ratio, self.approximate_reduction)

        if parallel_state.is_initialized() and parallel_state.get_grid_parallel_size() > 1:
            pab, pa, pb = allgather_mi.apply(pab, pa, pb, pred.flatten(2).shape[2])
            # divide by total number of samples (this is not exact but approximate)
            papb = torch.bmm(pa.permute(0, 2, 1), pb.to(pa))
        else:
            papb = torch.bmm(pa.permute(0, 2, 1), pb.to(pa))  # (batch, num_bins, num_bins)
        
        mi = torch.sum(
            pab * torch.log((pab + self.smooth_nr) / (papb + self.smooth_dr) + self.smooth_dr), dim=(-1, -2)
        )  # (batch)

        if self.reduction == 'sum':
            return torch.sum(mi).neg()  # sum over the batch and channel ndims
        if self.reduction == 'none':
            return mi.neg()
        if self.reduction == 'mean':
            return torch.mean(mi).neg()  # average over the batch and channel ndims

        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')


if __name__ == '__main__':
    N = 256
    img1 = torch.rand(1, 1, N, N, N).cuda()
    img2 = torch.rand(1, 1, N, N, N).cuda()
    #loss = FusedGlobalMutualInformationLoss('b-spline').cuda()
    loss = FusedGlobalMutualInformationLoss('gaussian', sigma_ratio=2.0).cuda()
    total = 0
    a = time()
    for i in range(10):
        out = loss(img1, img2)
        total += out.item()
    print(time() - a)
