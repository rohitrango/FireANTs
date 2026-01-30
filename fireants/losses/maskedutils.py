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


''' supports maskedutils for masked functions '''
import torch
from typing import List

POSSIBLE_MASKED_MODES: List[str] = ['mult', 'max']
DEFAULT_MASK_MODE: str = "mult"

def get_mask_function(mask1: torch.Tensor, mask2: torch.Tensor, masked_mode: str):
    if masked_mode == "mult":
        return mask1 * mask2
    if masked_mode == "max":
        return torch.maximum(mask1, mask2)
    raise NotImplementedError(f"{masked_mode} mode is not supported")


def get_tensors_and_mask(image1: torch.Tensor, image2: torch.Tensor, masked_mode: str):
    '''
    The function assumes the last channel is the mask and the rest is the image.
    When there is only one channel, treat the whole tensor as image and use an
    all-ones mask so downstream convolutions never see zero channels.
    '''
    c1 = image1.shape[1]
    c2 = image2.shape[1]
    if c1 <= 1 or c2 <= 1:
        raise ValueError("Expected at least 2 channels in the image, got 1 or less")
    image1, mask1 = torch.split(image1, [c1 - 1, 1], dim=1)
    image2, mask2 = torch.split(image2, [c2 - 1, 1], dim=1)
    mask = get_mask_function(mask1, mask2, masked_mode)
    return image1, image2, mask

def mask_loss_function(loss_tensor: torch.Tensor, mask_tensor: torch.Tensor, reduction: str):
    ''' given the loss function and final mask, compute the loss '''
    if reduction == 'mean':
        # sum over all dims except first (batch), then mean over batch
        weighted_loss = (loss_tensor * mask_tensor)
        # sum over all dims except batch
        sum_dims = tuple(range(1, weighted_loss.dim()))
        loss_sum = weighted_loss.sum(dim=sum_dims)
        mask_sum = mask_tensor.sum(dim=sum_dims)
        batch_mean = (loss_sum / (mask_sum + 1e-8)).mean()
        return batch_mean
    if reduction == "sum":
        return (loss_tensor * mask_tensor).sum()
    raise NotImplementedError(f"reduction type {reduction} is not supported")
