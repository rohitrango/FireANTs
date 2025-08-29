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
Implementation of the pytorch grid sample interpolator (combining affine and grid_sample)
'''
import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

def torch_grid_sampler_2d(
    input: torch.Tensor,
    affine: Optional[torch.Tensor] = None,
    grid: Optional[torch.Tensor] = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
    out_shape: tuple = None,
    is_displacement: bool = True
) -> torch.Tensor:
    """
    Baseline implementation of 3D grid sampler that handles:
    1. Affine-only transformation (grid=None)
    2. Full warp field (grid provided, is_displacement=False)
    3. Displacement field (grid provided, is_displacement=True)
    
    Args:
        input: Input tensor of shape [B, C, Y, X]
        affine: Affine matrix of shape [B, 2, 3]
        grid: Optional grid tensor of shape [B, Y, X, 2]
        mode: Interpolation mode ("bilinear", "nearest", "bicubic")
        padding_mode: Padding mode ("zeros", "border", "reflection")
        align_corners: Whether to align corners
        out_shape: Output shape tuple (Y, X)
        is_displacement: Whether grid is a displacement field
    
    Returns:
        Sampled tensor of shape [B, C, Y, X]
    """
    B, C, Y, X = input.shape
    
    # Case 1: Affine-only transformation
    if grid is None:
        if affine is None:
            raise ValueError("Either grid or affine must be provided")
        grid = F.affine_grid(affine, (B, C, *out_shape[-2:]), align_corners=align_corners)
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    # Case 2: Full warp field
    if not is_displacement:
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    # Case 3: Displacement field
    out_shape = grid.shape[1:-1]
    # Create identity grid if no affine provided
    if affine is None:
        affine = torch.eye(2, 3, device=input.device)[None].expand(B, -1, -1)
    # Create base grid
    base_grid = F.affine_grid(affine, (B, C, *out_shape[-2:]), align_corners=align_corners)
    # Add displacement
    base_grid = base_grid + grid
    return F.grid_sample(input, base_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

def torch_grid_sampler_3d(
    input: torch.Tensor,
    affine: Optional[torch.Tensor] = None,
    grid: Optional[torch.Tensor] = None,
    grid_affine: Optional[torch.Tensor] = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
    out_shape: tuple = None,
    is_displacement: bool = True,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Baseline implementation of 3D grid sampler that handles:
    1. Affine-only transformation (grid=None)
    2. Full warp field (grid provided, is_displacement=False)
    3. Displacement field (grid provided, is_displacement=True)
    
    Args:
        input: Input tensor of shape [B, C, Z, Y, X]
        affine: Affine matrix of shape [B, 3, 4]
        grid: Optional grid tensor of shape [B, Z, Y, X, 3]
        mode: Interpolation mode ("bilinear", "nearest", "bicubic")
        padding_mode: Padding mode ("zeros", "border", "reflection")
        align_corners: Whether to align corners
        out_shape: Output shape tuple (Z, Y, X)
        is_displacement: Whether grid is a displacement field
    
    Returns:
        Sampled tensor of shape [B, C, Z, Y, X]
    """
    B, C, Z, Y, X = input.shape


    # Case 1: Affine-only transformation
    if grid is None:
        if affine is None:
            raise ValueError("Either grid or affine must be provided")
        grid = F.affine_grid(affine, (B, C, *out_shape[-3:]), align_corners=align_corners)
        ret = F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        if output is not None:
            output.add_(ret)
            return output
        return ret
    
    # see if grid affine
    if grid_affine is not None:
        grid = torch.einsum('bij,b...j->b...i', grid_affine, grid)

    # Case 2: Full warp field
    if not is_displacement:
        ret = F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        if output is not None:
            output.add_(ret)
            return output
        return ret
    # Case 3: Displacement field
    out_shape = grid.shape[1:-1]
    # Create identity grid if no affine provided
    if affine is None:
        affine = torch.eye(3, 4, device=input.device)[None].expand(B, -1, -1)
    # Create base grid
    base_grid = F.affine_grid(affine, (B, C, *out_shape[-3:]), align_corners=align_corners)
    # Add displacement
    base_grid = base_grid + grid
    ret = F.grid_sample(input, base_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    if output is not None:
        output.add_(ret)
        return output
    return ret


def torch_warp_composer_2d(
    u: torch.Tensor,
    affine: Optional[torch.Tensor] = None,
    v: torch.Tensor = None,
    align_corners: bool = True,
    grid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Baseline implementation of 3D grid sampler that handles:
    warp = u \circ (Ax + v)
    
    Args:
        input: Input tensor of shape [B, Yi, Xi, 2]
        affine: Affine matrix of shape [B, 2, 3]
        grid: Optional grid tensor of shape [B, Y, X, 2]
        mode: Interpolation mode ("bilinear", "nearest", "bicubic")
        padding_mode: Padding mode ("zeros", "border", "reflection")
        align_corners: Whether to align corners
        out_shape: Output shape tuple (Y, X)
        is_displacement: Whether grid is a displacement field
    
    Returns:
        Sampled tensor of shape [B, C, Y, X]
    """
    mode = "bilinear"
    padding_mode = "zeros"
    B = v.shape[0]
    # if grid is not provided, create it from affine
    if grid is None:
        if affine is None:
            affine = torch.eye(2, 3, device=u.device)[None].expand(B, -1, -1)
        grid = F.affine_grid(affine, [B, 1] + list(v.shape[1:-1]), align_corners=align_corners)
    return F.grid_sample(u.permute(0, 3, 1, 2), grid + v, mode=mode, padding_mode=padding_mode, align_corners=align_corners).permute(0, 2, 3, 1)

def torch_warp_composer_3d(
    u: torch.Tensor,
    affine: Optional[torch.Tensor] = None,
    v: torch.Tensor = None,
    align_corners: bool = True,
    grid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    
    Args:
        input: Input tensor of shape [B, Z, Y, X, 3]
        affine: Affine matrix of shape [B, 3, 4]
        grid: Optional grid tensor of shape [B, Z, Y, X, 3]
        mode: Interpolation mode ("bilinear", "nearest", "bicubic")
        padding_mode: Padding mode ("zeros", "border", "reflection")
        align_corners: Whether to align corners
        out_shape: Output shape tuple (Z, Y, X)
        is_displacement: Whether grid is a displacement field
    
    Returns:
        Sampled tensor of shape [B, C, Z, Y, X]
    """
    mode = "bilinear"
    padding_mode = "zeros"
    B = v.shape[0]
    if grid is None:
        if affine is None:
            affine = torch.eye(3, 4, device=u.device)[None].expand(B, -1, -1)
        grid = F.affine_grid(affine, [B, 1] + list(v.shape[1:-1]), align_corners=align_corners)
    return F.grid_sample(u.permute(0, 4, 1, 2, 3), grid + v, mode=mode, padding_mode=padding_mode, align_corners=align_corners).permute(0, 2, 3, 4, 1)

def torch_affine_warp_3d(
    affine: Optional[torch.Tensor],
    grid: torch.Tensor,
    align_corners: bool = True,
) -> torch.Tensor:
    B = grid.shape[0]
    if affine is None:
        affine = torch.eye(3, 4, device=grid.device)[None].expand(B, -1, -1)
    return F.affine_grid(affine, [B, 1] + list(grid.shape[1:-1]), align_corners=align_corners) + grid

def torch_affine_warp_2d(
    affine: Optional[torch.Tensor],
    grid: torch.Tensor,
    align_corners: bool = True,
) -> torch.Tensor:
    B = grid.shape[0]
    if affine is None:
        affine = torch.eye(2, 3, device=grid.device)[None].expand(B, -1, -1)
    return F.affine_grid(affine, [B, 1] + list(grid.shape[1:-1]), align_corners=align_corners) + grid
