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
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    # Case 2: Full warp field
    if not is_displacement:
        return F.grid_sample(input, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    # Case 3: Displacement field
    out_shape = grid.shape[1:-1]
    # Create identity grid if no affine provided
    if affine is None:
        affine = torch.eye(3, 4, device=input.device)[None].expand(B, -1, -1)
    # Create base grid
    base_grid = F.affine_grid(affine, (B, C, *out_shape[-3:]), align_corners=align_corners)
    # Add displacement
    base_grid = base_grid + grid
    return F.grid_sample(input, base_grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)

