'''
Implementation of the fused grid sampler in 3d
'''
from time import time, sleep
import torch
from torch import nn
from torch.nn import functional as F
import sys
from typing import Union, Tuple, List, Optional, Dict, Any, Callable
from fireants.types import ItemOrList
import fireants_fused_ops as ffo

def get_min_coords3d(Z, Y, X, align_corners):
    if not align_corners:
        return -1.0 + 1.0/X, -1.0 + 1.0/Y, -1.0 + 1.0/Z
    return -1.0, -1.0, -1.0

# ZYX order
def get_max_coords3d(Z, Y, X, align_corners):
    if not align_corners:
        return 1.0 - 1.0/X, 1.0 - 1.0/Y, 1.0 - 1.0/Z
    return 1.0, 1.0, 1.0

def get_min_coords2d(Y, X, align_corners):
    if not align_corners:
        return -1.0 + 1.0/X, -1.0 + 1.0/Y
    return -1.0, -1.0

# ZYX order
def get_max_coords2d(Y, X, align_corners):
    if not align_corners:
        return 1.0 - 1.0/X, 1.0 - 1.0/Y
    return 1.0, 1.0

GRID_SAMPLE_INTERPOLATION_MODES = {
    "bilinear": 0,
    "nearest": 1,
    "bicubic": 2,
}

GRID_SAMPLE_PADDING_MODES = {
    "zeros": 0,
    "border": 1,
    "reflection": 2,
}


class FusedGridSampler3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, affine, grid, interpolation_mode, padding_mode, align_corners, out_shape, min_coords, max_coords, is_displacement):
        # check input sizes
        '''
        input: [B, C, Z, Y, X]
        affine: [B, 3, 4]
        grid: [B, Z, Y, X, 3]
        interpolation_mode: str
        padding_mode: str
        align_corners: bool
        out_shape: tuple (Z, Y, X)
        min_coords: tuple (xmin, ymin, zmin)
        max_coords: tuple (xmax, ymax, zmax)
        is_displacement: bool
        '''
        # grid is assumed to be in same shape as input
        # affine mode, assume output shape is specified
        if grid is None:
            Z, Y, X = out_shape
        else:
            Z, Y, X = grid.shape[1:-1]
        # get min and max coords
        if min_coords is None:
            min_coords = get_min_coords3d(Z, Y, X, align_corners)
        if max_coords is None:
            max_coords = get_max_coords3d(Z, Y, X, align_corners)
        # get output
        try:
            output = ffo.fused_grid_sampler_3d_forward(input, affine, grid, Z, Y, X, min_coords[0], min_coords[1], min_coords[2], max_coords[0], max_coords[1], max_coords[2], is_displacement, GRID_SAMPLE_INTERPOLATION_MODES[interpolation_mode], GRID_SAMPLE_PADDING_MODES[padding_mode], align_corners)
        except Exception as e:
            print(f"Error in fused_grid_sampler_3d_forward: {e}")
            print(f"Input shape: {input.shape}, dtype: {input.dtype}, device: {input.device}")
            print(f"Affine shape: {affine.shape}, dtype: {affine.dtype}, device: {affine.device}")
            print(f"Grid shape: {grid.shape}, dtype: {grid.dtype}, device: {grid.device}")
            raise e
        # save everything for backward
        ctx.save_for_backward(input, affine, grid)
        ctx.interpolation_mode = interpolation_mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        ctx.out_shape = (Z, Y, X)
        ctx.min_coords = min_coords
        ctx.max_coords = max_coords
        ctx.is_displacement = is_displacement
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, affine, grid = ctx.saved_tensors
        interpolation_mode = ctx.interpolation_mode
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        out_shape = ctx.out_shape
        min_coords = ctx.min_coords
        max_coords = ctx.max_coords
        is_displacement = ctx.is_displacement

        # check if any of the variables require grad
        input_requires_grad = input.requires_grad
        affine_requires_grad = affine is not None and affine.requires_grad
        grid_requires_grad = grid is not None and grid.requires_grad
        input_grad, affine_grad, grid_grad = None, None, None
        # initialize gradients
        if input_requires_grad:
            input_grad = torch.zeros_like(input)
        if affine_requires_grad:
            affine_grad = torch.zeros_like(affine)
        if grid_requires_grad:
            grid_grad = torch.zeros_like(grid)
        # call backward
        ffo.fused_grid_sampler_3d_backward(input, affine, grid, grad_output, input_grad, affine_grad, grid_grad, out_shape[0], out_shape[1], out_shape[2], min_coords[0], min_coords[1], min_coords[2], max_coords[0], max_coords[1], max_coords[2], is_displacement, GRID_SAMPLE_INTERPOLATION_MODES[interpolation_mode], GRID_SAMPLE_PADDING_MODES[padding_mode], align_corners)
        return input_grad, affine_grad, grid_grad, None, None, None, None, None, None, None


class FusedWarpComposer3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, affine, grid, align_corners, min_coords, max_coords):
        # check input sizes
        '''
        input: [B, Z, Y, X, 3]
        affine: [B, 3, 4]
        grid: [B, Z, Y, X, 3]
        align_corners: bool
        min_coords: tuple (xmin, ymin, zmin)
        max_coords: tuple (xmax, ymax, zmax)
        '''
        # grid is assumed to be in same shape as input
        # affine mode, assume output shape is specified
        Z, Y, X = grid.shape[1:-1]
        # get min and max coords
        if min_coords is None:
            min_coords = get_min_coords3d(Z, Y, X, align_corners)
        if max_coords is None:
            max_coords = get_max_coords3d(Z, Y, X, align_corners)
        # get output
        output = ffo.fused_grid_composer_3d_forward(input, affine, grid, min_coords[0], min_coords[1], min_coords[2], max_coords[0], max_coords[1], max_coords[2], align_corners)
        # save everything for backward
        ctx.save_for_backward(input, affine, grid)
        ctx.align_corners = align_corners
        ctx.min_coords = min_coords
        ctx.max_coords = max_coords
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, affine, grid = ctx.saved_tensors
        align_corners = ctx.align_corners
        min_coords = ctx.min_coords
        max_coords = ctx.max_coords
        # check if any of the variables require grad
        input_requires_grad = input.requires_grad
        affine_requires_grad = affine is not None and affine.requires_grad
        grid_requires_grad = grid is not None and grid.requires_grad
        input_grad, affine_grad, grid_grad = None, None, None
        # initialize gradients
        if input_requires_grad:
            input_grad = torch.zeros_like(input)
        if affine_requires_grad:
            affine_grad = torch.zeros_like(affine)
        if grid_requires_grad:
            grid_grad = torch.zeros_like(grid)
        # for item, name in zip([input, affine, grid, grad_output, input_grad, affine_grad, grid_grad, output], ["input", "affine", "grid", "grad_output", "input_grad", "affine_grad", "grid_grad", "output"]):
        #     if item is not None:
        #         print(f"{name}: {item.shape} {item.dtype} {item.device}, {item.is_contiguous()}")
        # call backward
        ffo.fused_grid_composer_3d_backward(input, affine, grid, grad_output.contiguous(), input_grad, affine_grad, grid_grad, min_coords[0], min_coords[1], min_coords[2], max_coords[0], max_coords[1], max_coords[2], align_corners)
        # return gradients
        return input_grad, affine_grad, grid_grad, None, None, None

class FusedAffineWarp3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, affine, grid, align_corners, min_coords, max_coords):
        # check input sizes
        '''
        affine: [B, 3, 4]
        grid: [B, Z, Y, X, 3]
        '''
        Z, Y, X = grid.shape[1:-1]
        if min_coords is None:
            min_coords = get_min_coords3d(Z, Y, X, align_corners)
        if max_coords is None:
            max_coords = get_max_coords3d(Z, Y, X, align_corners)
        # get output
        assert affine is None or affine.is_contiguous(), "affine must be contiguous"
        assert grid.is_contiguous(), "grid must be contiguous"
        # get output
        output = ffo.fused_warp_create_3d_forward(affine, grid, min_coords[0], min_coords[1], min_coords[2], max_coords[0], max_coords[1], max_coords[2])
        # save everything for backward
        ctx.save_for_backward(affine, grid)
        ctx.align_corners = align_corners
        ctx.min_coords = min_coords
        ctx.max_coords = max_coords
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        affine, grid = ctx.saved_tensors
        align_corners = ctx.align_corners
        min_coords = ctx.min_coords
        max_coords = ctx.max_coords
        grad_affine, grad_grid = None, None
        if affine is not None and affine.requires_grad:
            grad_affine = torch.zeros_like(affine)
        if grid.requires_grad:
            grad_grid = torch.zeros_like(grid)
        # call backward
        ffo.fused_warp_create_3d_backward(affine, grid, grad_output.contiguous(), grad_affine, grad_grid, min_coords[0], min_coords[1], min_coords[2], max_coords[0], max_coords[1], max_coords[2])
        # return gradients
        return grad_affine, grad_grid, None, None, None


def fused_grid_sampler_3d(
    input: torch.Tensor,
    affine: Optional[torch.Tensor] = None,
    grid: Optional[torch.Tensor] = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
    min_coords: Optional[tuple] = None,
    max_coords: Optional[tuple] = None,
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
        interpolation_mode: Interpolation mode ("bilinear", "nearest", "bicubic")
        padding_mode: Padding mode ("zeros", "border", "reflection")
        align_corners: Whether to align corners
        out_shape: Output shape tuple (Z, Y, X)
        is_displacement: Whether grid is a displacement field
    
    Returns:
        Sampled tensor of shape [B, C, Z, Y, X]
    """
    B, C, Z, Y, X = input.shape
    # input = input.contiguous()
    # affine = affine.contiguous() if affine is not None else None
    # grid = grid.contiguous() if grid is not None else None
    assert input.is_contiguous(), "input must be contiguous"
    assert affine is None or affine.is_contiguous(), "affine must be contiguous"
    assert grid is None or grid.is_contiguous(), "grid must be contiguous"
    # specify output shape if grid is not provided
    if grid is None:
        out_shape = out_shape[-3:]
    else:
        out_shape = grid.shape[1:-1]
    output = FusedGridSampler3d.apply(input, affine, grid, mode, padding_mode, align_corners, out_shape, min_coords, max_coords, is_displacement)
    return output

def fused_warp_composer_3d(
    u: torch.Tensor,
    affine: Optional[torch.Tensor] = None,
    v: torch.Tensor = None,
    align_corners: bool = True,
    min_coords: Optional[torch.Tensor] = None,
    max_coords: Optional[torch.Tensor] = None,
    grid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes u \circ (Ax + v)
    
    Args:
        u: Input tensor of shape [B, Z, Y, X, 3]
        affine: Affine matrix of shape [B, 3, 4]
        v: Optional grid tensor of shape [B, Z, Y, X, 3]
        align_corners: Whether to align corners
        grid: Optional grid tensor of shape [B, Z, Y, X, 3]  not used in this function, only for compatibility with torch composer
    Returns:
        Sampled tensor of shape [B, C, Z, Y, X]
    """
    assert u.is_contiguous(), "input must be contiguous"
    assert affine is None or affine.is_contiguous(), "affine must be contiguous"
    assert v.is_contiguous(), "grid must be contiguous"    
    output = FusedWarpComposer3d.apply(u, affine, v, align_corners, min_coords, max_coords)
    return output

def fused_affine_warp_3d(
    affine: Optional[torch.Tensor],
    grid: torch.Tensor,
    align_corners: bool = True,
    min_coords: Optional[torch.Tensor] = None,
    max_coords: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return FusedAffineWarp3d.apply(affine, grid, align_corners, min_coords, max_coords)