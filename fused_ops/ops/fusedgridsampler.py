'''
Fused Grid sampler
'''
from time import time, sleep
import torch
from torch import nn
from torch.nn import functional as F
import sys
from typing import Union, Tuple, List, Optional, Dict, Any, Callable
from fireants.types import ItemOrList
import fireants_fused_ops as ffo

def get_min_coords(Z, Y, X, align_corners):
    if not align_corners:
        return -1.0 + 1.0/X, -1.0 + 1.0/Y, -1.0 + 1.0/Z
    return -1.0, -1.0, -1.0

# ZYX order
def get_max_coords(Z, Y, X, align_corners):
    if not align_corners:
        return 1.0 - 1.0/X, 1.0 - 1.0/Y, 1.0 - 1.0/Z
    return 1.0, 1.0, 1.0

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
    def forward(ctx, input, affine_3d, grid, interpolation_mode, padding_mode, align_corners, out_shape, min_coords, max_coords, is_displacement):
        # check input sizes
        '''
        input: [B, C, Z, Y, X]
        affine_3d: [B, 3, 4]
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
            min_coords = get_min_coords(Z, Y, X, align_corners)
        if max_coords is None:
            max_coords = get_max_coords(Z, Y, X, align_corners)
        # get output
        output = ffo.fused_grid_sampler_3d_forward(input, affine_3d, grid, Z, Y, X, min_coords[0], min_coords[1], min_coords[2], max_coords[0], max_coords[1], max_coords[2], is_displacement, GRID_SAMPLE_INTERPOLATION_MODES[interpolation_mode], GRID_SAMPLE_PADDING_MODES[padding_mode], align_corners)
        # save everything for backward
        ctx.save_for_backward(input, affine_3d, grid)
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
        input, affine_3d, grid = ctx.saved_tensors
        interpolation_mode = ctx.interpolation_mode
        padding_mode = ctx.padding_mode
        align_corners = ctx.align_corners
        out_shape = ctx.out_shape
        min_coords = ctx.min_coords
        max_coords = ctx.max_coords
        is_displacement = ctx.is_displacement

        # check if any of the variables require grad
        input_requires_grad = input.requires_grad
        affine_3d_requires_grad = affine_3d is not None and affine_3d.requires_grad
        grid_requires_grad = grid is not None and grid.requires_grad
        input_grad, affine_3d_grad, grid_grad = None, None, None
        # initialize gradients
        if input_requires_grad:
            input_grad = torch.zeros_like(input)
        if affine_3d_requires_grad:
            affine_3d_grad = torch.zeros_like(affine_3d)
        if grid_requires_grad:
            grid_grad = torch.zeros_like(grid)
        # call backward
        ffo.fused_grid_sampler_3d_backward(input, affine_3d, grid, grad_output, input_grad, affine_3d_grad, grid_grad, out_shape[0], out_shape[1], out_shape[2], min_coords[0], min_coords[1], min_coords[2], max_coords[0], max_coords[1], max_coords[2], is_displacement, GRID_SAMPLE_INTERPOLATION_MODES[interpolation_mode], GRID_SAMPLE_PADDING_MODES[padding_mode], align_corners)
        return input_grad, affine_3d_grad, grid_grad, None, None, None, None, None, None, None


if __name__ == "__main__":
    Z, Y, X = 32, 64, 41
    Zi, Yi, Xi = 128, 431, 234
    align_corners = False
    input = torch.randn(1, 1, Zi, Yi, Xi).cuda().abs() + 1e-3
    affine_3d = torch.linalg.matrix_exp(torch.randn(3, 3))
    affine_3d = torch.cat([affine_3d, torch.zeros(3, 1)], dim=1)[None].cuda()
    # affine_3d = affine_3d.detach().requires_grad_(True)
    input = input.detach().requires_grad_(True)

    # displacement
    disp = (0.1 * torch.randn(1, Z, Y, X, 3)).cuda().detach().requires_grad_(True)
    print(disp.requires_grad)

    start_ours = time()
    output = FusedGridSampler3d.apply(input, affine_3d, disp, "bilinear", "zeros", align_corners, (Z, Y, X), None, None, True)
    (output.mean()).backward()
    input_grad_ours = input.grad
    grid_grad_ours = disp.grad
    input.grad = None
    disp.grad = None
    torch.cuda.synchronize()
    end_ours = time()
    print(f"Ours time: {end_ours - start_ours}")
    # baseline

    start_baseline = time()
    grid = F.affine_grid(affine_3d, (1, 1, Z, Y, X), align_corners=align_corners) + disp
    output_baseline = F.grid_sample(input, grid.expand(input.shape[0], -1, -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=align_corners)
    (output_baseline.mean()).backward()
    input_grad_baseline = input.grad
    grid_grad_baseline = disp.grad
    input.grad = None
    disp.grad = None
    torch.cuda.synchronize()
    end_baseline = time()
    print(f"Baseline time: {end_baseline - start_baseline}")

    print("\n-------------- Correctness ------------------\n")

    print(output.shape, output.device, output_baseline.shape, output_baseline.device)
    rel_error = (torch.abs(output - output_baseline) / (1e-5 + torch.abs(output_baseline))).mean()
    print(f"Relative error: {rel_error}\n")
    print(f"Input grads close: {torch.allclose(input_grad_ours, input_grad_baseline, rtol=1e-5)}")
    rel_input_grad_error = (torch.abs(input_grad_ours - input_grad_baseline) / (1e-5 + torch.abs(input_grad_baseline))).mean()
    print(f"Relative input grad error: {rel_input_grad_error}")
    print(f"Grid grads close: {torch.allclose(grid_grad_ours, grid_grad_baseline, rtol=1e-3)}")
    print(f"Grid grads max: {torch.abs(grid_grad_ours - grid_grad_baseline).max()}")
    rel_grid_grad_error = (torch.abs(grid_grad_ours - grid_grad_baseline) / (1e-5 + torch.abs(grid_grad_baseline))).mean()
    print(f"Relative grid grad error: {rel_grid_grad_error}")
