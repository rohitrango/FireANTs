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

def mem(tensor):
    if tensor is None:
        return 0
    return tensor.element_size() * tensor.numel()

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
    for _ in range(1):
        Z, Y, X = 440, 410, 320
        Zi, Yi, Xi = 430, 220, 450
        align_corners = True
        batch_size = 1
        device = "cuda:1"
        input = torch.randn(batch_size, 1, Zi, Yi, Xi).to(device).abs() + 1e-3
        affine_3d = torch.linalg.matrix_exp(torch.randn(3, 3))
        affine_3d = torch.cat([affine_3d, torch.zeros(3, 1)], dim=1)[None].to(device)
        affine_3d = affine_3d.detach().requires_grad_(False)
        # affine_3d = None
        input = input.detach().requires_grad_(False)
        # displacement
        disp = (0.1 * torch.randn(1, Z, Y, X, 3)).to(device).detach().requires_grad_(True)

        start_mem = torch.cuda.max_memory_allocated(device)
        print(f"Max memory allocated: {start_mem / 1024**2} MB")

        def test_ours():
            start_ours = time()
            output = FusedGridSampler3d.apply(input, affine_3d, disp, "bilinear", "zeros", align_corners, (Z, Y, X), None, None, True)
            (output.mean()).backward()
            input_grad_ours = input.grad if input.grad is not None else torch.ones(3)
            grid_grad_ours = disp.grad  if disp.grad is not None else torch.ones(5)
            affine_3d_grad_ours = (affine_3d.grad if affine_3d.grad is not None else torch.ones(1, 3, 4)) if affine_3d is not None else torch.ones(1, 3, 4)
            input.grad = None
            disp.grad = None
            if affine_3d is not None:
                affine_3d.grad = None
            torch.cuda.synchronize()
            end_ours = time()
            print(f"Ours time: {end_ours - start_ours}")
            return input_grad_ours, grid_grad_ours, affine_3d_grad_ours, start_ours, end_ours, output
        # baseline
        input_grad_ours, grid_grad_ours, affine_3d_grad_ours, start_ours, end_ours, output = test_ours()
        mem_ours = torch.cuda.max_memory_allocated(device) - start_mem - mem(input_grad_ours) - mem(grid_grad_ours) - mem(affine_3d_grad_ours) - mem(output)
        print(f"Max memory allocated: {mem_ours / 1024**2} MB")

        def test_baseline():
            start_baseline = time()
            if affine_3d is not None:
                grid = F.affine_grid(affine_3d, (1, 1, Z, Y, X), align_corners=align_corners) + disp
            else:
                grid = F.affine_grid(torch.eye(3, 4)[None].to(device), (1, 1, Z, Y, X), align_corners=align_corners) + disp

            output_baseline = F.grid_sample(input, grid.expand(input.shape[0], -1, -1, -1, -1), mode="bilinear", padding_mode="zeros", align_corners=align_corners)
            (output_baseline.mean()).backward()
            input_grad_baseline = input.grad if input.grad is not None else torch.ones(3)
            grid_grad_baseline = disp.grad if disp.grad is not None else torch.ones(5)
            affine_3d_grad_baseline = (affine_3d.grad if affine_3d.grad is not None else torch.ones(1, 3, 4)) if affine_3d is not None else torch.ones(1, 3, 4)
            input.grad = None
            disp.grad = None
            if affine_3d is not None:
                affine_3d.grad = None
            torch.cuda.synchronize()
            end_baseline = time()
            print(f"Baseline time: {end_baseline - start_baseline}")
            return input_grad_baseline, grid_grad_baseline, affine_3d_grad_baseline, start_baseline, end_baseline, output_baseline
        
        input_grad_baseline, grid_grad_baseline, affine_3d_grad_baseline, start_baseline, end_baseline, output_baseline = test_baseline()
        mem_baseline = torch.cuda.max_memory_allocated(device) - start_mem - mem(input_grad_baseline) - mem(grid_grad_baseline) - mem(affine_3d_grad_baseline) - mem(input_grad_ours) - mem(grid_grad_ours) - mem(affine_3d_grad_ours) - mem(output_baseline) - mem(output)
        # Check contiguous

        print(f"Speedup: {(end_baseline - start_baseline) / (end_ours - start_ours)}")
        print(f"Memory: {mem_baseline / 1024**2} MB, Ours: {mem_ours / 1024**2} MB, Baseline: {mem_baseline / 1024**2} MB")

        print("-------------- Correctness ------------------")

        print(output.shape, output.device, output_baseline.shape, output_baseline.device)
        rel_error = (torch.abs(output - output_baseline) / (1e-5 + torch.abs(output_baseline))).mean()
        print(f"Relative error: {rel_error}\n")
        print(f"Input grads close: {torch.allclose(input_grad_ours, input_grad_baseline, rtol=1e-5)}")
        rel_input_grad_error = (torch.abs(input_grad_ours - input_grad_baseline) / (1e-5 + torch.abs(input_grad_baseline))).mean()
        print(f"Relative input grad error: {rel_input_grad_error}")
        print(f"Grid grads close: {torch.allclose(grid_grad_ours, grid_grad_baseline, atol=1e-4)}")
        print(f"Grid grads max: {torch.abs(grid_grad_ours - grid_grad_baseline).max()}")
        rel_grid_grad_error = (torch.abs(grid_grad_ours - grid_grad_baseline) / (1e-5 + torch.abs(grid_grad_baseline))).mean()
        print(f"Relative grid grad error: {rel_grid_grad_error}")
        print(f"Affine grads close: {torch.allclose(affine_3d_grad_ours, affine_3d_grad_baseline, atol=1e-4, )}")
        rel_affine_grad_error = (torch.abs(affine_3d_grad_ours - affine_3d_grad_baseline) / (1e-4 + torch.abs(affine_3d_grad_baseline))).mean()
        print(f"Relative/absolute affine grad error: {rel_affine_grad_error},{torch.abs(affine_3d_grad_ours - affine_3d_grad_baseline).mean()}")
        print("\n\n\n")

    # print(affine_3d_grad_ours - affine_3d_grad_baseline)
    # print(affine_3d_grad_ours)
    # print(affine_3d_grad_baseline)