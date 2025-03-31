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
        ctx.save_for_backward(input, affine_3d, grid, interpolation_mode, padding_mode, align_corners, out_shape, min_coords, max_coords, is_displacement)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        pass


if __name__ == "__main__":
    Z, Y, X = 32, 64, 41
    Zi, Yi, Xi = 128, 431, 234
    align_corners = False
    input = torch.randn(1, 1, Zi, Yi, Xi).cuda().abs() + 1e-3
    affine_3d = torch.linalg.matrix_exp(torch.randn(3, 3))
    affine_3d = torch.cat([affine_3d, torch.zeros(3, 1)], dim=1)[None].cuda()

    output = FusedGridSampler3d.apply(input, affine_3d, None, "bilinear", "zeros", align_corners, (Z, Y, X), None, None, False)

    # baseline
    grid = F.affine_grid(affine_3d, (1, 1, Z, Y, X), align_corners=align_corners)
    output_baseline = F.grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=align_corners)

    print(output.shape, output.device, output_baseline.shape, output_baseline.device)
    rel_error = (torch.abs(output - output_baseline) / (1e-5 + torch.abs(output_baseline))).mean()
    print(f"Relative error: {rel_error}")
