import torch
import pytest
from time import time
from torch import nn
from torch.nn import functional as F
from ops.fusedgridsampler import FusedGridSampler3d
from ops.baseline_grid_sampler import baseline_grid_sampler_3d
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--method", type=str, default="fused")
    return parser.parse_args()

# Run: PYTHONPATH=./ python tests/test_fusedsampler_mem.py --method [baseline|fused]

if __name__ == "__main__":
    input = torch.randn(1, 2, 196, 224, 160).cuda()
    input_memory = torch.cuda.max_memory_allocated() / 1024**2
    # initialize affine_3d
    affine_3d = torch.linalg.matrix_exp(torch.randn(3, 3).cuda())[None]
    affine_3d = torch.cat([affine_3d, 0.01 * torch.randn(1, 3, 1).cuda()], dim=2)
    print(affine_3d.shape)
    arg = parse_args()
    out_shape = (240, 226, 190)
    if arg.method == "fused":
        output = FusedGridSampler3d.apply(input, affine_3d, None, "bilinear", "zeros", False, out_shape, None, None, False)
    else:
        output = baseline_grid_sampler_3d(input, affine_3d=affine_3d, out_shape=out_shape, interpolation_mode="bilinear", padding_mode="zeros", align_corners=False)
    total_memory = torch.cuda.max_memory_allocated() / 1024**2 
    output_memory = output.numel() * output.element_size() / 1024**2
    affine_memory = affine_3d.numel() * affine_3d.element_size() / 1024**2
    print(output.shape, output.device)
    print(f"Input memory: {input_memory:.2f}MB, Output memory: {output_memory:.2f}MB, Affine memory: {affine_memory:.2f}MB, Extra memory: {total_memory - input_memory - output_memory - affine_memory:.2f}MB")
