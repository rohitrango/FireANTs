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


import torch
import pytest
from time import time
from torch import nn
from torch.nn import functional as F
from fireants.interpolator.grid_sample import torch_grid_sampler_3d as baseline_grid_sampler_3d
from fireants.interpolator.fused_grid_sample import FusedGridSampler3d
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--method", type=str, default="fused")
    return parser.parse_args()

# Run: PYTHONPATH=./ python tests/test_fusedsampler_mem.py --method [baseline|fused]

if __name__ == "__main__":
    arg = parse_args()
    if arg.method == "fused":
        input = torch.randn(1, 2, 196, 224, 160).to(torch.bfloat16).cuda()
    else:
        input = torch.randn(1, 2, 196, 224, 160).cuda()

    input_memory = torch.cuda.max_memory_allocated() / 1024**2
    # initialize affine_3d
    affine_3d = torch.linalg.matrix_exp(torch.randn(3, 3).cuda())[None]
    affine_3d = torch.cat([affine_3d, 0.01 * torch.randn(1, 3, 1).cuda()], dim=2)
    print(affine_3d.shape)
    out_shape = (240, 226, 190)
    if arg.method == "fused":
        output = FusedGridSampler3d.apply(input, affine_3d, None, "bilinear", "zeros", False, out_shape, None, None, False)
    else:
        output = baseline_grid_sampler_3d(input, affine=affine_3d, out_shape=out_shape, mode="bilinear", padding_mode="zeros", align_corners=False)
    total_memory = torch.cuda.max_memory_allocated() / 1024**2 
    output_memory = output.numel() * output.element_size() / 1024**2
    affine_memory = affine_3d.numel() * affine_3d.element_size() / 1024**2
    print(output.shape, output.device)
    print(f"Input memory: {input_memory:.2f}MB, Output memory: {output_memory:.2f}MB, Affine memory: {affine_memory:.2f}MB, Extra memory: {total_memory - input_memory - output_memory - affine_memory:.2f}MB")
