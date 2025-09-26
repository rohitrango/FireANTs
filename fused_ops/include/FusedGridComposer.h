// Copyright (c) 2025 Rohit Jena. All rights reserved.
// 
// This file is part of FireANTs, distributed under the terms of
// the FireANTs License version 1.0. A copy of the license can be found
// in the LICENSE file at the root of this repository.
//
// IMPORTANT: This code is part of FireANTs and its use, reproduction, or
// distribution must comply with the full license terms, including:
// - Maintaining all copyright notices and bibliography references
// - Using only approved (re)-distribution channels 
// - Proper attribution in derivative works
//
// For full license details, see: https://github.com/rohitrango/FireANTs/blob/main/LICENSE 


#include <torch/torch.h>
#include <string>
#include <iostream>

void fused_grid_composer_3d_backward_impl(
    /* we need input, A, u */
    const torch::Tensor &input, 
    const std::optional<torch::Tensor> affine_3d,
    const torch::Tensor grid,
    /* we need grad_output, grad_input, grad_affine, grad_grid, some may be empty or zeros */
    const torch::Tensor &grad_output,
    const std::optional<torch::Tensor> &grad_input,
    const std::optional<torch::Tensor> &grad_affine,
    const std::optional<torch::Tensor> &grad_grid,
    /* input parameters = output size, grid bounds, is_displacement, interpolation_mode, padding_mode, align_corners */
    const float grid_xmin, 
    const float grid_ymin,
    const float grid_zmin,
    const float grid_xmax,
    const float grid_ymax,
    const float grid_zmax,
    bool align_corners);

torch::Tensor fused_grid_composer_3d_forward_impl(
    const torch::Tensor &input, 
    const std::optional<torch::Tensor> affine_3d,
    const torch::Tensor grid,
    const float grid_xmin, 
    const float grid_ymin,
    const float grid_zmin,
    const float grid_xmax,
    const float grid_ymax,
    const float grid_zmax,
    // interpolation is always bilinear
    // padding is always zeros
    // displacement is always True
    bool align_corners,
    std::optional<torch::Tensor> output);