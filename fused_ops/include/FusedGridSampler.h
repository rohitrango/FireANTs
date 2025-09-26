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

torch::Tensor fused_grid_sampler_3d_forward_impl(
    const torch::Tensor &input, 
    const std::optional<torch::Tensor> affine_3d,
    const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> grid_affine,
    std::optional<torch::Tensor> output,
    const int64_t out_D,
    const int64_t out_H,
    const int64_t out_W,
    const float grid_xmin, 
    const float grid_ymin,
    const float grid_zmin,
    const float grid_xmax,
    const float grid_ymax,
    const float grid_zmax,
    const bool is_displacement,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners);

void fused_grid_sampler_3d_backward_impl(
    /* we need input, A, u */
    const torch::Tensor &input, 
    const std::optional<torch::Tensor> affine_3d,
    const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> grid_affine,
    /* we need grad_output, grad_input, grad_affine, grad_grid, some may be empty or zeros */
    const torch::Tensor &grad_output,
    const std::optional<torch::Tensor> &grad_input,
    const std::optional<torch::Tensor> &grad_affine,
    const std::optional<torch::Tensor> &grad_grid,
    /* input parameters = output size, grid bounds, is_displacement, interpolation_mode, padding_mode, align_corners */
    const int64_t out_D,
    const int64_t out_H,
    const int64_t out_W,
    const float grid_xmin, 
    const float grid_ymin,
    const float grid_zmin,
    const float grid_xmax,
    const float grid_ymax,
    const float grid_zmax,
    const bool is_displacement,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners);