// Copyright (c) 2026 Rohit Jena. All rights reserved.
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

#pragma once

#include <torch/torch.h>
#include <optional>

// 3D generic label grid sampler: int-valued labels (stored as float), output = argmax over
// bilinear-interpolated label weights. Optional probability/weight map output.
// No interpolation_mode (fixed GenericLabel behavior).
std::tuple<torch::Tensor, std::optional<torch::Tensor>>
fused_grid_sampler_3d_generic_label_forward_impl(
    const torch::Tensor& input,
    const std::optional<torch::Tensor> affine_3d,
    const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> grid_affine,
    std::optional<torch::Tensor> output_labels,
    std::optional<torch::Tensor> output_weights,
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
    int64_t padding_mode,
    bool align_corners,
    bool return_weight,
    const std::optional<float> background_label = std::nullopt);

void fused_grid_sampler_3d_generic_label_backward_impl(
    const torch::Tensor& input,
    const std::optional<torch::Tensor> affine_3d,
    const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> grid_affine,
    const torch::Tensor& output_labels,
    const std::optional<torch::Tensor>& grad_weight,
    const std::optional<torch::Tensor>& grad_affine,
    const std::optional<torch::Tensor>& grad_grid,
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
    int64_t padding_mode,
    bool align_corners,
    bool return_weight);

// 2D generic label grid sampler (B, C, H, W).
std::tuple<torch::Tensor, std::optional<torch::Tensor>>
fused_grid_sampler_2d_generic_label_forward_impl(
    const torch::Tensor& input,
    const std::optional<torch::Tensor> affine_2d,
    const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> affine_2d_pregrid,
    std::optional<torch::Tensor> output_labels,
    std::optional<torch::Tensor> output_weights,
    const int64_t out_H,
    const int64_t out_W,
    const float grid_xmin,
    const float grid_ymin,
    const float grid_xmax,
    const float grid_ymax,
    const bool is_displacement,
    int64_t padding_mode,
    bool align_corners,
    bool return_weight,
    const std::optional<float> background_label = std::nullopt);

void fused_grid_sampler_2d_generic_label_backward_impl(
    const torch::Tensor& input,
    const std::optional<torch::Tensor> affine_2d,
    const std::optional<torch::Tensor> grid,
    const std::optional<torch::Tensor> affine_2d_pregrid,
    const torch::Tensor& output_labels,
    const std::optional<torch::Tensor>& grad_weight,
    const std::optional<torch::Tensor>& grad_affine,
    const std::optional<torch::Tensor>& grad_grid,
    const int64_t out_H,
    const int64_t out_W,
    const float grid_xmin,
    const float grid_ymin,
    const float grid_xmax,
    const float grid_ymax,
    const bool is_displacement,
    int64_t padding_mode,
    bool align_corners,
    bool return_weight);
