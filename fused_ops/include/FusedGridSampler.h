#include <torch/torch.h>
#include <string>
#include <iostream>

torch::Tensor fused_grid_sampler_3d_forward_impl(
    const torch::Tensor &input, 
    const std::optional<torch::Tensor> affine_3d,
    const std::optional<torch::Tensor> grid,
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
    /* we need grad_output, grad_input, grad_affine, grad_grid, some may be empty or zeros */
    const torch::Tensor &grad_output,
    const std::optional<torch::Tensor> grad_input,
    const std::optional<torch::Tensor> grad_affine,
    const std::optional<torch::Tensor> grad_grid,
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