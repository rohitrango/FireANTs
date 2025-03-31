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