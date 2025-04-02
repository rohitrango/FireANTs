#include <torch/torch.h>
#include <string>
#include <iostream>


torch::Tensor fused_warp_create_3d_forward_impl(
    const std::optional<torch::Tensor> affine,
    const torch::Tensor grid,
    const float grid_xmin, 
    const float grid_ymin,
    const float grid_zmin,
    const float grid_xmax,
    const float grid_ymax,
    const float grid_zmax);

void fused_warp_create_3d_backward_impl(
    /* we need input, A, u */
    const std::optional<torch::Tensor> affine,
    const torch::Tensor grid,
    const torch::Tensor &grad_output,
    const std::optional<torch::Tensor> &grad_affine,
    const std::optional<torch::Tensor> &grad_grid,
    const float grid_xmin, 
    const float grid_ymin,
    const float grid_zmin,
    const float grid_xmax,
    const float grid_ymax,
    const float grid_zmax
);