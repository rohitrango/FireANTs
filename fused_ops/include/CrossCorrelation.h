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


#include <torch/torch.h>
#include <string>

enum class Reduction {
    MEAN,
    SUM, 
    NONE
};

torch::Tensor create_intermediates(torch::Tensor input_img, torch::Tensor target_img, torch::Tensor intermediates);

torch::Tensor cc3d_fwd_interm_v1(torch::Tensor intermediates, int64_t kernel_volume, Reduction reduction, float nr, float dr);

torch::Tensor cc3d_bwd_modify_interm_v1(torch::Tensor intermediates, torch::Tensor input_img, torch::Tensor target_img, 
    torch::Tensor grad_output, 
    std::optional<torch::Tensor> grad_input_img, std::optional<torch::Tensor> grad_target_img, 
    int64_t kernel_size, float nr, float dr, Reduction reduction);

void cc3d_bwd_compute_grads(torch::Tensor intermediates, torch::Tensor input_img, torch::Tensor target_img, torch::Tensor grad_input_img, std::optional<torch::Tensor> grad_target_img);