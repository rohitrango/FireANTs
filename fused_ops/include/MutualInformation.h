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


#include <iostream>
#include <torch/torch.h>
#include <string>

enum class KernelType {
    GAUSSIAN,
    BSPLINE, 
    DELTA,
};

std::vector<torch::Tensor> mutual_information_histogram_fwd(torch::Tensor &input_img, torch::Tensor &target_img, int num_bins, KernelType kernel_type, float minval, float maxval, float sigma_ratio, bool approximate_reduction);

void mutual_information_histogram_bwd(torch::Tensor &input_img, torch::Tensor &target_img, torch::Tensor &grad_pab, torch::Tensor &grad_pa, torch::Tensor &grad_pb, int num_bins, torch::Tensor &grad_input_img, std::optional<torch::Tensor> &grad_target_img, KernelType kernel_type, float minval, float maxval, float sigma_ratio);