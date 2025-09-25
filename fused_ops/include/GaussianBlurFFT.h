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

void gaussian_blur_fft2(torch::Tensor &im_fft, int64_t ys, int64_t xs, int64_t ye, int64_t xe, float multiplier);
void gaussian_blur_fft3(torch::Tensor &im_fft, int64_t zs, int64_t ys, int64_t xs, int64_t ze, int64_t ye, int64_t xe, float multiplier);