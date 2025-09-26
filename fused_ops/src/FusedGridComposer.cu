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


// #define TORCH_ASSERT_NO_OPERATORS
#include <ATen/OpMathType.h>
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
// #include <ATen/native/cuda/KernelUtils.h>
// #include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <c10/macros/Macros.h>
#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include "common.h"

// Core CUDA headers
#include <cuda_runtime.h>
#include <torch/extension.h>
// PyTorch CUDA headers

using namespace at::cuda::detail;
using namespace at::native;
using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

inline void check_grid_composer_common_v2(
  const torch::Tensor& input,
  const torch::Tensor& grid
) {
  auto input_opt = input.options();
  auto grid_opt = grid.options();

  TORCH_CHECK(
    input.defined(),
    "grid_sampler(): expected input to not be undefined");
  TORCH_CHECK(
    grid.defined(),
    "grid_sampler(): expected grid to not be undefined");
  TORCH_CHECK(
    input_opt.device() == grid_opt.device(),
    "grid_sampler(): expected input and grid to be on same device, but input "
    "is on ", input_opt.device(), " and grid is on ", grid_opt.device());
  TORCH_CHECK(
    grid.size(-1) == input.dim() - 2,
    "grid_sampler(): expected grid to have size ", input.dim() - 2, " in last "
    "dimension, but got grid with sizes ", grid.sizes());
    for (int i = 2; i < input.dim(); i++) {
        TORCH_CHECK(input.size(i) > 0,
        "grid_sampler(): expected input to have non-empty spatial dimensions, "
        "but input has sizes ", input.sizes(), " with dimension ", i, " being "
        "empty");
    }
}

template<typename scalar_t, typename index_t>
__forceinline__ __device__
void safe_add_3d_oneoffset_composer(
                scalar_t *data, 
                int d, int h, int w,
                int D, int H, int W,
                scalar_t delta,
                const index_t NC_offset,
                const index_t memory_span) {
  if (within_bounds_3d(d, h, w, D, H, W)) {
    index_t offset = NC_offset + 3 * (w + W * (h + H * d));
      gpuAtomicAdd(data + offset, delta);
  }
}

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(BLOCKSIZE_3D)
__global__ void fused_grid_composer_3d_forward_kernel(
    const index_t count,
    const scalar_t* input,
    const scalar_t* grid,
    const scalar_t* affine_3d,
    const index_t N,
    const index_t Di,
    const index_t Hi,
    const index_t Wi,
    const index_t D,
    const index_t H,
    const index_t W,
    const float grid_xmin,
    const float grid_ymin,
    const float grid_zmin,
    const float grid_xmax,
    const float grid_ymax,
    const float grid_zmax,
    scalar_t* output,
    const bool align_corners,
    const bool broadcast_input,
    const bool broadcast_affine_3d,
    const bool broadcast_grid
    ) {

    using opmath_t = at::opmath_type<scalar_t>;
    // fix these values
    const GridSamplerPadding padding_mode = GridSamplerPadding::Zeros;

    CUDA_KERNEL_LOOP_TYPE(index, count, index_t) {
        const index_t w = index % W;
        const index_t h = (index / W) % H;
        const index_t d = (index / (H * W)) % D;
        const index_t n = index / (D * H * W);
        // we have 3 coordinates for each grid point, so we multiply the index by 3
        const index_t grid_offset = 3 * (w + W * (h + H * (d + (broadcast_grid ? 0 : (D * n)))));

        // this is only affine coordinate
        opmath_t ix = 0, iy = 0, iz = 0;
        opmath_t x = 0, y = 0, z = 0;
        ix = w * (grid_xmax - grid_xmin) / (W-1) + grid_xmin;
        iy = h * (grid_ymax - grid_ymin) / (H-1) + grid_ymin;
        iz = d * (grid_zmax - grid_zmin) / (D-1) + grid_zmin;
        if (!grid) {
            // if grid is not provided, then affine matrix is multiplied to input coordinate
            // displacement is ignored
            // just affine coordiante here, we load the entire affine matrix
            const scalar_t* affine_3d_ptr = affine_3d + (broadcast_affine_3d ? 0 : (12 * n));
            // get normalized coordinate
            x = affine_3d_ptr[0] * ix + affine_3d_ptr[1] * iy + affine_3d_ptr[2] * iz + affine_3d_ptr[3];
            y = affine_3d_ptr[4] * ix + affine_3d_ptr[5] * iy + affine_3d_ptr[6] * iz + affine_3d_ptr[7];
            z = affine_3d_ptr[8] * ix + affine_3d_ptr[9] * iy + affine_3d_ptr[10] * iz + affine_3d_ptr[11];
        }
        else {
            // grid is provided, load the grid coordinate            // get grid coordinate
            // apply affine matrix
            if(affine_3d) {
                const scalar_t* affine_3d_ptr = affine_3d + (broadcast_affine_3d ? 0 : (12 * n));
                x = affine_3d_ptr[0] * ix + affine_3d_ptr[1] * iy + affine_3d_ptr[2] * iz + affine_3d_ptr[3];
                y = affine_3d_ptr[4] * ix + affine_3d_ptr[5] * iy + affine_3d_ptr[6] * iz + affine_3d_ptr[7];
                z = affine_3d_ptr[8] * ix + affine_3d_ptr[9] * iy + affine_3d_ptr[10] * iz + affine_3d_ptr[11];
            }
            else {
                x = ix; y = iy; z = iz;
            }
            // add to displacement
            x += grid[grid_offset];
            y += grid[grid_offset + 1];
            z += grid[grid_offset + 2];
        }

        // get the corresponding input x, y, z co-ordinates from grid
        ix = grid_sampler_compute_source_index(x, Wi, padding_mode, align_corners);
        iy = grid_sampler_compute_source_index(y, Hi, padding_mode, align_corners);
        iz = grid_sampler_compute_source_index(z, Di, padding_mode, align_corners);

        // get corner pixel values from (x, y, z)
        // for 4d, we used north-east-south-west
        // for 5d, we add top-bottom
        index_t ix_tnw = static_cast<index_t>(::floor(ix));
        index_t iy_tnw = static_cast<index_t>(::floor(iy));
        index_t iz_tnw = static_cast<index_t>(::floor(iz));

        index_t ix_tne = ix_tnw + 1;
        index_t iy_tne = iy_tnw;
        index_t iz_tne = iz_tnw;

        index_t ix_tsw = ix_tnw;
        index_t iy_tsw = iy_tnw + 1;
        index_t iz_tsw = iz_tnw;

        index_t ix_tse = ix_tnw + 1;
        index_t iy_tse = iy_tnw + 1;
        index_t iz_tse = iz_tnw;

        index_t ix_bnw = ix_tnw;
        index_t iy_bnw = iy_tnw;
        index_t iz_bnw = iz_tnw + 1;

        index_t ix_bne = ix_tnw + 1;
        index_t iy_bne = iy_tnw;
        index_t iz_bne = iz_tnw + 1;

        index_t ix_bsw = ix_tnw;
        index_t iy_bsw = iy_tnw + 1;
        index_t iz_bsw = iz_tnw + 1;

        index_t ix_bse = ix_tnw + 1;
        index_t iy_bse = iy_tnw + 1;
        index_t iz_bse = iz_tnw + 1;

        // get surfaces to each neighbor:
        opmath_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
        opmath_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
        opmath_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
        opmath_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
        opmath_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
        opmath_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
        opmath_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
        opmath_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

        // get input and output pointers
        const scalar_t* inp_ptr_NC = input + (broadcast_input ? 0 : (n * (Di * Hi * Wi * 3)));
        scalar_t* out_ptr_NCDHW = output + 3 * (w + W * (h + H * (d + D * n))); // add batch, depth, height, width offset

        #pragma unroll
        for (index_t c = 0; c < 3; ++c, inp_ptr_NC += 1, out_ptr_NCDHW += 1) {
            //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
            // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
            // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
            // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
            opmath_t out_acc = 0;
            if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, Di, Hi, Wi)) {
                out_acc += inp_ptr_NC[3 * (ix_tnw + Wi * (iy_tnw + Hi * iz_tnw))] * tnw;
            }
            if (within_bounds_3d(iz_tne, iy_tne, ix_tne, Di, Hi, Wi)) {
                out_acc += inp_ptr_NC[3 * (ix_tne + Wi * (iy_tne + Hi * iz_tne))] * tne;
            }
            if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, Di, Hi, Wi)) {
                out_acc += inp_ptr_NC[3 * (ix_tsw + Wi * (iy_tsw + Hi * iz_tsw))] * tsw;
            }
            if (within_bounds_3d(iz_tse, iy_tse, ix_tse, Di, Hi, Wi)) {
                out_acc += inp_ptr_NC[3 * (ix_tse + Wi * (iy_tse + Hi * iz_tse))] * tse;
            }
            if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, Di, Hi, Wi)) {
                out_acc += inp_ptr_NC[3 * (ix_bnw + Wi * (iy_bnw + Hi * iz_bnw))] * bnw;
            }
            if (within_bounds_3d(iz_bne, iy_bne, ix_bne, Di, Hi, Wi)) {
                out_acc += inp_ptr_NC[3 * (ix_bne + Wi * (iy_bne + Hi * iz_bne))] * bne;
            }
            if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, Di, Hi, Wi)) {
                out_acc += inp_ptr_NC[3 * (ix_bsw + Wi * (iy_bsw + Hi * iz_bsw))] * bsw;
            }
            if (within_bounds_3d(iz_bse, iy_bse, ix_bse, Di, Hi, Wi)) {
                out_acc += inp_ptr_NC[3 * (ix_bse + Wi * (iy_bse + Hi * iz_bse))] * bse;
            }
            *out_ptr_NCDHW += out_acc;
        }
    }
}

// Note [Passing pointer and offset to fastAtomicAdd]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// For its internal bounds checking, fastAtomicAdd needs to know where the destination address
// lies relative to the entire tensor, so we pass the base grad_input.data and full offset information,
// including batch * channel offset (NC_offset).
template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(BLOCKSIZE_3D)
__global__ void fused_grid_composer_3d_backward_kernel(
        const index_t count, /* D * H * W */
        const scalar_t* input,
        const scalar_t* grid,
        const scalar_t* affine_3d,
        // grads
        const scalar_t* grad_output,
        scalar_t* grad_input,
        scalar_t* grad_affine_collect,
        scalar_t* grad_grid,
        // input size parameters
        const index_t N,
        const index_t Di,
        const index_t Hi,
        const index_t Wi,
        const index_t D,
        const index_t H,
        const index_t W,
        const float grid_xmin,
        const float grid_ymin,
        const float grid_zmin,
        const float grid_xmax,
        const float grid_ymax,
        const float grid_zmax,
        const bool align_corners,
        const bool broadcast_input,
        const bool broadcast_affine_3d,
        const bool broadcast_grid) {

    // batch index is separated from the other dimensions
    const index_t n = blockIdx.y;

    // collect affine gradients
    scalar_t _affine_grad_[12];
    #pragma unroll
    for(index_t i = 0; i < 12; ++i) {
        _affine_grad_[i] = 0;
    }
    // shared memory to take affine gradient sum over block
    __shared__ scalar_t _affine_grad_shared_[BLOCKSIZE_3D];   
    _affine_grad_shared_[threadIdx.x] = 0;

    const GridSamplerPadding padding_mode = GridSamplerPadding::Zeros;

    // also collect affine map locally to avoid loading multiple times
    scalar_t _affine_map_[12];
    if (affine_3d) {
        const index_t offset = broadcast_affine_3d ? 0 : (12 * n);
        #pragma unroll
        for (index_t i = 0; i < 12; ++i) {
            _affine_map_[i] = affine_3d[offset + i];
        }
    }

    // loop over
    CUDA_KERNEL_LOOP_TYPE(index, count, index_t) {
        const index_t w = index % W;
        const index_t h = (index / W) % H;
        const index_t d = (index / (H * W)) % D;
        const index_t grid_offset = 3 * (w + W * (h + H * (d + (broadcast_grid ? 0 : (D * n)))));

        // get the corresponding input x, y, z co-ordinates from grid
        scalar_t pax, pay, paz;   // pax = pre-affine x
        scalar_t ix, iy, iz;
        scalar_t x, y, z;

        ix = w * (grid_xmax - grid_xmin) / (W-1) + grid_xmin;
        iy = h * (grid_ymax - grid_ymin) / (H-1) + grid_ymin;
        iz = d * (grid_zmax - grid_zmin) / (D-1) + grid_zmin;        // get grid coordinates  phi = (A? * x + u?)
        pax = ix; pay = iy; paz = iz;
        // calculate Ax + grid
        if(affine_3d) {
            x = _affine_map_[0] * ix + _affine_map_[1] * iy + _affine_map_[2] * iz + _affine_map_[3];
            y = _affine_map_[4] * ix + _affine_map_[5] * iy + _affine_map_[6] * iz + _affine_map_[7];
            z = _affine_map_[8] * ix + _affine_map_[9] * iy + _affine_map_[10] * iz + _affine_map_[11];
        }
        else {
            x = ix; y = iy; z = iz;
        }

        if (grid) {
            x += grid[grid_offset];
            y += grid[grid_offset + 1];
            z += grid[grid_offset + 2];
        }

        // multipliers for gradients on ix, iy, and iz
        scalar_t gix_mult, giy_mult, giz_mult;
        ix = grid_sampler_compute_source_index_set_grad(x, Wi, padding_mode, align_corners, &gix_mult);
        iy = grid_sampler_compute_source_index_set_grad(y, Hi, padding_mode, align_corners, &giy_mult);
        iz = grid_sampler_compute_source_index_set_grad(z, Di, padding_mode, align_corners, &giz_mult);

        // for 5d, we add top-bottom
        index_t ix_tnw = static_cast<index_t>(std::floor(ix));
        index_t iy_tnw = static_cast<index_t>(std::floor(iy));
        index_t iz_tnw = static_cast<index_t>(std::floor(iz));

        index_t ix_tne = ix_tnw + 1;
        index_t iy_tne = iy_tnw;
        index_t iz_tne = iz_tnw;

        index_t ix_tsw = ix_tnw;
        index_t iy_tsw = iy_tnw + 1;
        index_t iz_tsw = iz_tnw;

        index_t ix_tse = ix_tnw + 1;
        index_t iy_tse = iy_tnw + 1;
        index_t iz_tse = iz_tnw;

        index_t ix_bnw = ix_tnw;
        index_t iy_bnw = iy_tnw;
        index_t iz_bnw = iz_tnw + 1;

        index_t ix_bne = ix_tnw + 1;
        index_t iy_bne = iy_tnw;
        index_t iz_bne = iz_tnw + 1;

        index_t ix_bsw = ix_tnw;
        index_t iy_bsw = iy_tnw + 1;
        index_t iz_bsw = iz_tnw + 1;

        index_t ix_bse = ix_tnw + 1;
        index_t iy_bse = iy_tnw + 1;
        index_t iz_bse = iz_tnw + 1;

        // get surfaces to each neighbor:
        scalar_t tnw = (ix_bse - ix)        * (iy_bse - iy)        * (iz_bse - iz);
        scalar_t tne = (ix        - ix_bsw) * (iy_bsw - iy)        * (iz_bsw - iz);
        scalar_t tsw = (ix_bne - ix)        * (iy        - iy_bne) * (iz_bne - iz);
        scalar_t tse = (ix        - ix_bnw) * (iy        - iy_bnw) * (iz_bnw - iz);
        scalar_t bnw = (ix_tse - ix)        * (iy_tse - iy)        * (iz - iz_tse);
        scalar_t bne = (ix        - ix_tsw) * (iy_tsw - iy)        * (iz - iz_tsw);
        scalar_t bsw = (ix_tne - ix)        * (iy        - iy_tne) * (iz - iz_tne);
        scalar_t bse = (ix        - ix_tnw) * (iy        - iy_tnw) * (iz - iz_tnw);

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0), giz = static_cast<scalar_t>(0);
        // get grad_output pointer
        const scalar_t *gOut_ptr_NCDHW = grad_output + 3 * (w + W * (h + H * (d + D * n)));
        // offset for grad_input
        index_t NC_offset = (broadcast_input ? 0 : (n * 3 * Hi * Wi * Di));
        const scalar_t *inp_ptr_NC = input + NC_offset;
        // get offsets to add
        const index_t grad_input_memory_span = (broadcast_input ? 1 : N) * (3 * Hi * Wi * Di);
        
        #pragma unroll
        for (index_t c = 0; c < 3; ++c, gOut_ptr_NCDHW += 1, NC_offset += 1, inp_ptr_NC += 1) {
            scalar_t gOut = *gOut_ptr_NCDHW;
            // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
            if (grad_input) {
                safe_add_3d_oneoffset_composer(grad_input, iz_tnw, iy_tnw, ix_tnw, Di, Hi, Wi, tnw * gOut, NC_offset, grad_input_memory_span);
                safe_add_3d_oneoffset_composer(grad_input, iz_tne, iy_tne, ix_tne, Di, Hi, Wi, tne * gOut, NC_offset, grad_input_memory_span);
                safe_add_3d_oneoffset_composer(grad_input, iz_tsw, iy_tsw, ix_tsw, Di, Hi, Wi, tsw * gOut, NC_offset, grad_input_memory_span);
                safe_add_3d_oneoffset_composer(grad_input, iz_tse, iy_tse, ix_tse, Di, Hi, Wi, tse * gOut, NC_offset, grad_input_memory_span);
                safe_add_3d_oneoffset_composer(grad_input, iz_bnw, iy_bnw, ix_bnw, Di, Hi, Wi, bnw * gOut, NC_offset, grad_input_memory_span);
                safe_add_3d_oneoffset_composer(grad_input, iz_bne, iy_bne, ix_bne, Di, Hi, Wi, bne * gOut, NC_offset, grad_input_memory_span);
                safe_add_3d_oneoffset_composer(grad_input, iz_bsw, iy_bsw, ix_bsw, Di, Hi, Wi, bsw * gOut, NC_offset, grad_input_memory_span);
                safe_add_3d_oneoffset_composer(grad_input, iz_bse, iy_bse, ix_bse, Di, Hi, Wi, bse * gOut, NC_offset, grad_input_memory_span);
            }
            // // calculate grad_grid
            if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, Di, Hi, Wi)) {
                scalar_t tnw_val = inp_ptr_NC[3 * (ix_tnw + Wi * (iy_tnw + Hi * iz_tnw))];
                gix -= tnw_val * (iy_bse - iy)        * (iz_bse - iz)        * gOut;
                giy -= tnw_val * (ix_bse - ix)        * (iz_bse - iz)        * gOut;
                giz -= tnw_val * (ix_bse - ix)        * (iy_bse - iy)        * gOut;
            }
            if (within_bounds_3d(iz_tne, iy_tne, ix_tne, Di, Hi, Wi)) {
                scalar_t tne_val = inp_ptr_NC[3 * (ix_tne + Wi * (iy_tne + Hi * iz_tne))];
                gix += tne_val * (iy_bsw - iy)        * (iz_bsw - iz)        * gOut;
                giy -= tne_val * (ix        - ix_bsw) * (iz_bsw - iz)        * gOut;
                giz -= tne_val * (ix        - ix_bsw) * (iy_bsw - iy)        * gOut;
            }
            if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, Di, Hi, Wi)) {
                scalar_t tsw_val = inp_ptr_NC[3 * (ix_tsw + Wi * (iy_tsw + Hi * iz_tsw))];
                gix -= tsw_val * (iy - iy_bne)        * (iz_bne - iz)        * gOut;
                giy += tsw_val * (ix_bne - ix)        * (iz_bne - iz)        * gOut;
                giz -= tsw_val * (ix_bne - ix)        * (iy        - iy_bne) * gOut;
            }
            if (within_bounds_3d(iz_tse, iy_tse, ix_tse, Di, Hi, Wi)) {
                scalar_t tse_val = inp_ptr_NC[3 * (ix_tse + Wi * (iy_tse + Hi * iz_tse))];
                gix += tse_val * (iy - iy_bnw)        * (iz_bnw - iz)        * gOut;
                giy += tse_val * (ix        - ix_bnw) * (iz_bnw - iz)        * gOut;
                giz -= tse_val * (ix        - ix_bnw) * (iy        - iy_bnw) * gOut;
            }
            if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, Di, Hi, Wi)) {
                scalar_t bnw_val = inp_ptr_NC[3 * (ix_bnw + Wi * (iy_bnw + Hi * iz_bnw))];
                gix -= bnw_val * (iy_tse - iy)        * (iz - iz_tse)        * gOut;
                giy -= bnw_val * (ix_tse - ix)        * (iz - iz_tse)        * gOut;
                giz += bnw_val * (ix_tse - ix)        * (iy_tse - iy)        * gOut;
            }
            if (within_bounds_3d(iz_bne, iy_bne, ix_bne, Di, Hi, Wi)) {
                scalar_t bne_val = inp_ptr_NC[3 * (ix_bne + Wi * (iy_bne + Hi * iz_bne))];
                gix += bne_val * (iy_tsw - iy)        * (iz - iz_tsw)        * gOut;
                giy -= bne_val * (ix        - ix_tsw) * (iz - iz_tsw)        * gOut;
                giz += bne_val * (ix        - ix_tsw) * (iy_tsw - iy)        * gOut;
            }
            if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, Di, Hi, Wi)) {
                scalar_t bsw_val = inp_ptr_NC[3 * (ix_bsw + Wi * (iy_bsw + Hi * iz_bsw))];
                gix -= bsw_val * (iy - iy_tne)        * (iz - iz_tne)        * gOut;
                giy += bsw_val * (ix_tne - ix)        * (iz - iz_tne)        * gOut;
                giz += bsw_val * (ix_tne - ix)        * (iy        - iy_tne) * gOut;
            }
            if (within_bounds_3d(iz_bse, iy_bse, ix_bse, Di, Hi, Wi)) {
                scalar_t bse_val = inp_ptr_NC[3 * (ix_bse + Wi * (iy_bse + Hi * iz_bse))];
                gix += bse_val * (iy - iy_tnw)        * (iz - iz_tnw)        * gOut;
                giy += bse_val * (ix        - ix_tnw) * (iz - iz_tnw)        * gOut;
                giz += bse_val * (ix        - ix_tnw) * (iy        - iy_tnw) * gOut;
            }
        }

        // multiply by grad_output multiplier
        gix = gix_mult * gix;
        giy = giy_mult * giy;
        giz = giz_mult * giz;

        // calculate grad_grid
        if (grad_grid) {
            if (broadcast_grid) {
                index_t grad_grid_index = 3 * (w + W * (h + H * (d)));
                gpuAtomicAdd(grad_grid + grad_grid_index, gix);
                gpuAtomicAdd(grad_grid + grad_grid_index + 1, giy);
                gpuAtomicAdd(grad_grid + grad_grid_index + 2, giz);
            } 
            else {
                index_t grad_grid_index = 3 * (w + W * (h + H * (d + D * n)));
                grad_grid[grad_grid_index] = gix;
                grad_grid[grad_grid_index + 1] = giy;
                grad_grid[grad_grid_index + 2] = giz;
            }
        }

        // if affine_3d grad is required
        if (grad_affine_collect) {
            // add it to local registers
            _affine_grad_[0] += gix * pax;
            _affine_grad_[1] += gix * pay;
            _affine_grad_[2] += gix * paz;
            _affine_grad_[3] += gix;
            _affine_grad_[4] += giy * pax;
            _affine_grad_[5] += giy * pay;
            _affine_grad_[6] += giy * paz;
            _affine_grad_[7] += giy;
            _affine_grad_[8] += giz * pax;
            _affine_grad_[9] += giz * pay;
            _affine_grad_[10] += giz * paz;
            _affine_grad_[11] += giz;
        }
    }

    // // if affine_3d grad is required
    if (grad_affine_collect) {
        // add it to local registers
        #pragma unroll
        for (int affid = 0; affid < 12; ++affid) {
            // put it in shared memory to compute the sum over the batch dimension
            _affine_grad_shared_[threadIdx.x] = _affine_grad_[affid];
            __syncthreads();
            
            // reduce over threads
            for (int tid = BLOCKSIZE_3D / 2; tid > 0; tid /= 2) {
                if (threadIdx.x < tid) {
                    _affine_grad_shared_[threadIdx.x] += _affine_grad_shared_[threadIdx.x + tid];
                }
                __syncthreads();
            }
            
            // write to global memory
            if (threadIdx.x == 0) {
                // broadcasted, we need to perform safe atomic add to avoid conflicts with other threads along batch dimension
                if (broadcast_affine_3d) {
                    const index_t offset = blockIdx.x*12 + affid;
                    gpuAtomicAdd(grad_affine_collect + offset, _affine_grad_shared_[0]);
                }
                else {
                    const index_t offset = affid + 12 * (blockIdx.x + gridDim.x * n);
                    grad_affine_collect[offset] = _affine_grad_shared_[0];
                }
            }
        }
    }
}

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
    std::optional<torch::Tensor> output) {

    int64_t D, H, W;
    TORCH_CHECK(input.dim() == 5, "input must be 5D");
    TORCH_CHECK(input.device().is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.size(4) == 3, "input must have 3 channels");

    // device and stream guards
    c10::DeviceGuard guard(input.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input.device().index());
    at::cuda::CUDAStreamGuard stream_guard(stream);

    // see if we need to broadcast any variable
    int64_t batch_size_max = input.size(0);
    if (affine_3d.has_value()) {
        batch_size_max = std::max(batch_size_max, affine_3d.value().size(0));
    }
    // check grid and inputs
    check_grid_composer_common_v2(input, grid);
    batch_size_max = std::max(batch_size_max, grid.size(0));

    // broadcast none by default 
    bool broadcast_input = false, broadcast_affine_3d = false, broadcast_grid = false;
    if (batch_size_max > 1) {
        if (input.size(0) == 1) {
            broadcast_input = true;
        } else if (input.size(0) != batch_size_max) {
            TORCH_CHECK(false, "input batch size must match batch size of affine_3d or grid");
        }

        // broadcast affine_3d if it exists
        if (affine_3d.has_value() && affine_3d.value().size(0) == 1) {
            broadcast_affine_3d = true;
        } else if (affine_3d.has_value() && affine_3d.value().size(0) != batch_size_max) {  
            TORCH_CHECK(false, "affine_3d batch size must match batch size of input or grid");
        }
        
        // broadcast grid if it exists
        if (grid.size(0) == 1) {
            broadcast_grid = true;
        } else if (grid.size(0) != batch_size_max) {
            TORCH_CHECK(false, "grid batch size must match batch size of input or affine_3d");
        }
    }

    // D, H, W will be determined by grid
    TORCH_CHECK(grid.is_contiguous(), "grid must be contiguous");
    TORCH_CHECK(grid.size(4) == 3, "grid must have 3 channels");
    D = grid.size(1);
    H = grid.size(2);
    W = grid.size(3);

    if (affine_3d.has_value()) {
        TORCH_CHECK(affine_3d.value().dim() == 3, "affine_3d must be (B, 3, 4)");
        TORCH_CHECK(affine_3d.value().device().is_cuda(), "affine_3d must be on CUDA");
        TORCH_CHECK(affine_3d.value().is_contiguous(), "affine_3d must be contiguous");
        TORCH_CHECK(affine_3d.value().size(1) == 3, "affine_3d must be (B, 3, 4)");
        TORCH_CHECK(affine_3d.value().size(2) == 4, "affine_3d must be (B, 3, 4)");
    }

    // define output
    int64_t N = batch_size_max;
    // specify output
    if (output.has_value()) {
        TORCH_CHECK(output.value().is_contiguous(), "output must be contiguous");
        TORCH_CHECK(output.value().size(4) == 3, "output must have 3 channels");
        TORCH_CHECK(output.value().device().is_cuda(), "output must be on CUDA");
        TORCH_CHECK(output.value().device() == input.device(), "output must be on the same device as input");
    } else {
        output.emplace(torch::zeros({batch_size_max, D, H, W, 3}, input.options()));
    }

    // torch::Tensor output = torch::zeros({batch_size_max, D, H, W, 3}, input.options());
    // input size parameters
    int64_t count = N * D * H * W;
    // input spatial size parameters
    int64_t Di = input.size(1);
    int64_t Hi = input.size(2);
    int64_t Wi = input.size(3);

    if (count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "fused_grid_composer_3d_forward_kernel", [&] {
            // check if grid is 32-bit
            if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
                canUse32BitIndexMath(output.value())) {
                fused_grid_composer_3d_forward_kernel<scalar_t>
                <<<GET_BLOCKS(count, 512), 512, 0, stream>>>(
                    static_cast<int>(count),
                    input.data_ptr<scalar_t>(),
                    grid.data_ptr<scalar_t>(),
                    affine_3d.has_value() ? affine_3d.value().data_ptr<scalar_t>() : nullptr,
                    static_cast<int>(N), static_cast<int>(Di), static_cast<int>(Hi), static_cast<int>(Wi),
                    static_cast<int>(D), static_cast<int>(H), static_cast<int>(W),
                    grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
                    // output
                    output.value().data_ptr<scalar_t>(),
                    align_corners,
                    broadcast_input,
                    broadcast_affine_3d,
                    broadcast_grid
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            } else {
                fused_grid_composer_3d_forward_kernel<scalar_t>
                <<<GET_BLOCKS(count, 512), 512, 0, stream>>>(
                    count,
                    input.data_ptr<scalar_t>(),
                    grid.data_ptr<scalar_t>(),
                    affine_3d.has_value() ? affine_3d.value().data_ptr<scalar_t>() : nullptr,
                    N, Di, Hi, Wi,
                    D, H, W,
                    grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
                    output.value().data_ptr<scalar_t>(),
                    align_corners,
                    broadcast_input,
                    broadcast_affine_3d,
                    broadcast_grid
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        });
    }
    return output.value();
}

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
    bool align_corners) {

    TORCH_CHECK(input.dim() == 5, "input must be 5D");
    TORCH_CHECK(input.device().is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.size(4) == 3, "input must have 3 channels");

    TORCH_CHECK(grad_input.has_value() || grad_affine.has_value() || grad_grid.has_value(), "at least one of grad_input, grad_affine, grad_grid must exist");

    // device and stream guards
    c10::DeviceGuard guard(input.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input.device().index());
    at::cuda::CUDAStreamGuard stream_guard(stream);

    // see if we need to broadcast any variable
    int64_t batch_size_max = input.size(0);
    if (affine_3d.has_value()) {
        batch_size_max = std::max(batch_size_max, affine_3d.value().size(0));
    }
    check_grid_composer_common_v2(input, grid);
    batch_size_max = std::max(batch_size_max, grid.size(0));
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");

    // broadcast none by default 
    bool broadcast_input = false, broadcast_affine_3d = false, broadcast_grid = false;
    if (batch_size_max > 1) {
        if (input.size(0) == 1) {
            broadcast_input = true;
        } else if (input.size(0) != batch_size_max) {
            TORCH_CHECK(false, "input batch size must match batch size of affine_3d or grid");
        }

        // broadcast affine_3d if it exists
        if (affine_3d.has_value() && affine_3d.value().size(0) == 1) {
            broadcast_affine_3d = true;
        } else if (affine_3d.has_value() && affine_3d.value().size(0) != batch_size_max) {  
            TORCH_CHECK(false, "affine_3d batch size must match batch size of input or grid");
        }
        
        // broadcast grid if it exists
        if (grid.size(0) == 1) {
            broadcast_grid = true;
        } else if (grid.size(0) != batch_size_max) {
            TORCH_CHECK(false, "grid batch size must match batch size of input or affine_3d");
        }
    }

    // determine if we need to compute gradients
    bool input_requires_grad = grad_input.has_value();
    bool affine_requires_grad = grad_affine.has_value() && affine_3d.has_value();
    bool grid_requires_grad = grad_grid.has_value();
    // if grid is provided and it is a displacement, there are no gradients w.r.t. affine
    if (!grid_requires_grad && !affine_requires_grad && !input_requires_grad) {
        // nothing to compute
        return;
    }

    // D, H, W will be determined by grid
    TORCH_CHECK(grid.is_contiguous(), "grid must be contiguous");
    TORCH_CHECK(grid.size(4) == 3, "grid must have 3 channels");
    int64_t D = grid.size(1);
    int64_t H = grid.size(2);
    int64_t W = grid.size(3);

    if (affine_3d.has_value()) {
        // TORCH_CHECK(input.size(0) == affine_3d.value().size(0), "input and affine_3d must have the same batch size");
        TORCH_CHECK(affine_3d.value().dim() == 3, "affine_3d must be (B, 3, 4)");
        TORCH_CHECK(affine_3d.value().device().is_cuda(), "affine_3d must be on CUDA");
        TORCH_CHECK(affine_3d.value().is_contiguous(), "affine_3d must be contiguous");
        TORCH_CHECK(affine_3d.value().size(1) == 3, "affine_3d must be (B, 3, 4)");
        TORCH_CHECK(affine_3d.value().size(2) == 4, "affine_3d must be (B, 3, 4)");
    }

    // define output
    int64_t N = batch_size_max;
    // input size parameters (put batch in a separate dimension)
    int64_t count = D * H * W;

    // input spatial size parameters
    int64_t Di = input.size(1);
    int64_t Hi = input.size(2);
    int64_t Wi = input.size(3);
    TORCH_CHECK(input.size(4) == 3, "last dimension should be 3");

    // initialize grid and dim
    dim3 blockSize3(BLOCKSIZE_3D, 1, 1);
    int64_t gridSize = GET_BLOCKS(count, BLOCKSIZE_3D);
    gridSize = std::min(gridSize, static_cast<int64_t>(65536));
    dim3 gridSize3(gridSize, batch_size_max, 1);

    // intermediate grad affine collector
    torch::Tensor grad_affine_collect;
    if (affine_requires_grad) {
        grad_affine_collect = torch::zeros({affine_3d.value().size(0), gridSize, 3, 4}, grad_affine.value().options());
    }

    if (count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "fused_grid_composer_3d_backward_kernel", [&] {
            // check if grid is 32-bit
            if (canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
                canUse32BitIndexMath(grad_output)) {
                fused_grid_composer_3d_backward_kernel<scalar_t>
                <<<gridSize3, blockSize3, 0, stream>>>(
                    static_cast<int>(count),
                    input.data_ptr<scalar_t>(),
                    grid.data_ptr<scalar_t>(),
                    affine_3d.has_value() ? affine_3d.value().data_ptr<scalar_t>() : nullptr,
                    // grads
                    grad_output.data_ptr<scalar_t>(),
                    input_requires_grad ? grad_input.value().data_ptr<scalar_t>() : nullptr,
                    affine_requires_grad ? grad_affine_collect.data_ptr<scalar_t>() : nullptr,
                    grid_requires_grad ? grad_grid.value().data_ptr<scalar_t>() : nullptr,
                    // input size parameters
                    static_cast<int>(N), 
                    static_cast<int>(Di), static_cast<int>(Hi), static_cast<int>(Wi),
                    static_cast<int>(D), static_cast<int>(H), static_cast<int>(W),
                    grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
                    align_corners,
                    broadcast_input,
                    broadcast_affine_3d,
                    broadcast_grid
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            } else {
                fused_grid_composer_3d_backward_kernel<scalar_t>
                <<<gridSize3, blockSize3, 0, stream>>>(
                    count,
                    input.data_ptr<scalar_t>(),
                    grid.data_ptr<scalar_t>(),
                    affine_3d.has_value() ? affine_3d.value().data_ptr<scalar_t>() : nullptr,
                    // grads
                    grad_output.data_ptr<scalar_t>(),
                    input_requires_grad ? grad_input.value().data_ptr<scalar_t>() : nullptr,
                    affine_requires_grad ? grad_affine_collect.data_ptr<scalar_t>() : nullptr,
                    grid_requires_grad ? grad_grid.value().data_ptr<scalar_t>() : nullptr,
                    // input size parameters
                    N, 
                    Di, Hi, Wi,
                    D, H, W,
                    grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
                    align_corners,
                    broadcast_input,
                    broadcast_affine_3d,
                    broadcast_grid
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        });
    }

    if (affine_requires_grad) {
        // sum over the batch dimension
        grad_affine.value().copy_(grad_affine_collect.sum(1));
    }
}
