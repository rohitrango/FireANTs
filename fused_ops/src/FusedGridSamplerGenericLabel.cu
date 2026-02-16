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

#include <ATen/OpMathType.h>
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/Dispatch.h>
#include <c10/macros/Macros.h>
#include <torch/torch.h>
#include <cmath>
#include "common.h"
#include "FusedGridSamplerGenericLabel.h"

using namespace at::cuda::detail;
using namespace at::native;
using at::native::detail::GridSamplerPadding;

#define BLOCKSIZE_3D 512

inline void check_grid_sampler_common_v2(
    const torch::Tensor& input,
    const torch::Tensor& grid) {
  TORCH_CHECK(input.defined(), "grid_sampler(): expected input to not be undefined");
  TORCH_CHECK(grid.defined(), "grid_sampler(): expected grid to not be undefined");
  TORCH_CHECK(
      input.options().device() == grid.options().device(),
      "grid_sampler(): expected input and grid on same device");
  TORCH_CHECK(
      grid.size(-1) == input.dim() - 2,
      "grid_sampler(): grid last dim must match input spatial dims");
  for (int i = 2; i < input.dim(); i++) {
    TORCH_CHECK(input.size(i) > 0, "grid_sampler(): non-empty spatial dimensions required");
  }
}

// --- 2D forward: accumulate distinct labels and weights, output argmax label + optional weight ---
template <typename scalar_t, typename grid_t, typename index_t>
C10_LAUNCH_BOUNDS_1(BLOCKSIZE_3D)
__global__ void fused_grid_sampler_2d_generic_label_forward_kernel(
    const index_t count,
    const scalar_t* input,
    const grid_t* grid,
    const grid_t* affine_2d,
    const grid_t* affine_2d_pregrid,
    const index_t N,
    const index_t C,
    const index_t Hi,
    const index_t Wi,
    const index_t H,
    const index_t W,
    const float grid_xmin,
    const float grid_ymin,
    const float grid_xmax,
    const float grid_ymax,
    const bool is_displacement,
    const scalar_t dummy,
    scalar_t* output_labels,
    scalar_t* output_weights,  // may be null
    const bool return_weight,
    const GridSamplerPadding padding_mode,
    const bool align_corners,
    const bool broadcast_input,
    const bool broadcast_affine_2d,
    const bool broadcast_grid,
    const bool broadcast_affine_2d_pregrid) {

  using gridmath_t = at::opmath_type<grid_t>;

  CUDA_KERNEL_LOOP_TYPE(index, count, index_t) {
    const index_t w = index % W;
    const index_t h = (index / W) % H;
    const index_t n = index / (H * W);
    const index_t grid_offset = 2 * (w + W * (h + H * (broadcast_grid ? 0 : n)));

    gridmath_t ix = 0, iy = 0;
    gridmath_t x = 0, y = 0;

    if (!grid) {
      const grid_t* affine_2d_ptr = affine_2d + (broadcast_affine_2d ? 0 : (6 * n));
      ix = w * (grid_xmax - grid_xmin) / (W - 1) + grid_xmin;
      iy = h * (grid_ymax - grid_ymin) / (H - 1) + grid_ymin;
      x = affine_2d_ptr[0] * ix + affine_2d_ptr[1] * iy + affine_2d_ptr[2];
      y = affine_2d_ptr[3] * ix + affine_2d_ptr[4] * iy + affine_2d_ptr[5];
    } else {
      grid_t dx = grid[grid_offset];
      grid_t dy = grid[grid_offset + 1];
      grid_t i_dx, i_dy;
      if (affine_2d_pregrid) {
        const grid_t* pregrid_ptr = affine_2d_pregrid + (broadcast_affine_2d_pregrid ? 0 : (4 * n));
        i_dx = pregrid_ptr[0] * dx + pregrid_ptr[1] * dy;
        i_dy = pregrid_ptr[2] * dx + pregrid_ptr[3] * dy;
      } else {
        i_dx = dx;
        i_dy = dy;
      }
      if (is_displacement) {
        ix = w * (grid_xmax - grid_xmin) / (W - 1) + grid_xmin;
        iy = h * (grid_ymax - grid_ymin) / (H - 1) + grid_ymin;
        if (affine_2d) {
          const grid_t* affine_2d_ptr = affine_2d + (broadcast_affine_2d ? 0 : (6 * n));
          x = affine_2d_ptr[0] * ix + affine_2d_ptr[1] * iy + affine_2d_ptr[2];
          y = affine_2d_ptr[3] * ix + affine_2d_ptr[4] * iy + affine_2d_ptr[5];
        } else {
          x = ix;
          y = iy;
        }
        x += i_dx;
        y += i_dy;
      } else {
        x = i_dx;
        y = i_dy;
      }
    }

    ix = grid_sampler_compute_source_index(x, Wi, padding_mode, align_corners);
    iy = grid_sampler_compute_source_index(y, Hi, padding_mode, align_corners);

    index_t ix_w = static_cast<index_t>(::floor(ix));
    index_t iy_n = static_cast<index_t>(::floor(iy));
    index_t ix_e = ix_w + 1;
    index_t iy_s = iy_n + 1;

    scalar_t nw = (ix_e - ix) * (iy_s - iy);
    scalar_t ne = (ix - ix_w) * (iy_s - iy);
    scalar_t sw = (ix_e - ix) * (iy - iy_n);
    scalar_t se = (ix - ix_w) * (iy - iy_n);

    const index_t inp_sC = Hi * Wi;
    const scalar_t* inp_ptr_NC = input + (broadcast_input ? 0 : (n * (C * inp_sC)));

    // At most 4 distinct values in 2D
    scalar_t val_list[4];
    scalar_t weight_list[4];
    int nvals = 0;

    auto add_or_accumulate = [&](scalar_t val, scalar_t w, scalar_t dum) {
      // ignore values less than dummy-1 (background label)
      if (val < dum) return;
      for (int i = 0; i < nvals; i++) {
        if (fabsf(val_list[i] - val) < 1e-5) {
          weight_list[i] += w;
          return;
        }
      }
      if (nvals < 4) {
        val_list[nvals] = val;
        weight_list[nvals] = w;
        nvals++;
      }
    };

    for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC) {
      scalar_t v_nw = within_bounds_2d(iy_n, ix_w, Hi, Wi) ? inp_ptr_NC[ix_w + Wi * iy_n] : dummy - 1;
      scalar_t v_ne = within_bounds_2d(iy_n, ix_e, Hi, Wi) ? inp_ptr_NC[ix_e + Wi * iy_n] : dummy - 1;
      scalar_t v_sw = within_bounds_2d(iy_s, ix_w, Hi, Wi) ? inp_ptr_NC[ix_w + Wi * iy_s] : dummy - 1;
      scalar_t v_se = within_bounds_2d(iy_s, ix_e, Hi, Wi) ? inp_ptr_NC[ix_e + Wi * iy_s] : dummy - 1;

      nvals = 0;
      add_or_accumulate(v_nw, nw, dummy);
      add_or_accumulate(v_ne, ne, dummy);
      add_or_accumulate(v_sw, sw, dummy);
      add_or_accumulate(v_se, se, dummy);

      scalar_t best_i;
      scalar_t best_w;

      // if there was any valid point, find the best one
      if (nvals > 0) {
        best_i = val_list[0];
        best_w = weight_list[0];
        for (int i = 1; i < nvals; i++) {
          if (weight_list[i] > best_w) {
            best_w = weight_list[i];
            best_i = val_list[i];
          }
        }
      }
      else {
        // all points were outside the image, set to background label
        best_w = 0.0;
        best_i = dummy;
      }

      index_t out_idx = w + W * (h + H * (c + C * n));
      output_labels[out_idx] = best_i;
      if (return_weight && output_weights) {
        output_weights[out_idx] = best_w;
      }
    }
  }
}

// --- 2D backward: grad_weight drives grid/affine only; no grad_input (label maps have no gradient) ---
template <typename scalar_t, typename grid_t, typename index_t>
C10_LAUNCH_BOUNDS_1(BLOCKSIZE_3D)
__global__ void fused_grid_sampler_2d_generic_label_backward_kernel(
    const index_t count,
    const scalar_t* input,
    const scalar_t* output_labels,
    const grid_t* grid,
    const grid_t* affine_2d,
    const grid_t* affine_2d_pregrid,
    const scalar_t* grad_weight,
    grid_t* grad_affine_collect,
    grid_t* grad_grid,
    const index_t N,
    const index_t C,
    const index_t Hi,
    const index_t Wi,
    const index_t H,
    const index_t W,
    const float grid_xmin,
    const float grid_ymin,
    const float grid_xmax,
    const float grid_ymax,
    const bool is_displacement,
    const bool return_weight,
    const GridSamplerPadding padding_mode,
    const bool align_corners,
    const bool broadcast_input,
    const bool broadcast_affine_2d,
    const bool broadcast_grid,
    const bool broadcast_affine_2d_pregrid) {

  const index_t n = blockIdx.y;
  using gridmath_t = at::opmath_type<grid_t>;

  gridmath_t _affine_grad_[6];
  for (int i = 0; i < 6; ++i) _affine_grad_[i] = 0;

  __shared__ gridmath_t _affine_grad_shared_[BLOCKSIZE_3D];
  gridmath_t _affine_map_[6];
  if (affine_2d) {
    const index_t offset = broadcast_affine_2d ? 0 : (6 * n);
    for (int i = 0; i < 6; ++i) _affine_map_[i] = affine_2d[offset + i];
  }

  CUDA_KERNEL_LOOP_TYPE(index, count, index_t) {
    const index_t w = index % W;
    const index_t h = (index / W) % H;
    const index_t grid_offset = 2 * (w + W * (h + H * (broadcast_grid ? 0 : n)));

    gridmath_t pax = 0, pay = 0;
    gridmath_t ix, iy;
    gridmath_t x, y;

    if (!grid) {
      ix = w * (grid_xmax - grid_xmin) / (W - 1) + grid_xmin;
      iy = h * (grid_ymax - grid_ymin) / (H - 1) + grid_ymin;
      pax = ix;
      pay = iy;
      x = _affine_map_[0] * ix + _affine_map_[1] * iy + _affine_map_[2];
      y = _affine_map_[3] * ix + _affine_map_[4] * iy + _affine_map_[5];
    } else {
      grid_t dx = grid[grid_offset];
      grid_t dy = grid[grid_offset + 1];
      grid_t i_dx, i_dy;
      if (affine_2d_pregrid) {
        const grid_t* pregrid_ptr = affine_2d_pregrid + (broadcast_affine_2d_pregrid ? 0 : (4 * n));
        i_dx = pregrid_ptr[0] * dx + pregrid_ptr[1] * dy;
        i_dy = pregrid_ptr[2] * dx + pregrid_ptr[3] * dy;
      } else {
        i_dx = dx;
        i_dy = dy;
      }
      if (is_displacement) {
        ix = w * (grid_xmax - grid_xmin) / (W - 1) + grid_xmin;
        iy = h * (grid_ymax - grid_ymin) / (H - 1) + grid_ymin;
        pax = ix;
        pay = iy;
        if (affine_2d) {
          x = _affine_map_[0] * ix + _affine_map_[1] * iy + _affine_map_[2];
          y = _affine_map_[3] * ix + _affine_map_[4] * iy + _affine_map_[5];
        } else {
          x = ix;
          y = iy;
        }
        x += i_dx;
        y += i_dy;
      } else {
        x = i_dx;
        y = i_dy;
      }
    }

    gridmath_t gix_mult, giy_mult;
    ix = grid_sampler_compute_source_index_set_grad(x, Wi, padding_mode, align_corners, &gix_mult);
    iy = grid_sampler_compute_source_index_set_grad(y, Hi, padding_mode, align_corners, &giy_mult);

    index_t ix_w = static_cast<index_t>(std::floor(ix));
    index_t iy_n = static_cast<index_t>(std::floor(iy));
    index_t ix_e = ix_w + 1;
    index_t iy_s = iy_n + 1;

    gridmath_t nw = (ix_e - ix) * (iy_s - iy);
    gridmath_t ne = (ix - ix_w) * (iy_s - iy);
    gridmath_t sw = (ix_e - ix) * (iy - iy_n);
    gridmath_t se = (ix - ix_w) * (iy - iy_n);

    const index_t inp_sC = Hi * Wi;

    for (index_t c = 0; c < C; ++c) {
      index_t out_idx = w + W * (h + H * (c + C * n));
      scalar_t out_label = output_labels[out_idx];
      index_t NC_offset = (broadcast_input ? 0 : (n * C * Hi * Wi)) + c * Hi * Wi;
      const scalar_t* inp_ptr_NC = input + (broadcast_input ? 0 : (n * C * Hi * Wi)) + c * Hi * Wi;

      scalar_t gOut = (return_weight && grad_weight) ? grad_weight[out_idx] : static_cast<scalar_t>(0);

      gridmath_t gix = 0, giy = 0;
      if (grad_affine_collect || grad_grid) {
        gridmath_t gOutGMT = static_cast<gridmath_t>(gOut);
        if (within_bounds_2d(iy_n, ix_w, Hi, Wi)) {
          scalar_t v = inp_ptr_NC[ix_w + Wi * iy_n];
          if (v == out_label) {
            gix -= (iy_s - iy) * gOutGMT;
            giy -= (ix_e - ix) * gOutGMT;
          }
        }
        if (within_bounds_2d(iy_n, ix_e, Hi, Wi)) {
          scalar_t v = inp_ptr_NC[ix_e + Wi * iy_n];
          if (v == out_label) {
            gix += (iy_s - iy) * gOutGMT;
            giy -= (ix - ix_w) * gOutGMT;
          }
        }
        if (within_bounds_2d(iy_s, ix_w, Hi, Wi)) {
          scalar_t v = inp_ptr_NC[ix_w + Wi * iy_s];
          if (v == out_label) {
            gix -= (iy - iy_n) * gOutGMT;
            giy += (ix_e - ix) * gOutGMT;
          }
        }
        if (within_bounds_2d(iy_s, ix_e, Hi, Wi)) {
          scalar_t v = inp_ptr_NC[ix_e + Wi * iy_s];
          if (v == out_label) {
            gix += (iy - iy_n) * gOutGMT;
            giy += (ix - ix_w) * gOutGMT;
          }
        }
        gix = gix_mult * gix;
        giy = giy_mult * giy;

        if (grad_affine_collect) {
          _affine_grad_[0] += gix * pax;
          _affine_grad_[1] += gix * pay;
          _affine_grad_[2] += gix;
          _affine_grad_[3] += giy * pax;
          _affine_grad_[4] += giy * pay;
          _affine_grad_[5] += giy;
        }
        if (affine_2d_pregrid) {
          const grid_t* pregrid_ptr = affine_2d_pregrid + (broadcast_affine_2d_pregrid ? 0 : (4 * n));
          gridmath_t pax2 = gix * pregrid_ptr[0] + giy * pregrid_ptr[2];
          gridmath_t pay2 = gix * pregrid_ptr[1] + giy * pregrid_ptr[3];
          gix = pax2;
          giy = pay2;
        }
        if (grad_grid) {
          if (broadcast_grid) {
            index_t gi = 2 * (w + W * h);
            gpuAtomicAdd(grad_grid + gi, static_cast<grid_t>(gix));
            gpuAtomicAdd(grad_grid + gi + 1, static_cast<grid_t>(giy));
          } else {
            index_t gi = 2 * (w + W * (h + H * n));
            grad_grid[gi] += static_cast<grid_t>(gix);
            grad_grid[gi + 1] += static_cast<grid_t>(giy);
          }
        }
      }
    }
  }

  if (grad_affine_collect) {
    for (int affid = 0; affid < 6; ++affid) {
      _affine_grad_shared_[threadIdx.x] = _affine_grad_[affid];
      __syncthreads();
      for (int tid = BLOCKSIZE_3D / 2; tid > 0; tid /= 2) {
        if (threadIdx.x < tid)
          _affine_grad_shared_[threadIdx.x] += _affine_grad_shared_[threadIdx.x + tid];
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        if (broadcast_affine_2d) {
          gpuAtomicAdd(grad_affine_collect + blockIdx.x * 6 + affid, static_cast<grid_t>(_affine_grad_shared_[0]));
        } else {
          grad_affine_collect[affid + 6 * (blockIdx.x + gridDim.x * n)] = static_cast<grid_t>(_affine_grad_shared_[0]);
        }
      }
      __syncthreads();
    }
  }
}

// --- 3D forward: 8 neighbors, at most 8 distinct values ---
template <typename scalar_t, typename grid_t, typename index_t>
C10_LAUNCH_BOUNDS_1(BLOCKSIZE_3D)
__global__ void fused_grid_sampler_3d_generic_label_forward_kernel(
    const index_t count,
    const scalar_t* input,
    const grid_t* grid,
    const grid_t* affine_3d,
    const grid_t* affine_3d_pregrid,
    const index_t N,
    const index_t C,
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
    const bool is_displacement,
    const scalar_t dummy,
    scalar_t* output_labels,
    scalar_t* output_weights,
    const bool return_weight,
    const GridSamplerPadding padding_mode,
    const bool align_corners,
    const bool broadcast_input,
    const bool broadcast_affine_3d,
    const bool broadcast_grid,
    const bool broadcast_affine_3d_pregrid) {

  using gridmath_t = at::opmath_type<grid_t>;

  CUDA_KERNEL_LOOP_TYPE(index, count, index_t) {
    const index_t w = index % W;
    const index_t h = (index / W) % H;
    const index_t d = (index / (H * W)) % D;
    const index_t n = index / (D * H * W);
    const index_t grid_offset = 3 * (w + W * (h + H * (d + (broadcast_grid ? 0 : (D * n)))));

    gridmath_t ix = 0, iy = 0, iz = 0;
    gridmath_t x = 0, y = 0, z = 0;

    if (!grid) {
      const grid_t* aff = affine_3d + (broadcast_affine_3d ? 0 : (12 * n));
      ix = w * (grid_xmax - grid_xmin) / (W - 1) + grid_xmin;
      iy = h * (grid_ymax - grid_ymin) / (H - 1) + grid_ymin;
      iz = d * (grid_zmax - grid_zmin) / (D - 1) + grid_zmin;
      x = aff[0] * ix + aff[1] * iy + aff[2] * iz + aff[3];
      y = aff[4] * ix + aff[5] * iy + aff[6] * iz + aff[7];
      z = aff[8] * ix + aff[9] * iy + aff[10] * iz + aff[11];
    } else {
      grid_t dx = grid[grid_offset];
      grid_t dy = grid[grid_offset + 1];
      grid_t dz = grid[grid_offset + 2];
      grid_t i_dx, i_dy, i_dz;
      if (affine_3d_pregrid) {
        const grid_t* pre = affine_3d_pregrid + (broadcast_affine_3d_pregrid ? 0 : (9 * n));
        i_dx = pre[0] * dx + pre[1] * dy + pre[2] * dz;
        i_dy = pre[3] * dx + pre[4] * dy + pre[5] * dz;
        i_dz = pre[6] * dx + pre[7] * dy + pre[8] * dz;
      } else {
        i_dx = dx;
        i_dy = dy;
        i_dz = dz;
      }
      if (is_displacement) {
        ix = w * (grid_xmax - grid_xmin) / (W - 1) + grid_xmin;
        iy = h * (grid_ymax - grid_ymin) / (H - 1) + grid_ymin;
        iz = d * (grid_zmax - grid_zmin) / (D - 1) + grid_zmin;
        if (affine_3d) {
          const grid_t* aff = affine_3d + (broadcast_affine_3d ? 0 : (12 * n));
          x = aff[0] * ix + aff[1] * iy + aff[2] * iz + aff[3];
          y = aff[4] * ix + aff[5] * iy + aff[6] * iz + aff[7];
          z = aff[8] * ix + aff[9] * iy + aff[10] * iz + aff[11];
        } else {
          x = ix;
          y = iy;
          z = iz;
        }
        x += i_dx;
        y += i_dy;
        z += i_dz;
      } else {
        x = i_dx;
        y = i_dy;
        z = i_dz;
      }
    }

    ix = grid_sampler_compute_source_index(x, Wi, padding_mode, align_corners);
    iy = grid_sampler_compute_source_index(y, Hi, padding_mode, align_corners);
    iz = grid_sampler_compute_source_index(z, Di, padding_mode, align_corners);

    index_t ix_tnw = static_cast<index_t>(::floor(ix));
    index_t iy_tnw = static_cast<index_t>(::floor(iy));
    index_t iz_tnw = static_cast<index_t>(::floor(iz));
    index_t ix_tne = ix_tnw + 1, iy_tne = iy_tnw, iz_tne = iz_tnw;
    index_t ix_tsw = ix_tnw, iy_tsw = iy_tnw + 1, iz_tsw = iz_tnw;
    index_t ix_tse = ix_tnw + 1, iy_tse = iy_tnw + 1, iz_tse = iz_tnw;
    index_t ix_bnw = ix_tnw, iy_bnw = iy_tnw, iz_bnw = iz_tnw + 1;
    index_t ix_bne = ix_tnw + 1, iy_bne = iy_tnw, iz_bne = iz_tnw + 1;
    index_t ix_bsw = ix_tnw, iy_bsw = iy_tnw + 1, iz_bsw = iz_tnw + 1;
    index_t ix_bse = ix_tnw + 1, iy_bse = iy_tnw + 1, iz_bse = iz_tnw + 1;

    scalar_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    scalar_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    scalar_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    scalar_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    scalar_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    scalar_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    scalar_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    scalar_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

    const index_t inp_sC = Di * Hi * Wi;
    const scalar_t* inp_ptr_NC = input + (broadcast_input ? 0 : (n * (C * inp_sC)));

    scalar_t val_list[8];
    scalar_t weight_list[8];
    int nvals = 0;

    auto add_or_accumulate = [&](scalar_t val, scalar_t w, scalar_t dum) {
      // ignore values less than dummy-1 (background label)
      if (val < dum) return;
      for (int i = 0; i < nvals; i++) {
        if (fabsf(val_list[i] - val) < 1e-5) {
          weight_list[i] += w;
          return;
        }
      }
      if (nvals < 8) {
        val_list[nvals] = val;
        weight_list[nvals] = w;
        nvals++;
      }
    };

    for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC) {
      scalar_t v_tnw = within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, Di, Hi, Wi) ? inp_ptr_NC[ix_tnw + Wi * (iy_tnw + Hi * iz_tnw)] : dummy - 1;
      scalar_t v_tne = within_bounds_3d(iz_tne, iy_tne, ix_tne, Di, Hi, Wi) ? inp_ptr_NC[ix_tne + Wi * (iy_tne + Hi * iz_tne)] : dummy - 1;
      scalar_t v_tsw = within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, Di, Hi, Wi) ? inp_ptr_NC[ix_tsw + Wi * (iy_tsw + Hi * iz_tsw)] : dummy - 1;
      scalar_t v_tse = within_bounds_3d(iz_tse, iy_tse, ix_tse, Di, Hi, Wi) ? inp_ptr_NC[ix_tse + Wi * (iy_tse + Hi * iz_tse)] : dummy - 1;
      scalar_t v_bnw = within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, Di, Hi, Wi) ? inp_ptr_NC[ix_bnw + Wi * (iy_bnw + Hi * iz_bnw)] : dummy - 1;
      scalar_t v_bne = within_bounds_3d(iz_bne, iy_bne, ix_bne, Di, Hi, Wi) ? inp_ptr_NC[ix_bne + Wi * (iy_bne + Hi * iz_bne)] : dummy - 1;
      scalar_t v_bsw = within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, Di, Hi, Wi) ? inp_ptr_NC[ix_bsw + Wi * (iy_bsw + Hi * iz_bsw)] : dummy - 1;
      scalar_t v_bse = within_bounds_3d(iz_bse, iy_bse, ix_bse, Di, Hi, Wi) ? inp_ptr_NC[ix_bse + Wi * (iy_bse + Hi * iz_bse)] : dummy - 1;

      nvals = 0;
      add_or_accumulate(v_tnw, tnw, dummy);
      add_or_accumulate(v_tne, tne, dummy);
      add_or_accumulate(v_tsw, tsw, dummy);
      add_or_accumulate(v_tse, tse, dummy);
      add_or_accumulate(v_bnw, bnw, dummy);
      add_or_accumulate(v_bne, bne, dummy);
      add_or_accumulate(v_bsw, bsw, dummy);
      add_or_accumulate(v_bse, bse, dummy);

      scalar_t best_i;
      scalar_t best_w;

      // if there was any valid point, find the best one
      if (nvals > 0) {
        best_i = val_list[0];
        best_w = weight_list[0];
        for (int i = 1; i < nvals; i++) {
          if (weight_list[i] > best_w) {
            best_w = weight_list[i];
            best_i = val_list[i];
          }
        }
      }
      else {
        // all points were outside the image, set to background label
        best_w = 0.0;
        best_i = dummy;
      }

      index_t out_sC = D * H * W;
      index_t out_idx = w + W * (h + H * (d + D * (c + C * n)));
      output_labels[out_idx] = best_i;
      if (return_weight && output_weights) {
        output_weights[out_idx] = best_w;
      }
    }
  }
}

// --- 3D backward ---
template <typename scalar_t, typename grid_t, typename index_t>
C10_LAUNCH_BOUNDS_1(BLOCKSIZE_3D)
__global__ void fused_grid_sampler_3d_generic_label_backward_kernel(
    const index_t count,
    const scalar_t* input,
    const scalar_t* output_labels,
    const grid_t* grid,
    const grid_t* affine_3d,
    const grid_t* affine_3d_pregrid,
    const scalar_t* grad_weight,
    grid_t* grad_affine_collect,
    grid_t* grad_grid,
    const index_t N,
    const index_t C,
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
    const bool is_displacement,
    const bool return_weight,
    const GridSamplerPadding padding_mode,
    const bool align_corners,
    const bool broadcast_input,
    const bool broadcast_affine_3d,
    const bool broadcast_grid,
    const bool broadcast_affine_3d_pregrid) {

  const index_t n = blockIdx.y;
  using gridmath_t = at::opmath_type<grid_t>;

  gridmath_t _affine_grad_[12];
  for (int i = 0; i < 12; ++i) _affine_grad_[i] = 0;
  __shared__ gridmath_t _affine_grad_shared_[BLOCKSIZE_3D];
  gridmath_t _affine_map_[12];
  if (affine_3d) {
    const index_t off = broadcast_affine_3d ? 0 : (12 * n);
    for (int i = 0; i < 12; ++i) _affine_map_[i] = affine_3d[off + i];
  }

  CUDA_KERNEL_LOOP_TYPE(index, count, index_t) {
    const index_t w = index % W;
    const index_t h = (index / W) % H;
    const index_t d = (index / (H * W)) % D;
    const index_t grid_offset = 3 * (w + W * (h + H * (d + (broadcast_grid ? 0 : (D * n)))));

    gridmath_t pax = 0, pay = 0, paz = 0;
    gridmath_t ix, iy, iz;
    gridmath_t x, y, z;

    if (!grid) {
      ix = w * (grid_xmax - grid_xmin) / (W - 1) + grid_xmin;
      iy = h * (grid_ymax - grid_ymin) / (H - 1) + grid_ymin;
      iz = d * (grid_zmax - grid_zmin) / (D - 1) + grid_zmin;
      pax = ix;
      pay = iy;
      paz = iz;
      x = _affine_map_[0] * ix + _affine_map_[1] * iy + _affine_map_[2] * iz + _affine_map_[3];
      y = _affine_map_[4] * ix + _affine_map_[5] * iy + _affine_map_[6] * iz + _affine_map_[7];
      z = _affine_map_[8] * ix + _affine_map_[9] * iy + _affine_map_[10] * iz + _affine_map_[11];
    } else {
      grid_t dx = grid[grid_offset];
      grid_t dy = grid[grid_offset + 1];
      grid_t dz = grid[grid_offset + 2];
      grid_t i_dx, i_dy, i_dz;
      if (affine_3d_pregrid) {
        const grid_t* pre = affine_3d_pregrid + (broadcast_affine_3d_pregrid ? 0 : (9 * n));
        i_dx = pre[0] * dx + pre[1] * dy + pre[2] * dz;
        i_dy = pre[3] * dx + pre[4] * dy + pre[5] * dz;
        i_dz = pre[6] * dx + pre[7] * dy + pre[8] * dz;
      } else {
        i_dx = dx;
        i_dy = dy;
        i_dz = dz;
      }
      if (is_displacement) {
        ix = w * (grid_xmax - grid_xmin) / (W - 1) + grid_xmin;
        iy = h * (grid_ymax - grid_ymin) / (H - 1) + grid_ymin;
        iz = d * (grid_zmax - grid_zmin) / (D - 1) + grid_zmin;
        pax = ix;
        pay = iy;
        paz = iz;
        if (affine_3d) {
          x = _affine_map_[0] * ix + _affine_map_[1] * iy + _affine_map_[2] * iz + _affine_map_[3];
          y = _affine_map_[4] * ix + _affine_map_[5] * iy + _affine_map_[6] * iz + _affine_map_[7];
          z = _affine_map_[8] * ix + _affine_map_[9] * iy + _affine_map_[10] * iz + _affine_map_[11];
        } else {
          x = ix;
          y = iy;
          z = iz;
        }
        x += i_dx;
        y += i_dy;
        z += i_dz;
      } else {
        x = i_dx;
        y = i_dy;
        z = i_dz;
      }
    }

    gridmath_t gix_mult, giy_mult, giz_mult;
    ix = grid_sampler_compute_source_index_set_grad(x, Wi, padding_mode, align_corners, &gix_mult);
    iy = grid_sampler_compute_source_index_set_grad(y, Hi, padding_mode, align_corners, &giy_mult);
    iz = grid_sampler_compute_source_index_set_grad(z, Di, padding_mode, align_corners, &giz_mult);

    index_t ix_tnw = static_cast<index_t>(std::floor(ix));
    index_t iy_tnw = static_cast<index_t>(std::floor(iy));
    index_t iz_tnw = static_cast<index_t>(std::floor(iz));
    index_t ix_tne = ix_tnw + 1, iy_tne = iy_tnw, iz_tne = iz_tnw;
    index_t ix_tsw = ix_tnw, iy_tsw = iy_tnw + 1, iz_tsw = iz_tnw;
    index_t ix_tse = ix_tnw + 1, iy_tse = iy_tnw + 1, iz_tse = iz_tnw;
    index_t ix_bnw = ix_tnw, iy_bnw = iy_tnw, iz_bnw = iz_tnw + 1;
    index_t ix_bne = ix_tnw + 1, iy_bne = iy_tnw, iz_bne = iz_tnw + 1;
    index_t ix_bsw = ix_tnw, iy_bsw = iy_tnw + 1, iz_bsw = iz_tnw + 1;
    index_t ix_bse = ix_tnw + 1, iy_bse = iy_tnw + 1, iz_bse = iz_tnw + 1;

    gridmath_t tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    gridmath_t tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    gridmath_t tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    gridmath_t tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    gridmath_t bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    gridmath_t bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    gridmath_t bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    gridmath_t bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

    const index_t inp_sC = Di * Hi * Wi;

    for (index_t c = 0; c < C; ++c) {
      index_t out_idx = w + W * (h + H * (d + D * (c + C * n)));
      scalar_t out_label = output_labels[out_idx];
      scalar_t gOut = (return_weight && grad_weight) ? grad_weight[out_idx] : static_cast<scalar_t>(0);
      const scalar_t* inp_ptr_NC = input + (broadcast_input ? 0 : (n * C * Di * Hi * Wi)) + c * Di * Hi * Wi;

      gridmath_t gix = 0, giy = 0, giz = 0;
      if (grad_affine_collect || grad_grid) {
        gridmath_t gOutGMT = static_cast<gridmath_t>(gOut);
        if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, Di, Hi, Wi)) {
          scalar_t v = inp_ptr_NC[ix_tnw + Wi * (iy_tnw + Hi * iz_tnw)];
          if (v == out_label) {
            gix -= (iy_bse - iy) * (iz_bse - iz) * gOutGMT;
            giy -= (ix_bse - ix) * (iz_bse - iz) * gOutGMT;
            giz -= (ix_bse - ix) * (iy_bse - iy) * gOutGMT;
          }
        }
        if (within_bounds_3d(iz_tne, iy_tne, ix_tne, Di, Hi, Wi)) {
          scalar_t v = inp_ptr_NC[ix_tne + Wi * (iy_tne + Hi * iz_tne)];
          if (v == out_label) {
            gix += (iy_bsw - iy) * (iz_bsw - iz) * gOutGMT;
            giy -= (ix - ix_bsw) * (iz_bsw - iz) * gOutGMT;
            giz -= (ix - ix_bsw) * (iy_bsw - iy) * gOutGMT;
          }
        }
        if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, Di, Hi, Wi)) {
          scalar_t v = inp_ptr_NC[ix_tsw + Wi * (iy_tsw + Hi * iz_tsw)];
          if (v == out_label) {
            gix -= (iy - iy_bne) * (iz_bne - iz) * gOutGMT;
            giy += (ix_bne - ix) * (iz_bne - iz) * gOutGMT;
            giz -= (ix_bne - ix) * (iy - iy_bne) * gOutGMT;
          }
        }
        if (within_bounds_3d(iz_tse, iy_tse, ix_tse, Di, Hi, Wi)) {
          scalar_t v = inp_ptr_NC[ix_tse + Wi * (iy_tse + Hi * iz_tse)];
          if (v == out_label) {
            gix += (iy - iy_bnw) * (iz_bnw - iz) * gOutGMT;
            giy += (ix - ix_bnw) * (iz_bnw - iz) * gOutGMT;
            giz -= (ix - ix_bnw) * (iy - iy_bnw) * gOutGMT;
          }
        }
        if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, Di, Hi, Wi)) {
          scalar_t v = inp_ptr_NC[ix_bnw + Wi * (iy_bnw + Hi * iz_bnw)];
          if (v == out_label) {
            gix -= (iy_tse - iy) * (iz - iz_tse) * gOutGMT;
            giy -= (ix_tse - ix) * (iz - iz_tse) * gOutGMT;
            giz += (ix_tse - ix) * (iy_tse - iy) * gOutGMT;
          }
        }
        if (within_bounds_3d(iz_bne, iy_bne, ix_bne, Di, Hi, Wi)) {
          scalar_t v = inp_ptr_NC[ix_bne + Wi * (iy_bne + Hi * iz_bne)];
          if (v == out_label) {
            gix += (iy_tsw - iy) * (iz - iz_tsw) * gOutGMT;
            giy -= (ix - ix_tsw) * (iz - iz_tsw) * gOutGMT;
            giz += (ix - ix_tsw) * (iy_tsw - iy) * gOutGMT;
          }
        }
        if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, Di, Hi, Wi)) {
          scalar_t v = inp_ptr_NC[ix_bsw + Wi * (iy_bsw + Hi * iz_bsw)];
          if (v == out_label) {
            gix -= (iy - iy_tne) * (iz - iz_tne) * gOutGMT;
            giy += (ix_tne - ix) * (iz - iz_tne) * gOutGMT;
            giz += (ix_tne - ix) * (iy - iy_tne) * gOutGMT;
          }
        }
        if (within_bounds_3d(iz_bse, iy_bse, ix_bse, Di, Hi, Wi)) {
          scalar_t v = inp_ptr_NC[ix_bse + Wi * (iy_bse + Hi * iz_bse)];
          if (v == out_label) {
            gix += (iy - iy_tnw) * (iz - iz_tnw) * gOutGMT;
            giy += (ix - ix_tnw) * (iz - iz_tnw) * gOutGMT;
            giz += (ix - ix_tnw) * (iy - iy_tnw) * gOutGMT;
          }
        }
        gix = gix_mult * gix;
        giy = giy_mult * giy;
        giz = giz_mult * giz;

        if (grad_affine_collect) {
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
        if (affine_3d_pregrid) {
          const grid_t* pre = affine_3d_pregrid + (broadcast_affine_3d_pregrid ? 0 : (9 * n));
          gridmath_t pg0 = gix * pre[0] + giy * pre[3] + giz * pre[6];
          gridmath_t pg1 = gix * pre[1] + giy * pre[4] + giz * pre[7];
          gridmath_t pg2 = gix * pre[2] + giy * pre[5] + giz * pre[8];
          gix = pg0;
          giy = pg1;
          giz = pg2;
        }
        if (grad_grid) {
          if (broadcast_grid) {
            index_t gi = 3 * (w + W * (h + H * d));
            gpuAtomicAdd(grad_grid + gi, static_cast<grid_t>(gix));
            gpuAtomicAdd(grad_grid + gi + 1, static_cast<grid_t>(giy));
            gpuAtomicAdd(grad_grid + gi + 2, static_cast<grid_t>(giz));
          } else {
            index_t gi = 3 * (w + W * (h + H * (d + D * n)));
            grad_grid[gi] += static_cast<grid_t>(gix);
            grad_grid[gi + 1] += static_cast<grid_t>(giy);
            grad_grid[gi + 2] += static_cast<grid_t>(giz);
          }
        }
      }
    }
  }

  if (grad_affine_collect) {
    for (int affid = 0; affid < 12; ++affid) {
      _affine_grad_shared_[threadIdx.x] = _affine_grad_[affid];
      __syncthreads();
      for (int tid = BLOCKSIZE_3D / 2; tid > 0; tid /= 2) {
        if (threadIdx.x < tid)
          _affine_grad_shared_[threadIdx.x] += _affine_grad_shared_[threadIdx.x + tid];
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        if (broadcast_affine_3d) {
          gpuAtomicAdd(grad_affine_collect + blockIdx.x * 12 + affid, static_cast<grid_t>(_affine_grad_shared_[0]));
        } else {
          grad_affine_collect[affid + 12 * (blockIdx.x + gridDim.x * n)] = static_cast<grid_t>(_affine_grad_shared_[0]);
        }
      }
      __syncthreads();
    }
  }
}

// ---------- C++ 2D ----------
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
    const std::optional<float> background_label) {

  int64_t H, W;
  TORCH_CHECK(input.dim() == 4, "input must be 4D");
  TORCH_CHECK(input.device().is_cuda(), "input must be on CUDA");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(grid.has_value() || affine_2d.has_value(), "one of grid or affine_2d must exist");

  c10::DeviceGuard guard(input.device());
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input.device().index());
  at::cuda::CUDAStreamGuard stream_guard(stream);

  int64_t batch_size_max = input.size(0);
  if (affine_2d.has_value()) batch_size_max = std::max(batch_size_max, affine_2d.value().size(0));
  if (grid.has_value()) batch_size_max = std::max(batch_size_max, grid.value().size(0));
  if (affine_2d_pregrid.has_value()) batch_size_max = std::max(batch_size_max, affine_2d_pregrid.value().size(0));

  bool broadcast_input = false, broadcast_affine_2d = false, broadcast_grid = false, broadcast_affine_2d_pregrid = false;
  if (batch_size_max > 1) {
    if (input.size(0) == 1) broadcast_input = true;
    else if (input.size(0) != batch_size_max) TORCH_CHECK(false, "input batch size mismatch");
    if (affine_2d.has_value() && affine_2d.value().size(0) == 1) broadcast_affine_2d = true;
    else if (affine_2d.has_value() && affine_2d.value().size(0) != batch_size_max) TORCH_CHECK(false, "affine_2d batch size mismatch");
    if (affine_2d_pregrid.has_value() && affine_2d_pregrid.value().size(0) == 1) broadcast_affine_2d_pregrid = true;
    else if (affine_2d_pregrid.has_value() && affine_2d_pregrid.value().size(0) != batch_size_max) TORCH_CHECK(false, "affine_2d_pregrid batch size mismatch");
    if (grid.has_value() && grid.value().size(0) == 1) broadcast_grid = true;
    else if (grid.has_value() && grid.value().size(0) != batch_size_max) TORCH_CHECK(false, "grid batch size mismatch");
  }

  if (grid.has_value()) {
    check_grid_sampler_common_v2(input, grid.value());
    TORCH_CHECK(grid.value().is_contiguous(), "grid must be contiguous");
    H = grid.value().size(1);
    W = grid.value().size(2);
  } else {
    H = out_H;
    W = out_W;
  }

  if (affine_2d.has_value()) {
    TORCH_CHECK(affine_2d.value().dim() == 3 && affine_2d.value().size(1) == 2 && affine_2d.value().size(2) == 3, "affine_2d must be (B, 2, 3)");
    TORCH_CHECK(affine_2d.value().device().is_cuda() && affine_2d.value().is_contiguous(), "affine_2d must be CUDA contiguous");
  }
  if (affine_2d_pregrid.has_value()) {
    TORCH_CHECK(affine_2d_pregrid.value().dim() == 3 && affine_2d_pregrid.value().size(1) == 2 && affine_2d_pregrid.value().size(2) == 2, "affine_2d_pregrid must be (B, 2, 2)");
    TORCH_CHECK(affine_2d_pregrid.value().device().is_cuda() && affine_2d_pregrid.value().is_contiguous(), "affine_2d_pregrid must be CUDA contiguous");
  }

  int64_t N = batch_size_max;
  int64_t C = input.size(1);
  int64_t Hi = input.size(2);
  int64_t Wi = input.size(3);

  // Allocate output_labels if not provided
  if (!output_labels.has_value()) {
    output_labels.emplace(torch::zeros({batch_size_max, C, H, W}, input.options()));
  }
  TORCH_CHECK(output_labels.value().dim() == 4 && output_labels.value().size(0) == N && output_labels.value().size(1) == C && output_labels.value().size(2) == H && output_labels.value().size(3) == W, "output_labels shape mismatch");
  TORCH_CHECK(output_labels.value().device().is_cuda() && output_labels.value().is_contiguous(), "output_labels must be CUDA contiguous");

  // Determine dummy value: use background_label if provided, else use min of input
  // Fill output_labels with dummy value
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "fused_grid_sampler_2d_generic_label_forward_dummy", [&] {
    scalar_t dummy;
    if (background_label.has_value()) {
      dummy = static_cast<scalar_t>(background_label.value());
    } else {
      dummy = input.min().item<scalar_t>();
    }
    // Fill output_labels with dummy value
    output_labels.value().fill_(dummy);
  });

  std::optional<torch::Tensor> out_weights_opt;
  if (return_weight) {
    if (!output_weights.has_value())
      output_weights.emplace(torch::zeros({batch_size_max, C, H, W}, input.options()));
    TORCH_CHECK(output_weights.value().sizes() == output_labels.value().sizes(), "output_weights shape must match output_labels");
    TORCH_CHECK(output_weights.value().device().is_cuda() && output_weights.value().is_contiguous(), "output_weights must be CUDA contiguous");
    out_weights_opt = output_weights;
  }

  int64_t count = N * H * W;
  auto grid_scalar_type = grid.has_value() ? grid.value().scalar_type() : (affine_2d.has_value() ? affine_2d.value().scalar_type() : input.scalar_type());

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "fused_grid_sampler_2d_generic_label_forward", [&] {
      using input_t = scalar_t;
      scalar_t dummy;
      if (background_label.has_value()) {
        dummy = static_cast<scalar_t>(background_label.value());
      } else {
        dummy = input.min().item<scalar_t>();
      }
      AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grid_scalar_type, "fused_grid_sampler_2d_generic_label_forward_grid", [&] {
        using grid_t = scalar_t;
        if (canUse32BitIndexMath(input) && canUse32BitIndexMath(output_labels.value())) {
          fused_grid_sampler_2d_generic_label_forward_kernel<input_t, grid_t, int32_t>
              <<<GET_BLOCKS(count, 512), 512, 0, stream>>>(
                  static_cast<int32_t>(count),
                  input.data_ptr<input_t>(),
                  grid.has_value() ? grid.value().data_ptr<grid_t>() : nullptr,
                  affine_2d.has_value() ? affine_2d.value().data_ptr<grid_t>() : nullptr,
                  affine_2d_pregrid.has_value() ? affine_2d_pregrid.value().data_ptr<grid_t>() : nullptr,
                  static_cast<int32_t>(N), static_cast<int32_t>(C), static_cast<int32_t>(Hi), static_cast<int32_t>(Wi),
                  static_cast<int32_t>(H), static_cast<int32_t>(W),
                  grid_xmin, grid_ymin, grid_xmax, grid_ymax,
                  is_displacement,
                  dummy,
                  output_labels.value().data_ptr<input_t>(),
                  (return_weight && output_weights.has_value()) ? output_weights.value().data_ptr<input_t>() : nullptr,
                  return_weight,
                  static_cast<GridSamplerPadding>(padding_mode),
                  align_corners,
                  broadcast_input, broadcast_affine_2d, broadcast_grid, broadcast_affine_2d_pregrid);
        } else {
          fused_grid_sampler_2d_generic_label_forward_kernel<input_t, grid_t, int64_t>
              <<<GET_BLOCKS(count, 512), 512, 0, stream>>>(
                  count,
                  input.data_ptr<input_t>(),
                  grid.has_value() ? grid.value().data_ptr<grid_t>() : nullptr,
                  affine_2d.has_value() ? affine_2d.value().data_ptr<grid_t>() : nullptr,
                  affine_2d_pregrid.has_value() ? affine_2d_pregrid.value().data_ptr<grid_t>() : nullptr,
                  N, C, Hi, Wi, H, W,
                  grid_xmin, grid_ymin, grid_xmax, grid_ymax,
                  is_displacement,
                  dummy,
                  output_labels.value().data_ptr<input_t>(),
                  (return_weight && output_weights.has_value()) ? output_weights.value().data_ptr<input_t>() : nullptr,
                  return_weight,
                  static_cast<GridSamplerPadding>(padding_mode),
                  align_corners,
                  broadcast_input, broadcast_affine_2d, broadcast_grid, broadcast_affine_2d_pregrid);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
    });
  }

  return {output_labels.value(), out_weights_opt};
}

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
    bool return_weight) {

  int64_t H, W;
  TORCH_CHECK(input.dim() == 4 && input.device().is_cuda() && input.is_contiguous(), "input must be 4D CUDA contiguous");
  TORCH_CHECK(grid.has_value() || affine_2d.has_value(), "one of grid or affine_2d must exist");
  TORCH_CHECK(grad_affine.has_value() || grad_grid.has_value(), "at least one of grad_affine or grad_grid required");

  c10::DeviceGuard guard(input.device());
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input.device().index());
  at::cuda::CUDAStreamGuard stream_guard(stream);

  int64_t batch_size_max = input.size(0);
  if (affine_2d.has_value()) batch_size_max = std::max(batch_size_max, affine_2d.value().size(0));
  if (grid.has_value()) batch_size_max = std::max(batch_size_max, grid.value().size(0));
  if (affine_2d_pregrid.has_value()) batch_size_max = std::max(batch_size_max, affine_2d_pregrid.value().size(0));

  bool broadcast_input = false, broadcast_affine_2d = false, broadcast_grid = false, broadcast_affine_2d_pregrid = false;
  if (batch_size_max > 1) {
    if (input.size(0) == 1) broadcast_input = true;
    if (affine_2d.has_value() && affine_2d.value().size(0) == 1) broadcast_affine_2d = true;
    if (affine_2d_pregrid.has_value() && affine_2d_pregrid.value().size(0) == 1) broadcast_affine_2d_pregrid = true;
    if (grid.has_value() && grid.value().size(0) == 1) broadcast_grid = true;
  }

  if (grid.has_value()) {
    check_grid_sampler_common_v2(input, grid.value());
    TORCH_CHECK(grid.value().is_contiguous(), "grid must be contiguous");
    H = grid.value().size(1);
    W = grid.value().size(2);
  } else {
    H = out_H;
    W = out_W;
  }

  bool affine_requires_grad = grad_affine.has_value() && affine_2d.has_value();
  bool grid_requires_grad = grad_grid.has_value() && grid.has_value();
  if (grid.has_value() && !is_displacement) affine_requires_grad = false;
  if (!return_weight || !grad_weight.has_value()) {
    affine_requires_grad = false;
    grid_requires_grad = false;
  }

  if (!grid_requires_grad && !affine_requires_grad) return;

  int64_t N = batch_size_max;
  int64_t C = input.size(1);
  int64_t Hi = input.size(2);
  int64_t Wi = input.size(3);
  int64_t count = H * W;
  dim3 blockSize3(BLOCKSIZE_3D, 1, 1);
  int64_t gridSize = GET_BLOCKS(count, BLOCKSIZE_3D);
  gridSize = std::min(gridSize, static_cast<int64_t>(65536));
  dim3 gridSize3(gridSize, batch_size_max, 1);

  torch::Tensor grad_affine_collect;
  if (affine_requires_grad) {
    grad_affine_collect = torch::zeros({affine_2d.value().size(0), gridSize, 2, 3}, grad_affine.value().options());
  }

  auto grid_scalar_type = grid.has_value() ? grid.value().scalar_type() : (affine_2d.has_value() ? affine_2d.value().scalar_type() : input.scalar_type());

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "fused_grid_sampler_2d_generic_label_backward", [&] {
    using input_t = scalar_t;
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grid_scalar_type, "fused_grid_sampler_2d_generic_label_backward_grid", [&] {
      using grid_t = scalar_t;
      fused_grid_sampler_2d_generic_label_backward_kernel<input_t, grid_t, int64_t>
          <<<gridSize3, blockSize3, 0, stream>>>(
              count,
              input.data_ptr<input_t>(),
              output_labels.data_ptr<input_t>(),
              grid.has_value() ? grid.value().data_ptr<grid_t>() : nullptr,
              affine_2d.has_value() ? affine_2d.value().data_ptr<grid_t>() : nullptr,
              affine_2d_pregrid.has_value() ? affine_2d_pregrid.value().data_ptr<grid_t>() : nullptr,
              (return_weight && grad_weight.has_value()) ? grad_weight.value().data_ptr<input_t>() : nullptr,
              affine_requires_grad ? grad_affine_collect.data_ptr<grid_t>() : nullptr,
              grid_requires_grad ? grad_grid.value().data_ptr<grid_t>() : nullptr,
              N, C, Hi, Wi, H, W,
              grid_xmin, grid_ymin, grid_xmax, grid_ymax,
              is_displacement,
              return_weight,
              static_cast<GridSamplerPadding>(padding_mode),
              align_corners,
              broadcast_input, broadcast_affine_2d, broadcast_grid, broadcast_affine_2d_pregrid);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  });

  if (affine_requires_grad) {
    grad_affine.value().copy_(grad_affine_collect.sum(1));
  }
}

// ---------- C++ 3D ----------
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
    const std::optional<float> background_label) {

  int64_t D, H, W;
  TORCH_CHECK(input.dim() == 5, "input must be 5D");
  TORCH_CHECK(input.device().is_cuda(), "input must be on CUDA");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(grid.has_value() || affine_3d.has_value(), "one of grid or affine_3d must exist");

  c10::DeviceGuard guard(input.device());
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input.device().index());
  at::cuda::CUDAStreamGuard stream_guard(stream);

  int64_t batch_size_max = input.size(0);
  if (affine_3d.has_value()) batch_size_max = std::max(batch_size_max, affine_3d.value().size(0));
  if (grid.has_value()) batch_size_max = std::max(batch_size_max, grid.value().size(0));
  if (grid_affine.has_value()) batch_size_max = std::max(batch_size_max, grid_affine.value().size(0));

  bool broadcast_input = false, broadcast_affine_3d = false, broadcast_grid = false, broadcast_affine_3d_pregrid = false;
  if (batch_size_max > 1) {
    if (input.size(0) == 1) broadcast_input = true;
    else if (input.size(0) != batch_size_max) TORCH_CHECK(false, "input batch size mismatch");
    if (affine_3d.has_value() && affine_3d.value().size(0) == 1) broadcast_affine_3d = true;
    else if (affine_3d.has_value() && affine_3d.value().size(0) != batch_size_max) TORCH_CHECK(false, "affine_3d batch size mismatch");
    if (grid_affine.has_value() && grid_affine.value().size(0) == 1) broadcast_affine_3d_pregrid = true;
    else if (grid_affine.has_value() && grid_affine.value().size(0) != batch_size_max) TORCH_CHECK(false, "grid_affine batch size mismatch");
    if (grid.has_value() && grid.value().size(0) == 1) broadcast_grid = true;
    else if (grid.has_value() && grid.value().size(0) != batch_size_max) TORCH_CHECK(false, "grid batch size mismatch");
  }

  if (grid.has_value()) {
    check_grid_sampler_common_v2(input, grid.value());
    TORCH_CHECK(grid.value().is_contiguous(), "grid must be contiguous");
    D = grid.value().size(1);
    H = grid.value().size(2);
    W = grid.value().size(3);
  } else {
    D = out_D;
    H = out_H;
    W = out_W;
  }

  if (affine_3d.has_value()) {
    TORCH_CHECK(affine_3d.value().dim() == 3 && affine_3d.value().size(1) == 3 && affine_3d.value().size(2) == 4, "affine_3d must be (B, 3, 4)");
    TORCH_CHECK(affine_3d.value().device().is_cuda() && affine_3d.value().is_contiguous(), "affine_3d must be CUDA contiguous");
  }
  if (grid_affine.has_value()) {
    TORCH_CHECK(grid_affine.value().dim() == 3 && grid_affine.value().size(1) == 3 && grid_affine.value().size(2) == 3, "grid_affine must be (B, 3, 3)");
    TORCH_CHECK(grid_affine.value().device().is_cuda() && grid_affine.value().is_contiguous(), "grid_affine must be CUDA contiguous");
  }

  int64_t N = batch_size_max;
  int64_t C = input.size(1);
  int64_t Di = input.size(2);
  int64_t Hi = input.size(3);
  int64_t Wi = input.size(4);

  // Allocate output_labels if not provided
  if (!output_labels.has_value()) {
    output_labels.emplace(torch::zeros({batch_size_max, C, D, H, W}, input.options()));
  }
  TORCH_CHECK(output_labels.value().dim() == 5 && output_labels.value().size(0) == N && output_labels.value().size(1) == C && output_labels.value().size(2) == D && output_labels.value().size(3) == H && output_labels.value().size(4) == W, "output_labels shape mismatch");
  TORCH_CHECK(output_labels.value().device().is_cuda() && output_labels.value().is_contiguous(), "output_labels must be CUDA contiguous");

  // Determine dummy value: use background_label if provided, else use min of input
  // Fill output_labels with dummy value
  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "fused_grid_sampler_3d_generic_label_forward_dummy", [&] {
    scalar_t dummy;
    if (background_label.has_value()) {
      dummy = static_cast<scalar_t>(background_label.value());
    } else {
      dummy = input.min().item<scalar_t>();
    }
    // Fill output_labels with dummy value
    output_labels.value().fill_(dummy);
  });

  std::optional<torch::Tensor> out_weights_opt;
  if (return_weight) {
    if (!output_weights.has_value())
      output_weights.emplace(torch::zeros({batch_size_max, C, D, H, W}, input.options()));
    TORCH_CHECK(output_weights.value().sizes() == output_labels.value().sizes(), "output_weights shape must match output_labels");
    TORCH_CHECK(output_weights.value().device().is_cuda() && output_weights.value().is_contiguous(), "output_weights must be CUDA contiguous");
    out_weights_opt = output_weights;
  }

  int64_t count = N * D * H * W;
  auto grid_scalar_type = grid.has_value() ? grid.value().scalar_type() : (affine_3d.has_value() ? affine_3d.value().scalar_type() : input.scalar_type());

  if (count > 0) {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "fused_grid_sampler_3d_generic_label_forward", [&] {
      using input_t = scalar_t;
      scalar_t dummy;
      if (background_label.has_value()) {
        dummy = static_cast<scalar_t>(background_label.value());
      } else {
        dummy = input.min().item<scalar_t>();
      }
      AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grid_scalar_type, "fused_grid_sampler_3d_generic_label_forward_grid", [&] {
        using grid_t = scalar_t;
        if (canUse32BitIndexMath(input) && canUse32BitIndexMath(output_labels.value())) {
          fused_grid_sampler_3d_generic_label_forward_kernel<input_t, grid_t, int32_t>
              <<<GET_BLOCKS(count, 512), 512, 0, stream>>>(
                  static_cast<int32_t>(count),
                  input.data_ptr<input_t>(),
                  grid.has_value() ? grid.value().data_ptr<grid_t>() : nullptr,
                  affine_3d.has_value() ? affine_3d.value().data_ptr<grid_t>() : nullptr,
                  grid_affine.has_value() ? grid_affine.value().data_ptr<grid_t>() : nullptr,
                  static_cast<int32_t>(N), static_cast<int32_t>(C), static_cast<int32_t>(Di), static_cast<int32_t>(Hi), static_cast<int32_t>(Wi),
                  static_cast<int32_t>(D), static_cast<int32_t>(H), static_cast<int32_t>(W),
                  grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
                  is_displacement,
                  dummy,
                  output_labels.value().data_ptr<input_t>(),
                  (return_weight && output_weights.has_value()) ? output_weights.value().data_ptr<input_t>() : nullptr,
                  return_weight,
                  static_cast<GridSamplerPadding>(padding_mode),
                  align_corners,
                  broadcast_input, broadcast_affine_3d, broadcast_grid, broadcast_affine_3d_pregrid);
        } else {
          fused_grid_sampler_3d_generic_label_forward_kernel<input_t, grid_t, int64_t>
              <<<GET_BLOCKS(count, 512), 512, 0, stream>>>(
                  count,
                  input.data_ptr<input_t>(),
                  grid.has_value() ? grid.value().data_ptr<grid_t>() : nullptr,
                  affine_3d.has_value() ? affine_3d.value().data_ptr<grid_t>() : nullptr,
                  grid_affine.has_value() ? grid_affine.value().data_ptr<grid_t>() : nullptr,
                  N, C, Di, Hi, Wi, D, H, W,
                  grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
                  is_displacement,
                  dummy,
                  output_labels.value().data_ptr<input_t>(),
                  (return_weight && output_weights.has_value()) ? output_weights.value().data_ptr<input_t>() : nullptr,
                  return_weight,
                  static_cast<GridSamplerPadding>(padding_mode),
                  align_corners,
                  broadcast_input, broadcast_affine_3d, broadcast_grid, broadcast_affine_3d_pregrid);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
    });
  }

  return {output_labels.value(), out_weights_opt};
}

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
    bool return_weight) {

  int64_t D, H, W;
  TORCH_CHECK(input.dim() == 5 && input.device().is_cuda() && input.is_contiguous(), "input must be 5D CUDA contiguous");
  TORCH_CHECK(grid.has_value() || affine_3d.has_value(), "one of grid or affine_3d must exist");
  TORCH_CHECK(grad_affine.has_value() || grad_grid.has_value(), "at least one of grad_affine or grad_grid required");

  c10::DeviceGuard guard(input.device());
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input.device().index());
  at::cuda::CUDAStreamGuard stream_guard(stream);

  int64_t batch_size_max = input.size(0);
  if (affine_3d.has_value()) batch_size_max = std::max(batch_size_max, affine_3d.value().size(0));
  if (grid.has_value()) batch_size_max = std::max(batch_size_max, grid.value().size(0));
  if (grid_affine.has_value()) batch_size_max = std::max(batch_size_max, grid_affine.value().size(0));

  bool broadcast_input = false, broadcast_affine_3d = false, broadcast_grid = false, broadcast_affine_3d_pregrid = false;
  if (batch_size_max > 1) {
    if (input.size(0) == 1) broadcast_input = true;
    if (affine_3d.has_value() && affine_3d.value().size(0) == 1) broadcast_affine_3d = true;
    if (grid_affine.has_value() && grid_affine.value().size(0) == 1) broadcast_affine_3d_pregrid = true;
    if (grid.has_value() && grid.value().size(0) == 1) broadcast_grid = true;
  }

  if (grid.has_value()) {
    check_grid_sampler_common_v2(input, grid.value());
    TORCH_CHECK(grid.value().is_contiguous(), "grid must be contiguous");
    D = grid.value().size(1);
    H = grid.value().size(2);
    W = grid.value().size(3);
  } else {
    D = out_D;
    H = out_H;
    W = out_W;
  }

  bool affine_requires_grad = grad_affine.has_value() && affine_3d.has_value();
  bool grid_requires_grad = grad_grid.has_value() && grid.has_value();
  if (grid.has_value() && !is_displacement) affine_requires_grad = false;
  if (!return_weight || !grad_weight.has_value()) {
    affine_requires_grad = false;
    grid_requires_grad = false;
  }

  if (!grid_requires_grad && !affine_requires_grad) return;

  int64_t N = batch_size_max;
  int64_t C = input.size(1);
  int64_t Di = input.size(2);
  int64_t Hi = input.size(3);
  int64_t Wi = input.size(4);
  int64_t count = D * H * W;
  dim3 blockSize3(BLOCKSIZE_3D, 1, 1);
  int64_t gridSize = GET_BLOCKS(count, BLOCKSIZE_3D);
  gridSize = std::min(gridSize, static_cast<int64_t>(65536));
  dim3 gridSize3(gridSize, batch_size_max, 1);

  torch::Tensor grad_affine_collect;
  if (affine_requires_grad) {
    grad_affine_collect = torch::zeros({affine_3d.value().size(0), gridSize, 3, 4}, grad_affine.value().options());
  }

  auto grid_scalar_type = grid.has_value() ? grid.value().scalar_type() : (affine_3d.has_value() ? affine_3d.value().scalar_type() : input.scalar_type());

  AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, input.scalar_type(), "fused_grid_sampler_3d_generic_label_backward", [&] {
    using input_t = scalar_t;
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grid_scalar_type, "fused_grid_sampler_3d_generic_label_backward_grid", [&] {
      using grid_t = scalar_t;
      fused_grid_sampler_3d_generic_label_backward_kernel<input_t, grid_t, int64_t>
          <<<gridSize3, blockSize3, 0, stream>>>(
              count,
              input.data_ptr<input_t>(),
              output_labels.data_ptr<input_t>(),
              grid.has_value() ? grid.value().data_ptr<grid_t>() : nullptr,
              affine_3d.has_value() ? affine_3d.value().data_ptr<grid_t>() : nullptr,
              grid_affine.has_value() ? grid_affine.value().data_ptr<grid_t>() : nullptr,
              (return_weight && grad_weight.has_value()) ? grad_weight.value().data_ptr<input_t>() : nullptr,
              affine_requires_grad ? grad_affine_collect.data_ptr<grid_t>() : nullptr,
              grid_requires_grad ? grad_grid.value().data_ptr<grid_t>() : nullptr,
              N, C, Di, Hi, Wi, D, H, W,
              grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
              is_displacement,
              return_weight,
              static_cast<GridSamplerPadding>(padding_mode),
              align_corners,
              broadcast_input, broadcast_affine_3d, broadcast_grid, broadcast_affine_3d_pregrid);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  });

  if (affine_requires_grad) {
    grad_affine.value().copy_(grad_affine_collect.sum(1));
  }
}
