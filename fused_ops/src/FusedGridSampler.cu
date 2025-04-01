// #define TORCH_ASSERT_NO_OPERATORS
#include <ATen/OpMathType.h>
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/cuda/GridSampler.cuh>
#include <ATen/native/cuda/UpSample.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
// #include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>
#include <c10/macros/Macros.h>
#include <torch/torch.h>
#include <cmath>
#include <iostream>

// Core CUDA headers
#include <cuda_runtime.h>
#include <torch/extension.h>
// PyTorch CUDA headers

using namespace at::cuda::detail;
using namespace at::native;
using at::native::detail::GridSamplerInterpolation;
using at::native::detail::GridSamplerPadding;

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(512)
__global__ void fused_grid_sampler_3d_forward_kernel(
    const index_t count,
    const scalar_t* input,
    const scalar_t* grid,
    const scalar_t* affine_3d,
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
    scalar_t* output,
    const GridSamplerInterpolation interpolation_mode,
    const GridSamplerPadding padding_mode,
    const bool align_corners,
    const bool broadcast_input,
    const bool broadcast_affine_3d,
    const bool broadcast_grid
    ) {

    using opmath_t = at::opmath_type<scalar_t>;

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
        if (!grid) {
            // if grid is not provided, then affine matrix is multiplied to input coordinate
            // displacement is ignored
            // just affine coordiante here, we load the entire affine matrix
            const scalar_t* affine_3d_ptr = affine_3d + (broadcast_affine_3d ? 0 : (12 * n));
            // get normalized coordinate
            ix = w * (grid_xmax - grid_xmin) / (W-1) + grid_xmin;
            iy = h * (grid_ymax - grid_ymin) / (H-1) + grid_ymin;
            iz = d * (grid_zmax - grid_zmin) / (D-1) + grid_zmin;
            // apply affine matrix
            x = affine_3d_ptr[0] * ix + affine_3d_ptr[1] * iy + affine_3d_ptr[2] * iz + affine_3d_ptr[3];
            y = affine_3d_ptr[4] * ix + affine_3d_ptr[5] * iy + affine_3d_ptr[6] * iz + affine_3d_ptr[7];
            z = affine_3d_ptr[8] * ix + affine_3d_ptr[9] * iy + affine_3d_ptr[10] * iz + affine_3d_ptr[11];
        }
        else {
            // grid is provided, load the grid coordinate
            // x = grid[grid_offset];
            // y = grid[grid_offset + 1];
            // z = grid[grid_offset + 2];
            // if these are warp coordinates (`is_displacement` is false), affine matrix is ignored
            // if these are displacement coordinates, get the grid coordinates x, y, z, multiply by affine matrix, then add to displacement
            if (is_displacement) {
                // get grid coordinate
                ix = w * (grid_xmax - grid_xmin) / (W-1) + grid_xmin;
                iy = h * (grid_ymax - grid_ymin) / (H-1) + grid_ymin;
                iz = d * (grid_zmax - grid_zmin) / (D-1) + grid_zmin;
                // apply affine matrix
                if(affine_3d) {
                    const scalar_t* affine_3d_ptr = affine_3d + (broadcast_affine_3d ? 0 : (12 * n));
                    x = affine_3d_ptr[0] * ix + affine_3d_ptr[1] * iy + affine_3d_ptr[2] * iz + affine_3d_ptr[3];
                    y = affine_3d_ptr[4] * ix + affine_3d_ptr[5] * iy + affine_3d_ptr[6] * iz + affine_3d_ptr[7];
                    z = affine_3d_ptr[8] * ix + affine_3d_ptr[9] * iy + affine_3d_ptr[10] * iz + affine_3d_ptr[11];
                }
                // add to displacement
                x += grid[grid_offset];
                y += grid[grid_offset + 1];
                z += grid[grid_offset + 2];
            }
            else {
                // just get warp
                x = grid[grid_offset];
                y = grid[grid_offset + 1];
                z = grid[grid_offset + 2];
            }
        }

        // get the corresponding input x, y, z co-ordinates from grid
        ix = grid_sampler_compute_source_index(x, Wi, padding_mode, align_corners);
        iy = grid_sampler_compute_source_index(y, Hi, padding_mode, align_corners);
        iz = grid_sampler_compute_source_index(z, Di, padding_mode, align_corners);

        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
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

            // get input and output strides
            index_t out_sC = D * H * W;
            index_t inp_sC = Di * Hi * Wi;
            // get input and output pointers
            const scalar_t* inp_ptr_NC = input + (broadcast_input ? 0 : (n * (C * inp_sC)));
            scalar_t* out_ptr_NCDHW = output + (w + W * (h + H * (d + D * C * n))); // add batch, depth, height, width offset

            for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
                //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
                // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
                // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
                // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
                opmath_t out_acc = 0;
                if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, Di, Hi, Wi)) {
                    out_acc += inp_ptr_NC[ix_tnw + Wi * (iy_tnw + Hi * iz_tnw)] * tnw;
                }
                if (within_bounds_3d(iz_tne, iy_tne, ix_tne, Di, Hi, Wi)) {
                    out_acc += inp_ptr_NC[ix_tne + Wi * (iy_tne + Hi * iz_tne)] * tne;
                }
                if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, Di, Hi, Wi)) {
                    out_acc += inp_ptr_NC[ix_tsw + Wi * (iy_tsw + Hi * iz_tsw)] * tsw;
                }
                if (within_bounds_3d(iz_tse, iy_tse, ix_tse, Di, Hi, Wi)) {
                    out_acc += inp_ptr_NC[ix_tse + Wi * (iy_tse + Hi * iz_tse)] * tse;
                }
                if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, Di, Hi, Wi)) {
                    out_acc += inp_ptr_NC[ix_bnw + Wi * (iy_bnw + Hi * iz_bnw)] * bnw;
                }
                if (within_bounds_3d(iz_bne, iy_bne, ix_bne, Di, Hi, Wi)) {
                    out_acc += inp_ptr_NC[ix_bne + Wi * (iy_bne + Hi * iz_bne)] * bne;
                }
                if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, Di, Hi, Wi)) {
                    out_acc += inp_ptr_NC[ix_bsw + Wi * (iy_bsw + Hi * iz_bsw)] * bsw;
                }
                if (within_bounds_3d(iz_bse, iy_bse, ix_bse, Di, Hi, Wi)) {
                    out_acc += inp_ptr_NC[ix_bse + Wi * (iy_bse + Hi * iz_bse)] * bse;
                }
                *out_ptr_NCDHW = out_acc;
            }
        } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
            index_t ix_nearest = static_cast<index_t>(std::nearbyint(ix));
            index_t iy_nearest = static_cast<index_t>(std::nearbyint(iy));
            index_t iz_nearest = static_cast<index_t>(std::nearbyint(iz));

            index_t out_sC = D * H * W;
            index_t inp_sC = Di * Hi * Wi;

            // assign nearest neighbour pixel value to output pixel
            const scalar_t* inp_ptr_NC = input + (broadcast_input ? 0 : (n * (C * inp_sC)));
            scalar_t* out_ptr_NCDHW = output + (w + W * (h + H * (d + D * C * n))); // add batch, depth, height, width offset

            for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
                if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, Di, Hi, Wi)) {
                    *out_ptr_NCDHW = inp_ptr_NC[ix_nearest + Wi * (iy_nearest + Hi * iz_nearest)];
                } else {
                    *out_ptr_NCDHW = static_cast<scalar_t>(0);
                }
            }
        }
    }
}

// Note [Passing pointer and offset to fastAtomicAdd]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// For its internal bounds checking, fastAtomicAdd needs to know where the destination address
// lies relative to the entire tensor, so we pass the base grad_input.data and full offset information,
// including batch * channel offset (NC_offset).
template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(256)
__global__ void fused_grid_sampler_3d_backward_kernel(
        const index_t nthreads,
        TensorInfo<const scalar_t, index_t> grad_output,
        TensorInfo<const scalar_t, index_t> input,
        TensorInfo<const scalar_t, index_t> grid,
        TensorInfo<scalar_t, index_t> grad_input,    // initialized to zeros (or unused if input_requires_grad is false)
        TensorInfo<scalar_t, index_t> grad_grid,     // initialized to empty
        const GridSamplerInterpolation interpolation_mode,
        const GridSamplerPadding padding_mode,
        bool align_corners,
        const index_t grad_input_memory_span,
        const bool input_requires_grad) {

    index_t C = input.sizes[1];
    index_t inp_D = input.sizes[2];
    index_t inp_H = input.sizes[3];
    index_t inp_W = input.sizes[4];
    index_t out_D = grid.sizes[1];
    index_t out_H = grid.sizes[2];
    index_t out_W = grid.sizes[3];

    index_t inp_sN = input.strides[0];
    index_t inp_sC = input.strides[1];
    index_t inp_sD = input.strides[2];
    index_t inp_sH = input.strides[3];
    index_t inp_sW = input.strides[4];

    index_t grid_sN = grid.strides[0];
    index_t grid_sD = grid.strides[1];
    index_t grid_sH = grid.strides[2];
    index_t grid_sW = grid.strides[3];
    index_t grid_sCoor = grid.strides[4];

    index_t gOut_sN = grad_output.strides[0];
    index_t gOut_sC = grad_output.strides[1];
    index_t gOut_sD = grad_output.strides[2];
    index_t gOut_sH = grad_output.strides[3];
    index_t gOut_sW = grad_output.strides[4];
    // gInp_* (and NC_offset below) are not really needed if input_requires_grad is false.
    int64_t gInp_sN = 0;
    int64_t gInp_sC = 0;
    int64_t gInp_sD = 0;
    int64_t gInp_sH = 0;
    int64_t gInp_sW = 0;
    if (input_requires_grad) {
        gInp_sN = grad_input.strides[0];
        gInp_sC = grad_input.strides[1];
        gInp_sD = grad_input.strides[2];
        gInp_sH = grad_input.strides[3];
        gInp_sW = grad_input.strides[4];
    }
    index_t gGrid_sW = grad_grid.strides[3];

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
        const index_t w = index % out_W;
        const index_t h = (index / out_W) % out_H;
        const index_t d = (index / (out_H * out_W)) % out_D;
        const index_t n = index / (out_D * out_H * out_W);
        const auto grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

        // get the corresponding input x, y, z co-ordinates from grid
        scalar_t ix = grid.data[grid_offset];
        scalar_t iy = grid.data[grid_offset + grid_sCoor];
        scalar_t iz = grid.data[grid_offset + 2 * grid_sCoor];

        // multipliers for gradients on ix, iy, and iz
        scalar_t gix_mult, giy_mult, giz_mult;
        ix = grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &gix_mult);
        iy = grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &giy_mult);
        iz = grid_sampler_compute_source_index_set_grad(iz, inp_D, padding_mode, align_corners, &giz_mult);

        if (interpolation_mode == GridSamplerInterpolation::Bilinear) {
            // get corner pixel values from (x, y, z)
            // for 4d, we used north-east-south-west
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
            const scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
            index_t NC_offset;
            if (input_requires_grad) {
                NC_offset = n * gInp_sN;
            }
            const scalar_t *inp_ptr_NC = input.data + n * inp_sN;
            // calculate bilinear weighted pixel value and set output pixel
            for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, NC_offset += gInp_sC, inp_ptr_NC += inp_sC) {
                scalar_t gOut = *gOut_ptr_NCDHW;

                // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
                if (input_requires_grad) {
                    safe_add_3d(grad_input.data, iz_tnw, iy_tnw, ix_tnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw * gOut,
                                            NC_offset, grad_input_memory_span);
                    safe_add_3d(grad_input.data, iz_tne, iy_tne, ix_tne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne * gOut,
                                            NC_offset, grad_input_memory_span);
                    safe_add_3d(grad_input.data, iz_tsw, iy_tsw, ix_tsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw * gOut,
                                            NC_offset, grad_input_memory_span);
                    safe_add_3d(grad_input.data, iz_tse, iy_tse, ix_tse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse * gOut,
                                            NC_offset, grad_input_memory_span);
                    safe_add_3d(grad_input.data, iz_bnw, iy_bnw, ix_bnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw * gOut,
                                            NC_offset, grad_input_memory_span);
                    safe_add_3d(grad_input.data, iz_bne, iy_bne, ix_bne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne * gOut,
                                            NC_offset, grad_input_memory_span);
                    safe_add_3d(grad_input.data, iz_bsw, iy_bsw, ix_bsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw * gOut,
                                            NC_offset, grad_input_memory_span);
                    safe_add_3d(grad_input.data, iz_bse, iy_bse, ix_bse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse * gOut,
                                            NC_offset, grad_input_memory_span);
                }
                // calculate grad_grid
                if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
                    scalar_t tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
                    gix -= tnw_val * (iy_bse - iy)        * (iz_bse - iz)        * gOut;
                    giy -= tnw_val * (ix_bse - ix)        * (iz_bse - iz)        * gOut;
                    giz -= tnw_val * (ix_bse - ix)        * (iy_bse - iy)        * gOut;
                }
                if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
                    scalar_t tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
                    gix += tne_val * (iy_bsw - iy)        * (iz_bsw - iz)        * gOut;
                    giy -= tne_val * (ix        - ix_bsw) * (iz_bsw - iz)        * gOut;
                    giz -= tne_val * (ix        - ix_bsw) * (iy_bsw - iy)        * gOut;
                }
                if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
                    scalar_t tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
                    gix -= tsw_val * (iy - iy_bne)        * (iz_bne - iz)        * gOut;
                    giy += tsw_val * (ix_bne - ix)        * (iz_bne - iz)        * gOut;
                    giz -= tsw_val * (ix_bne - ix)        * (iy        - iy_bne) * gOut;
                }
                if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
                    scalar_t tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
                    gix += tse_val * (iy - iy_bnw)        * (iz_bnw - iz)        * gOut;
                    giy += tse_val * (ix        - ix_bnw) * (iz_bnw - iz)        * gOut;
                    giz -= tse_val * (ix        - ix_bnw) * (iy        - iy_bnw) * gOut;
                }
                if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
                    scalar_t bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
                    gix -= bnw_val * (iy_tse - iy)        * (iz - iz_tse)        * gOut;
                    giy -= bnw_val * (ix_tse - ix)        * (iz - iz_tse)        * gOut;
                    giz += bnw_val * (ix_tse - ix)        * (iy_tse - iy)        * gOut;
                }
                if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
                    scalar_t bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
                    gix += bne_val * (iy_tsw - iy)        * (iz - iz_tsw)        * gOut;
                    giy -= bne_val * (ix        - ix_tsw) * (iz - iz_tsw)        * gOut;
                    giz += bne_val * (ix        - ix_tsw) * (iy_tsw - iy)        * gOut;
                }
                if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
                    scalar_t bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
                    gix -= bsw_val * (iy - iy_tne)        * (iz - iz_tne)        * gOut;
                    giy += bsw_val * (ix_tne - ix)        * (iz - iz_tne)        * gOut;
                    giz += bsw_val * (ix_tne - ix)        * (iy        - iy_tne) * gOut;
                }
                if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
                    scalar_t bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
                    gix += bse_val * (iy - iy_tnw)        * (iz - iz_tnw)        * gOut;
                    giy += bse_val * (ix        - ix_tnw) * (iz - iz_tnw)        * gOut;
                    giz += bse_val * (ix        - ix_tnw) * (iy        - iy_tnw) * gOut;
                }
            }

            // assuming grad_grid is contiguous
            // thus we can
            //     1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
            //     2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
            scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
            gGrid_ptr_NDHW[0] = gix_mult * gix;
            gGrid_ptr_NDHW[1] = giy_mult * giy;
            gGrid_ptr_NDHW[2] = giz_mult * giz;
        } else if (interpolation_mode == GridSamplerInterpolation::Nearest) {
            if (input_requires_grad) {
                auto ix_nearest = static_cast<index_t>(std::nearbyint(ix));
                auto iy_nearest = static_cast<index_t>(std::nearbyint(iy));
                auto iz_nearest = static_cast<index_t>(std::nearbyint(iz));

                // assign nearest neighbour pixel value to output pixel
                const scalar_t *gOut_ptr_NCDHW = grad_output.data + n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
                index_t NC_offset = n * gInp_sN;
                for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, NC_offset += gInp_sC) {
                    // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
                    safe_add_3d(grad_input.data, iz_nearest, iy_nearest, ix_nearest,
                                            gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, *gOut_ptr_NCDHW,
                                            NC_offset, grad_input_memory_span);
                }
            }
            // assuming grad_grid is contiguous
            // thus we can
            //     1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
            //     2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
            scalar_t *gGrid_ptr_NDHW = grad_grid.data + index * gGrid_sW;
            gGrid_ptr_NDHW[0] = static_cast<scalar_t>(0);
            gGrid_ptr_NDHW[1] = static_cast<scalar_t>(0);
            gGrid_ptr_NDHW[2] = static_cast<scalar_t>(0);
        }
    }
}


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
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
    // See NOTE [ grid_sampler Native Functions ].
    // Add checks here in case this is called instead of grid_sampler.
    //   check_grid_sampler_common(input, grid);
    //   check_grid_sampler_3d(input, grid, interpolation_mode);

    int64_t D, H, W;

    TORCH_CHECK(input.dim() == 5, "input must be 5D");
    TORCH_CHECK(input.device().is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(grid.has_value() || affine_3d.has_value(), "one of grid or affine_3d must exist");

    // see if we need to broadcast any variable
    int64_t batch_size_max = input.size(0);
    if (affine_3d.has_value()) {
        batch_size_max = std::max(batch_size_max, affine_3d.value().size(0));
    }
    if (grid.has_value()) {
        batch_size_max = std::max(batch_size_max, grid.value().size(0));
    }
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
        if (grid.has_value() && grid.value().size(0) == 1) {
            broadcast_grid = true;
        } else if (grid.has_value() && grid.value().size(0) != batch_size_max) {
            TORCH_CHECK(false, "grid batch size must match batch size of input or affine_3d");
        }
    }

    // D, H, W will be determined by grid
    if (grid.has_value()) {
        check_grid_sampler_common(input, grid.value());
        check_grid_sampler_3d(input, grid.value(), interpolation_mode);
        TORCH_CHECK(grid.value().is_contiguous(), "grid must be contiguous");
        D = grid.value().size(1);
        H = grid.value().size(2);
        W = grid.value().size(3);
    }
    else {
        // grid does not exist, affine must exist
        // size of output is determined by input (coordinates determined by affine)
        D = out_D;
        H = out_H;
        W = out_W;
    }

    if (affine_3d.has_value()) {
        TORCH_CHECK(input.size(0) == affine_3d.value().size(0), "input and affine_3d must have the same batch size");
        TORCH_CHECK(affine_3d.value().dim() == 3, "affine_3d must be (B, 3, 4)");
        TORCH_CHECK(affine_3d.value().device().is_cuda(), "affine_3d must be on CUDA");
        TORCH_CHECK(affine_3d.value().is_contiguous(), "affine_3d must be contiguous");
        TORCH_CHECK(affine_3d.value().size(1) == 3, "affine_3d must be (B, 3, 4)");
        TORCH_CHECK(affine_3d.value().size(2) == 4, "affine_3d must be (B, 3, 4)");
    }

    // define output
    int64_t N = batch_size_max;
    int64_t C = input.size(1);
    torch::Tensor output = torch::zeros({batch_size_max, C, D, H, W}, input.options());

    // input size parameters
    int64_t count = N * D * H * W;

    // input spatial size parameters
    int64_t Di = input.size(2);
    int64_t Hi = input.size(3);
    int64_t Wi = input.size(4);

    if (count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "fused_grid_sampler_3d_forward_kernel", [&] {
            // check if grid is 32-bit
            bool grid32bit;
            if(grid.has_value()) {
                grid32bit = canUse32BitIndexMath(grid.value());
            } else {
                grid32bit = true;
            }
            if (canUse32BitIndexMath(input) && grid32bit &&
                canUse32BitIndexMath(output)) {
                fused_grid_sampler_3d_forward_kernel<scalar_t>
                <<<GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                    static_cast<int>(count),
                    input.data_ptr<scalar_t>(),
                    grid.has_value() ? grid.value().data_ptr<scalar_t>() : nullptr,
                    affine_3d.has_value() ? affine_3d.value().data_ptr<scalar_t>() : nullptr,
                    static_cast<int>(N), static_cast<int>(C), static_cast<int>(Di), static_cast<int>(Hi), static_cast<int>(Wi),
                    static_cast<int>(D), static_cast<int>(H), static_cast<int>(W),
                    grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
                    is_displacement,
                    // output
                    output.data_ptr<scalar_t>(),
                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                    static_cast<GridSamplerPadding>(padding_mode),
                    align_corners,
                    broadcast_input,
                    broadcast_affine_3d,
                    broadcast_grid
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            } else {
                fused_grid_sampler_3d_forward_kernel<scalar_t>
                <<<GET_BLOCKS(count, 512), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                    count,
                    input.data_ptr<scalar_t>(),
                    grid.has_value() ? grid.value().data_ptr<scalar_t>() : nullptr,
                    affine_3d.has_value() ? affine_3d.value().data_ptr<scalar_t>() : nullptr,
                    N, C, Di, Hi, Wi,
                    D, H, W,
                    grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
                    is_displacement,
                    output.data_ptr<scalar_t>(),
                    static_cast<GridSamplerInterpolation>(interpolation_mode),
                    static_cast<GridSamplerPadding>(padding_mode),
                    align_corners,
                    broadcast_input,
                    broadcast_affine_3d,
                    broadcast_grid
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        });
    }
    return output;
}

