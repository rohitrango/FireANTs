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

template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(BLOCKSIZE_3D)
__global__ void fused_warp_create_3d_forward_kernel(
    const index_t count,
    const scalar_t* grid,
    const scalar_t* affine,
    const index_t N,
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
    const bool broadcast_affine,
    const bool broadcast_grid
    ) {
    using opmath_t = at::opmath_type<scalar_t>;
    // fix these values
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
        if(affine) {
            const scalar_t* affine_ptr = affine + (broadcast_affine ? 0 : (12 * n));
            // get normalized coordinate
            x = affine_ptr[0] * ix + affine_ptr[1] * iy + affine_ptr[2] * iz + affine_ptr[3];
            y = affine_ptr[4] * ix + affine_ptr[5] * iy + affine_ptr[6] * iz + affine_ptr[7];
            z = affine_ptr[8] * ix + affine_ptr[9] * iy + affine_ptr[10] * iz + affine_ptr[11];
        }
        else {
            x = ix; y = iy; z = iz;
        }
        // add to displacement
        x += grid[grid_offset];
        y += grid[grid_offset + 1];
        z += grid[grid_offset + 2];
        // 
        output[grid_offset] = x;
        output[grid_offset + 1] = y;
        output[grid_offset + 2] = z;
    }
}

// Note [Passing pointer and offset to fastAtomicAdd]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// For its internal bounds checking, fastAtomicAdd needs to know where the destination address
// lies relative to the entire tensor, so we pass the base grad_input.data and full offset information,
// including batch * channel offset (NC_offset).
template <typename scalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(BLOCKSIZE_3D)
__global__ void fused_warp_create_3d_backward_kernel(
        const index_t count, /* D * H * W */
        const scalar_t* grid,
        const scalar_t* affine,
        // grads
        const scalar_t* grad_output,
        scalar_t* grad_affine_collect,
        scalar_t* grad_grid,
        // input size parameters
        const index_t N,
        const index_t D,
        const index_t H,
        const index_t W,
        const float grid_xmin,
        const float grid_ymin,
        const float grid_zmin,
        const float grid_xmax,
        const float grid_ymax,
        const float grid_zmax,
        const bool broadcast_affine,
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

    // also collect affine map locally to avoid loading multiple times
    scalar_t _affine_map_[12];
    if (affine) {
        const index_t offset = broadcast_affine ? 0 : (12 * n);
        #pragma unroll
        for (index_t i = 0; i < 12; ++i) {
            _affine_map_[i] = affine[offset + i];
        }
    }
    // loop over
    CUDA_KERNEL_LOOP_TYPE(index, count, index_t) {
        const index_t w = index % W;
        const index_t h = (index / W) % H;
        const index_t d = (index / (H * W)) % D;
        const index_t grid_offset = 3 * (w + W * (h + H * (d + (broadcast_grid ? 0 : (D * n)))));

        // get the corresponding input x, y, z co-ordinates from grid
        scalar_t ix, iy, iz;

        ix = w * (grid_xmax - grid_xmin) / (W-1) + grid_xmin;
        iy = h * (grid_ymax - grid_ymin) / (H-1) + grid_ymin;
        iz = d * (grid_zmax - grid_zmin) / (D-1) + grid_zmin;        // get grid coordinates  phi = (A? * x + u?)
        // calculate Ax + grid
        scalar_t gix, giy, giz;
        gix = grad_output[grid_offset];
        giy = grad_output[grid_offset + 1];
        giz = grad_output[grid_offset + 2];
        // copy these to grad grid
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
            _affine_grad_[0] += gix * ix;
            _affine_grad_[1] += gix * iy;
            _affine_grad_[2] += gix * iz;
            _affine_grad_[3] += gix;
            _affine_grad_[4] += giy * ix;
            _affine_grad_[5] += giy * iy;
            _affine_grad_[6] += giy * iz;
            _affine_grad_[7] += giy;
            _affine_grad_[8] += giz * ix;
            _affine_grad_[9] += giz * iy;
            _affine_grad_[10] += giz * iz;
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
                if (broadcast_affine) {
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

torch::Tensor fused_warp_create_3d_forward_impl(
    const std::optional<torch::Tensor> affine,
    const torch::Tensor grid,
    const float grid_xmin, 
    const float grid_ymin,
    const float grid_zmin,
    const float grid_xmax,
    const float grid_ymax,
    const float grid_zmax) {

    // device and stream guards
    c10::DeviceGuard guard(grid.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(grid.device().index());
    at::cuda::CUDAStreamGuard stream_guard(stream);

    int64_t D, H, W;
    D = grid.size(1);
    H = grid.size(2);
    W = grid.size(3);

    // see if we need to broadcast any variable
    int64_t batch_size_max = grid.size(0);
    if (affine.has_value()) {
        batch_size_max = std::max(batch_size_max, affine.value().size(0));
    }
    // broadcast none by default 
    bool broadcast_affine = false, broadcast_grid = false;
    if (batch_size_max > 1) {
        if (affine.has_value() && affine.value().size(0) == 1) {
            broadcast_affine = true;
        } else if (affine.has_value() && affine.value().size(0) != batch_size_max) {
            TORCH_CHECK(false, "affine batch size must match batch size of grid");
        }

        // broadcast grid if it exists
        if (grid.size(0) == 1) {
            broadcast_grid = true;
        } else if (grid.size(0) != batch_size_max) {
            TORCH_CHECK(false, "grid batch size must match batch size of affine");
        }
    }
    // D, H, W will be determined by grid
    TORCH_CHECK(grid.is_contiguous(), "grid must be contiguous");
    TORCH_CHECK(grid.size(4) == 3, "grid must have 3 channels");

    if (affine.has_value()) {
        TORCH_CHECK(affine.value().dim() == 3, "affine must be (B, 3, 4)");
        TORCH_CHECK(affine.value().device().is_cuda(), "affine must be on CUDA");
        TORCH_CHECK(affine.value().is_contiguous(), "affine must be contiguous");
        TORCH_CHECK(affine.value().size(1) == 3, "affine must be (B, 3, 4)");
        TORCH_CHECK(affine.value().size(2) == 4, "affine must be (B, 3, 4)");
    }

    // define output
    int64_t N = batch_size_max;
    // specify output
    torch::Tensor output = torch::zeros({N, D, H, W, 3}, grid.options());
    // input size parameters
    int64_t count = N * D * H * W;
    // input spatial size parameters
    if (count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grid.scalar_type(), "fused_warp_create_3d_forward_kernel", [&] {
            // check if grid is 32-bit
            if (canUse32BitIndexMath(grid)) {
                fused_warp_create_3d_forward_kernel<scalar_t>
                <<<GET_BLOCKS(count, 512), 512, 0, stream>>>(
                    static_cast<int>(count),
                    grid.data_ptr<scalar_t>(),
                    affine.has_value() ? affine.value().data_ptr<scalar_t>() : nullptr,
                    static_cast<int>(N), static_cast<int>(D), static_cast<int>(H), static_cast<int>(W),
                    grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
                    // output
                    output.data_ptr<scalar_t>(),
                    broadcast_affine,
                    broadcast_grid
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            } else {
                fused_warp_create_3d_forward_kernel<scalar_t>
                <<<GET_BLOCKS(count, 512), 512, 0, stream>>>(
                    count,
                    grid.data_ptr<scalar_t>(),
                    affine.has_value() ? affine.value().data_ptr<scalar_t>() : nullptr,
                    N, D, H, W,
                    grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
                    output.data_ptr<scalar_t>(),
                    broadcast_affine,
                    broadcast_grid
                    );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        });
    }
    return output;
}

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
    ) {

    TORCH_CHECK(grid.device().is_cuda(), "grid must be on CUDA");
    TORCH_CHECK(grid.is_contiguous(), "grid must be contiguous");
    TORCH_CHECK(grid.size(4) == 3, "grid must have 3 channels");

    TORCH_CHECK(grad_affine.has_value() || grad_grid.has_value(), "at least one of grad_affine, grad_grid must exist");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");

    // device and stream guards
    c10::DeviceGuard guard(grid.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(grid.device().index());
    at::cuda::CUDAStreamGuard stream_guard(stream);

    // see if we need to broadcast any variable
    int64_t batch_size_max = grid.size(0);
    if (affine.has_value()) {
        batch_size_max = std::max(batch_size_max, affine.value().size(0));
    }
    batch_size_max = std::max(batch_size_max, grid.size(0));

    // broadcast none by default 
    bool broadcast_affine = false, broadcast_grid = false;
    if (batch_size_max > 1) {
        if (affine.has_value() && affine.value().size(0) == 1) {
            broadcast_affine = true;
        } else if (affine.has_value() && affine.value().size(0) != batch_size_max) {
            TORCH_CHECK(false, "affine batch size must match batch size of grid");
        }
        // broadcast grid if it exists
        if (grid.size(0) == 1) {
            broadcast_grid = true;
        } else if (grid.size(0) != batch_size_max) {
            TORCH_CHECK(false, "grid batch size must match batch size of affine");
        }
    }

    // determine if we need to compute gradients
    bool affine_requires_grad = grad_affine.has_value();
    bool grid_requires_grad = grad_grid.has_value();
    // if grid is provided and it is a displacement, there are no gradients w.r.t. affine
    if (!grid_requires_grad && !affine_requires_grad) {
        // nothing to compute
        return;
    }

    // D, H, W will be determined by grid
    TORCH_CHECK(grid.is_contiguous(), "grid must be contiguous");
    TORCH_CHECK(grid.size(4) == 3, "grid must have 3 channels");
    int64_t D = grid.size(1);
    int64_t H = grid.size(2);
    int64_t W = grid.size(3);

    if (affine.has_value()) {
        TORCH_CHECK(affine.value().dim() == 3, "affine must be (B, 3, 4)");
        TORCH_CHECK(affine.value().device().is_cuda(), "affine must be on CUDA");
        TORCH_CHECK(affine.value().is_contiguous(), "affine must be contiguous");
        TORCH_CHECK(affine.value().size(1) == 3, "affine must be (B, 3, 4)");
        TORCH_CHECK(affine.value().size(2) == 4, "affine must be (B, 3, 4)");
    }

    // define output
    int64_t N = batch_size_max;
    // input size parameters (put batch in a separate dimension)
    int64_t count = D * H * W;

    // initialize grid and dim
    dim3 blockSize3(BLOCKSIZE_3D, 1, 1);
    int64_t gridSize = GET_BLOCKS(count, BLOCKSIZE_3D);
    gridSize = std::min(gridSize, static_cast<int64_t>(65536));
    dim3 gridSize3(gridSize, batch_size_max, 1);

    // intermediate grad affine collector
    torch::Tensor grad_affine_collect;
    if (affine_requires_grad) {
        grad_affine_collect = torch::zeros({affine.value().size(0), gridSize, 3, 4}, grad_affine.value().options());
    }

    if (count > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grid.scalar_type(), "fused_warp_create_3d_backward_kernel", [&] {
            // check if grid is 32-bit
            if (canUse32BitIndexMath(grid)) {
                fused_warp_create_3d_backward_kernel<scalar_t>
                <<<gridSize3, blockSize3, 0, stream>>>(
                    static_cast<int>(count),
                    grid.data_ptr<scalar_t>(),
                    affine.has_value() ? affine.value().data_ptr<scalar_t>() : nullptr,
                    // grads
                    grad_output.data_ptr<scalar_t>(),
                    affine_requires_grad ? grad_affine_collect.data_ptr<scalar_t>() : nullptr,
                    grid_requires_grad ? grad_grid.value().data_ptr<scalar_t>() : nullptr,
                    // input size parameters
                    static_cast<int>(N), 
                    static_cast<int>(D), static_cast<int>(H), static_cast<int>(W),
                    grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
                    broadcast_affine,
                    broadcast_grid
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            } else {
                fused_warp_create_3d_backward_kernel<scalar_t>
                <<<gridSize3, blockSize3, 0, stream>>>(
                    count,
                    grid.data_ptr<scalar_t>(),
                    affine.has_value() ? affine.value().data_ptr<scalar_t>() : nullptr,
                    // grads
                    grad_output.data_ptr<scalar_t>(),
                    affine_requires_grad ? grad_affine_collect.data_ptr<scalar_t>() : nullptr,
                    grid_requires_grad ? grad_grid.value().data_ptr<scalar_t>() : nullptr,
                    // input size parameters
                    N, 
                    D, H, W,
                    grid_xmin, grid_ymin, grid_zmin, grid_xmax, grid_ymax, grid_zmax,
                    broadcast_affine,
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
