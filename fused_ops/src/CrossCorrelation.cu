// Core CUDA headers
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>
// PyTorch CUDA headers
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>

// C++ standard headers
#include <cmath>
#include <assert.h>

// Local headers
#include "CrossCorrelation.h"
#include "common.h"

template <typename scalar_t> 
__global__ void cc3d_fwd_interm_v1_kernel(
    scalar_t* __restrict__ interm, scalar_t* __restrict__ out, 
    int kernel_volume, Reduction reduce, 
    int batch_size, int n_out_channels, int height, int width, int depth, 
    int out_size, float nr, float dr) {

    // initialize thread id
    int n = batch_size * n_out_channels * height * width * depth;
    CUDA_KERNEL_LOOP(i, n) {
        int b = i / (n_out_channels * height * width * depth);
        int c = (i / (height * width * depth)) % n_out_channels;
        int h = (i / (width * depth)) % height;
        int w = (i / depth) % width;
        int d = i % depth;

        // int inp_index_common = b * (5 * n_out_channels * height * width * depth) + h * (width * depth) + w * depth + d;
        int input_idx_common = d + depth * (w + width * (h + height * (5 * n_out_channels * b)));
        int spatial_mult = depth * width * height;

        // Get variables (all means)
        float Ival = static_cast<float>(interm[input_idx_common + c * spatial_mult]);
        float Jval = static_cast<float>(interm[input_idx_common + (c + n_out_channels) * spatial_mult]);
        float I2val = static_cast<float>(interm[input_idx_common + (c + 2 * n_out_channels) * spatial_mult]);
        float J2val = static_cast<float>(interm[input_idx_common + (c + 3 * n_out_channels) * spatial_mult]);
        float IJval = static_cast<float>(interm[input_idx_common + (c + 4 * n_out_channels) * spatial_mult]);
        
        // compute cross-correlation
        float kv = (float)kernel_volume;
        float aval = kv * (IJval - Ival * Jval);
        float bval = kv * (I2val - Ival * Ival);
        float cval = kv * (J2val - Jval * Jval);
        // cross-correlation
        kv = (aval * aval + nr)/(bval * cval + dr);

        if (reduce == Reduction::NONE) {
            int out_idx = d + depth * (w + width * (h + height * (c + n_out_channels * b)));
            out[out_idx] = static_cast<scalar_t>(kv);
        }
        else {
            int out_idx = i % out_size;
            gpuAtomicAdd(out + out_idx, static_cast<scalar_t>(kv));
        }
    }
}


torch::Tensor cc3d_fwd_interm_v1(torch::Tensor intermediates, int kernel_volume, Reduction reduction, float nr, float dr) {
    /*
    intermediates: [batch_size, n_channels, height, width, depth]
       * note that intermediates contains 5 blocks 
           block of I, J, I**2, J**2, IJ

    kernel_volume: int
    reduction: Reduction

    output: [batch_size, n_out_channels, height, width, depth] if reduction == Reduction::NONE
            [batch_size, n_out_channels] if reduction == Reduction::SUM
            [batch_size, n_out_channels] if reduction == Reduction::MEAN

    */
    auto n = intermediates.numel() / 5;
    CHECK_INPUT(intermediates);
    
    int batch_size = intermediates.size(0);
    int n_channels = intermediates.size(1);
    int height = intermediates.size(2);
    int width = intermediates.size(3);
    int depth = intermediates.size(4);
    assert(n_channels % 5 == 0);
    int n_out_channels = n_channels / 5;   // one output channel for each block of 5 input channels

    // initialize output tensor
    int out_size = 0;
    torch::Tensor out;
    if (reduction == Reduction::NONE) {
        out = torch::zeros({batch_size, n_out_channels, height, width, depth}, intermediates.scalar_type());
    } else {
        out_size = (int)(sqrtf((float)(height * width * depth * n_out_channels)));
        // std::cout << "out_size: " << out_size << std::endl;
        out = torch::zeros({out_size}, intermediates.scalar_type());
    }
    out = out.to(intermediates.device());

    // initialize blocks
    dim3 blockSize(BLOCKSIZE_3D);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, intermediates.scalar_type(), "cc3d_fwd_interm_v1", ([&] {
        cc3d_fwd_interm_v1_kernel<scalar_t><<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
            intermediates.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), kernel_volume, reduction, 
            batch_size, n_out_channels, height, width, depth, out_size, nr, dr);
    }));

    // reduce output if necessary
    if (reduction == Reduction::NONE) {
        return out;
    } 
    // reduce output
    out = out.sum();
    if (reduction == Reduction::MEAN) {
        out = out / n;
    }
    return out;
}
