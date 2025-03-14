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
__global__ void create_intermediates_kernel3d(
    scalar_t* __restrict__ input_img, scalar_t* __restrict__ target_img, scalar_t* __restrict__ intermediates,
    int batch_size, int n_channels, int height, int width, int depth) {

    int n = batch_size * n_channels * height * width * depth;
    CUDA_KERNEL_LOOP(i, n) {
        int b = i / (n_channels * height * width * depth);
        int c = (i / (height * width * depth)) % n_channels;
        int h = (i / (width * depth)) % height;
        int w = (i / depth) % width;
        int d = i % depth;

        // read input and target images
        scalar_t I = input_img[i];
        scalar_t J = target_img[i];

        // compute intermediate values
        scalar_t I2 = I * I;
        scalar_t J2 = J * J;
        scalar_t IJ = I * J;

        // write intermediate values
        int out_idx_start = d + depth * (w + width * (h + height * (c + 5 * n_channels * (b))));
        int out_offset = n_channels * height * width * depth;
        intermediates[out_idx_start] = I;
        intermediates[out_idx_start + out_offset] = J;
        intermediates[out_idx_start + 2 * out_offset] = I2;
        intermediates[out_idx_start + 3 * out_offset] = J2;
        intermediates[out_idx_start + 4 * out_offset] = IJ;
    }
}

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

template <typename scalar_t>
__global__ void cc3d_bwd_interm_v1_kernel_reduced(
    scalar_t* __restrict__ interm, scalar_t* __restrict__ out, 
    int kernel_volume, Reduction reduce, 
    int batch_size, int n_out_channels, int height, int width, int depth, 
    int out_size, float nr, float dr) {
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

torch::Tensor cc3d_bwd_interm_v1(torch::Tensor intermediates, torch::Tensor input_img, torch::Tensor target_img, 
    torch::Tensor grad_output, 
    std::optional<torch::Tensor> grad_input_img, std::optional<torch::Tensor> grad_target_img, 
    int kernel_size, float nr, float dr, Reduction reduction) {
    /*
    intermediates: [batch_size, 5*n_channels, height, width, depth]
       * note that intermediates contains 5 blocks 
           block of I, J, I**2, J**2, IJ

    kernel_volume: int
    reduction: Reduction

    input_img: [batch_size, n_channels, height, width, depth]
    target_img: [batch_size, n_channels, height, width, depth]
    grad_output: [batch_size, n_out_channels, height, width, depth]
    grad_input_img: [batch_size, n_channels, height, width, depth]
    grad_target_img: [batch_size, n_channels, height, width, depth]
    */

    auto n = intermediates.numel() / 5;
    int batch_size = intermediates.size(0);
    int n_channels = intermediates.size(1);
    int height = intermediates.size(2);
    int width = intermediates.size(3);
    int depth = intermediates.size(4);
    assert(n_channels % 5 == 0);
    int n_out_channels = n_channels / 5;   // one output channel for each block of 5 input channels

    CHECK_INPUT(intermediates);
    CHECK_INPUT(input_img);
    CHECK_INPUT(target_img);
    CHECK_INPUT(grad_output);
    if (grad_input_img.has_value()) {
        CHECK_INPUT(grad_input_img.value());
    }
    if (grad_target_img.has_value()) {
        CHECK_INPUT(grad_target_img.value());
    }

    if (reduction == Reduction::NONE) {
        throw std::runtime_error("NONE reduction not supported for backward pass yet");
    }
    else {
        // initialize output tensors
        float grad_output_val = grad_output.cpu().item<float>();
        // dont do it yet (let's see if more stable gradients can be achieved)
        // if (reduction == Reduction::MEAN) {
        //     grad_output_val = grad_output_val / n;
        // }
        dim3 blockSize(BLOCKSIZE_3D);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

        // AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, intermediates.scalar_type(), "cc3d_bwd_interm_v1", ([&] {
        //     // initialize output gradient pointers if they exist
        //     scalar_t* grad_input_img_ptr = grad_input_img.has_value() ? grad_input_img.value().data_ptr<scalar_t>() : nullptr;
        //     scalar_t* grad_target_img_ptr = grad_target_img.has_value() ? grad_target_img.value().data_ptr<scalar_t>() : nullptr;

        //     cc3d_bwd_interm_v1_kernel_reduced<scalar_t><<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
        //         intermediates.data_ptr<scalar_t>(), input_img.data_ptr<scalar_t>(), target_img.data_ptr<scalar_t>(),  
        //         grad_input_img_ptr, grad_target_img_ptr, kernel_size, nr, dr, reduction);
        // }));
        
        std::cout << "grad output shape: " << grad_output.sizes() << std::endl;
        return torch::Tensor();
    }
}

torch::Tensor create_intermediates(torch::Tensor input_img, torch::Tensor target_img, torch::Tensor intermediates) {
    /*
    input_img: [batch_size, n_channels, height, width, depth] or [batch_size, n_channels, height, width]
    target_img: [batch_size, n_channels, height, width, depth] or [batch_size, n_channels, height, width]
    */

    // std::cout << "input_img: " << input_img.sizes() << std::endl;
    // std::cout << "target_img: " << target_img.sizes() << std::endl;
    // std::cout << "input_img contiguous: " << input_img.is_contiguous() << std::endl;
    // std::cout << "target_img contiguous: " << target_img.is_contiguous() << std::endl;
    // std::cout << "input_img strides: " << input_img.strides() << std::endl;
    // std::cout << "target_img strides: " << target_img.strides() << std::endl;

    auto numdims = input_img.dim();

    CHECK_INPUT(target_img);
    CHECK_INPUT(input_img);
    CHECK_INPUT(intermediates);

    if (numdims == 5) {
        // 3D case
        auto batch_size = input_img.size(0);
        auto n_channels = input_img.size(1);
        auto height = input_img.size(2);
        auto width = input_img.size(3);
        auto depth = input_img.size(4);
        auto n = input_img.numel();

        // torch::Tensor intermediates = torch::zeros({batch_size, 5 * n_channels, height, width, depth}, input_img.scalar_type());
        // intermediates = intermediates.to(input_img.device());
        dim3 blockSize(BLOCKSIZE_3D);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_img.scalar_type(), "create_intermediates", ([&] {
            create_intermediates_kernel3d<scalar_t><<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
                input_img.data_ptr<scalar_t>(), target_img.data_ptr<scalar_t>(), intermediates.data_ptr<scalar_t>(),
                batch_size, n_channels, height, width, depth);
        }));
        return intermediates;
    }
    else if (numdims == 4) {
        // 2D case
        auto batch_size = input_img.size(0);
        auto n_channels = input_img.size(1);
        auto height = input_img.size(2);
        auto width = input_img.size(3);
        throw std::runtime_error("2D case not implemented yet");
    }
    else {
        throw std::runtime_error("Invalid input image dimensions");
    }
}