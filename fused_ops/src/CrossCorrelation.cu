// Core CUDA headers
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>
// PyTorch CUDA headers
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

// C++ standard headers
#include <cmath>
// #include <math.h>
#include <c10/macros/Macros.h>
#include <assert.h>

// Local headers
#include "CrossCorrelation.h"
#include "common.h"

using namespace at::native;

template <typename scalar_t, typename index_t>
__global__ void create_intermediates_kernel3d(
    scalar_t* __restrict__ input_img, scalar_t* __restrict__ target_img, scalar_t* __restrict__ intermediates,
    index_t batch_size, index_t n_channels, index_t height, index_t width, index_t depth) {

    using opmath_t = at::opmath_type<scalar_t>;

    index_t n = batch_size * n_channels * height * width * depth;
    CUDA_KERNEL_LOOP_TYPE(i, n, index_t) {
        index_t b = i / (n_channels * height * width * depth);
        index_t c = (i / (height * width * depth)) % n_channels;
        index_t h = (i / (width * depth)) % height;
        index_t w = (i / depth) % width;
        index_t d = i % depth;

        // read input and target images
        opmath_t I = input_img[i];
        opmath_t J = target_img[i];

        // compute intermediate values
        opmath_t I2 = I * I;
        opmath_t J2 = J * J;
        opmath_t IJ = I * J;

        // write intermediate values
        index_t out_idx_start = d + depth * (w + width * (h + height * (c + 5 * n_channels * (b))));
        index_t out_offset = n_channels * height * width * depth;
        intermediates[out_idx_start] = I;
        intermediates[out_idx_start + out_offset] = J;
        intermediates[out_idx_start + 2 * out_offset] = I2;
        intermediates[out_idx_start + 3 * out_offset] = J2;
        intermediates[out_idx_start + 4 * out_offset] = IJ;
    }
}

template <typename scalar_t, typename index_t> 
C10_LAUNCH_BOUNDS_1(BLOCKSIZE_3D)
__global__ void cc3d_fwd_interm_v1_kernel(
    scalar_t* __restrict__ interm, scalar_t* __restrict__ out, 
    index_t kernel_volume, Reduction reduce, 
    index_t batch_size, index_t n_out_channels, index_t height, index_t width, index_t depth, 
    index_t out_size, float nr, float dr) {

    using opmath_t = at::opmath_type<scalar_t>;

    // initialize output value
    // scalar_t out_val = 0;
    __shared__ opmath_t out_val[BLOCKSIZE_3D];
    out_val[threadIdx.x] = 0;

    // initialize thread id
    index_t n = batch_size * n_out_channels * height * width * depth;
    CUDA_KERNEL_LOOP_TYPE(i, n, index_t) {
        index_t b = i / (n_out_channels * height * width * depth);
        index_t c = (i / (height * width * depth)) % n_out_channels;
        index_t h = (i / (width * depth)) % height;
        index_t w = (i / depth) % width;
        index_t d = i % depth;

        // int inp_index_common = b * (5 * n_out_channels * height * width * depth) + h * (width * depth) + w * depth + d;
        index_t input_idx_common = d + depth * (w + width * (h + height * (5 * n_out_channels * b)));
        index_t spatial_mult = depth * width * height;

        // Get variables (all means)
        opmath_t Ival = interm[input_idx_common + c * spatial_mult];
        opmath_t Jval = interm[input_idx_common + (c + n_out_channels) * spatial_mult];
        opmath_t I2val = interm[input_idx_common + (c + 2 * n_out_channels) * spatial_mult];
        opmath_t J2val = interm[input_idx_common + (c + 3 * n_out_channels) * spatial_mult];
        opmath_t IJval = interm[input_idx_common + (c + 4 * n_out_channels) * spatial_mult];
        
        // compute cross-correlation
        opmath_t kv = static_cast<opmath_t>(kernel_volume);
        opmath_t aval = kv * (IJval - Ival * Jval);
        opmath_t bval = kv * (I2val - Ival * Ival);
        opmath_t cval = kv * (J2val - Jval * Jval);
        // cross-correlation
        kv = (aval * aval + nr)/(bval * cval + dr);

        if (reduce == Reduction::NONE) {
            index_t out_idx = d + depth * (w + width * (h + height * (c + n_out_channels * b)));
            out[out_idx] = static_cast<scalar_t>(kv);
        }
        else {
            // int out_idx = i % out_size;
            // gpuAtomicAdd(out + out_idx, static_cast<scalar_t>(kv));
            out_val[threadIdx.x] += kv;
        }
    }
    // collected output value
    if (reduce != Reduction::NONE) {
        __syncthreads();
        // int out_idx = d + depth * (w + width * (h + height * (c + n_out_channels * b)));
        // out[out_idx] = static_cast<scalar_t>(out_val);
        for (int i = BLOCKSIZE_3D / 2; i > 0; i /= 2) {
            if (threadIdx.x < i) {
                out_val[threadIdx.x] += out_val[threadIdx.x + i];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            out[blockIdx.x] = out_val[0];
        }
    }
}

/*
This kernel computes the quantities required to compute the gradients of the input and target images.

Modifying the intermediates in-place will save memory, and depending on the value of `compute_grad_target`
we compute the other quantities.

mean(I) = mu
mean(J) = rho
mean(I**2) = mu2
mean(J**2) = rho2
mean(IJ) = IJ

A = k * (IJ - mu * rho)
B = k * (mu2 - mu**2)
C = k * (rho2 - rho**2)

In intermediates, we have (mu, rho, mu2, rho2, IJ) for each channel.

*/
template <typename scalar_t, typename index_t>
__global__ void cc3d_bwd_modify_interm_v1_kernel(
    scalar_t* __restrict__ interm, scalar_t* __restrict__ input_img, scalar_t* __restrict__ target_img, 
    scalar_t* __restrict__ grad_input_img, scalar_t* __restrict__ grad_target_img, scalar_t* __restrict__ grad_output,
    index_t kernel_size, float nr, float dr, Reduction reduction, bool compute_grad_input, bool compute_grad_target,
    index_t batch_size, index_t n_out_channels, index_t height, index_t width, index_t depth, scalar_t grad_output_val) {

    // define opmath type
    using opmath_t = at::opmath_type<scalar_t>;

    float kernel_volume = (float)kernel_size;
    kernel_volume = powf(kernel_volume, 3);

    // input image dimensions
    index_t n = batch_size * n_out_channels * height * width * depth;

    CUDA_KERNEL_LOOP_TYPE(i, n, index_t) {
        index_t b = i / (n_out_channels * height * width * depth);
        index_t c = (i / (height * width * depth)) % n_out_channels;
        index_t h = (i / (width * depth)) % height;
        index_t w = (i / depth) % width;
        index_t d = i % depth;

        index_t input_idx_common = d + depth * (w + width * (h + height * (5 * n_out_channels * b)));
        index_t spatial_mult = depth * width * height;

        // get intermediate values
        opmath_t Ival = interm[input_idx_common + c * spatial_mult];
        opmath_t Jval = interm[input_idx_common + (c + n_out_channels) * spatial_mult];
        opmath_t I2val = interm[input_idx_common + (c + 2 * n_out_channels) * spatial_mult];
        opmath_t J2val = interm[input_idx_common + (c + 3 * n_out_channels) * spatial_mult];
        opmath_t IJval = interm[input_idx_common + (c + 4 * n_out_channels) * spatial_mult];

        // compute intermediate values
        opmath_t aval = kernel_volume * (IJval - Ival * Jval);
        opmath_t bval = kernel_volume * (I2val - Ival * Ival);
        opmath_t cval = kernel_volume * (J2val - Jval * Jval);

        // compute output gradient
        opmath_t gOVal = reduction == Reduction::NONE ? grad_output[i] : grad_output_val;
        gOVal = 2 * gOVal * aval / (bval * cval + dr);    // 2gn / A = D

        // add this term for further use
        bval += dr;
        cval += dr;

        // write the first three entries in the intermediate array 
        interm[input_idx_common + c * spatial_mult] = gOVal;   // D
        interm[input_idx_common + (c + n_out_channels) * spatial_mult] = gOVal * aval / bval;
        interm[input_idx_common + (c + 2 * n_out_channels) * spatial_mult] = gOVal * (aval / bval * Ival - Jval);
        if (compute_grad_target) {
            interm[input_idx_common + (c + 3 * n_out_channels) * spatial_mult] = gOVal * aval / cval;
            interm[input_idx_common + (c + 4 * n_out_channels) * spatial_mult] = gOVal * (aval / cval * Jval - Ival);
        }
        
    } // cuda kernel loop
}

template <typename scalar_t, typename index_t>
__global__ void cc3d_bwd_compute_grads_kernel(
    scalar_t* __restrict__ interm, scalar_t* __restrict__ input_img, scalar_t* __restrict__ target_img, 
    scalar_t* __restrict__ grad_input_img, scalar_t* __restrict__ grad_target_img,
    index_t batch_size, index_t n_out_channels, index_t height, index_t width, index_t depth) {

    using opmath_t = at::opmath_type<scalar_t>;
    index_t n = batch_size * n_out_channels * height * width * depth;

    CUDA_KERNEL_LOOP_TYPE(i, n, index_t) {
        index_t b = i / (n_out_channels * height * width * depth);
        index_t c = (i / (height * width * depth)) % n_out_channels;
        index_t h = (i / (width * depth)) % height;
        index_t w = (i / depth) % width;
        index_t d = i % depth;

        // get intermediate values
        index_t input_idx_common = d + depth * (w + width * (h + height * (5 * n_out_channels * b)));
        index_t spatial_mult = depth * width * height;

        // get image values  
        opmath_t Ival = input_img[i];
        opmath_t Jval = target_img[i];

        opmath_t gini_a = interm[input_idx_common + c * spatial_mult];
        opmath_t gini_b = interm[input_idx_common + (c + n_out_channels) * spatial_mult];
        opmath_t gini_mu1 = interm[input_idx_common + (c + 2 * n_out_channels) * spatial_mult];

        // compute grad_input
        grad_input_img[i] = gini_a * Jval - gini_b * Ival + gini_mu1;

        if (grad_target_img != nullptr) {
            // get gini_
            opmath_t gini_c = interm[input_idx_common + (c + 3 * n_out_channels) * spatial_mult];
            opmath_t gini_mu2 = interm[input_idx_common + (c + 4 * n_out_channels) * spatial_mult];
            // compute grad_target
            grad_target_img[i] = gini_a * Ival - gini_c * Jval + gini_mu2;
        }
    }
}

torch::Tensor cc3d_fwd_interm_v1(torch::Tensor intermediates, int64_t kernel_volume, Reduction reduction, float nr, float dr) {
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
    int64_t n = intermediates.numel() / 5;
    CHECK_INPUT(intermediates);

    // device and stream guards
    c10::DeviceGuard guard(intermediates.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(intermediates.device().index());
    at::cuda::CUDAStreamGuard stream_guard(stream);
    
    int64_t batch_size = intermediates.size(0);
    int64_t n_channels = intermediates.size(1);
    int64_t height = intermediates.size(2);
    int64_t width = intermediates.size(3);
    int64_t depth = intermediates.size(4);
    assert(n_channels % 5 == 0);
    int64_t n_out_channels = n_channels / 5;   // one output channel for each block of 5 input channels

    // initialize blocks
    dim3 blockSize(BLOCKSIZE_3D);
    int64_t gridsize = (n + blockSize.x - 1) / blockSize.x;
    gridsize = std::min(gridsize, static_cast<int64_t>(65536));
    dim3 gridSize(gridsize);

    // initialize output tensor
    int64_t out_size = 0;
    torch::Tensor out;
    if (reduction == Reduction::NONE) {
        out = torch::zeros({batch_size, n_out_channels, height, width, depth}, intermediates.scalar_type());
    } else {
        // each grid block will store the final output
        out = torch::zeros({gridsize}, intermediates.scalar_type());
    }
    out = out.to(intermediates.device());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, intermediates.scalar_type(), "cc3d_fwd_interm_v1", ([&] {
        if (canUse32BitIndexMath(intermediates) && canUse32BitIndexMath(out)) {
            cc3d_fwd_interm_v1_kernel<scalar_t><<<gridSize, blockSize, 0, stream>>>(
                intermediates.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), static_cast<int>(kernel_volume), reduction, 
                static_cast<int>(batch_size), static_cast<int>(n_out_channels), static_cast<int>(height), static_cast<int>(width), static_cast<int>(depth), static_cast<int>(out_size), nr, dr);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            cc3d_fwd_interm_v1_kernel<scalar_t><<<gridSize, blockSize, 0, stream>>>(
                intermediates.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), kernel_volume, reduction, 
                batch_size, n_out_channels, height, width, depth, out_size, nr, dr);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
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

torch::Tensor cc3d_bwd_modify_interm_v1(torch::Tensor intermediates, torch::Tensor input_img, torch::Tensor target_img, 
    torch::Tensor grad_output, 
    std::optional<torch::Tensor> grad_input_img, std::optional<torch::Tensor> grad_target_img, 
    int64_t kernel_size, float nr, float dr, Reduction reduction) {
    /*
    intermediates: [batch_size, 5*n_channels, height, width, depth]
       * note that intermediates contains 5 blocks 
           block of I, J, I**2, J**2, IJ

    kernel_volume: int
    reduction: Reduction

    input_img: [batch_size, n_channels, height, width, depth]
    target_img: [batch_size, n_channels, height, width, depth]
    grad_output: [batch_size, n_out_channels, height, width, depth]
    grad_input_img: [batch_size, n_channels, height, width, depth] (optional)
    grad_target_img: [batch_size, n_channels, height, width, depth] (optional)
    */

    if (!grad_input_img.has_value() && !grad_target_img.has_value()) {
        throw std::runtime_error("At least one of grad_input_img or grad_target_img must be provided");
    }

    // device and stream guards
    c10::DeviceGuard guard(intermediates.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(intermediates.device().index());
    at::cuda::CUDAStreamGuard stream_guard(stream);

    int64_t n = intermediates.numel() / 5;
    int64_t batch_size = intermediates.size(0);
    int64_t n_channels = intermediates.size(1);
    int64_t height = intermediates.size(2);
    int64_t width = intermediates.size(3);
    int64_t depth = intermediates.size(4);
    assert(n_channels % 5 == 0);
    int64_t n_out_channels = n_channels / 5;   // one output channel for each block of 5 input channels

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

    // for mean, dont normalize the gradient yet (let's see if more stable gradients can be achieved)
    // if (reduction == Reduction::MEAN) {
    //     grad_output_val = grad_output_val / n;
    // }
    dim3 blockSize(BLOCKSIZE_3D);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    bool compute_grad_input = grad_input_img.has_value();
    bool compute_grad_target = grad_target_img.has_value();

    // this 
    TORCH_CHECK(compute_grad_input, "compute_grad_input must be true - why would you call this function otherwise?");

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, intermediates.scalar_type(), "cc3d_bwd_interm_v1", ([&] {
        // initialize output tensors
        scalar_t grad_output_val = 1;
        if (reduction == Reduction::NONE) {
            // do nothing
        }
        else {
            // only a single value, extract it
            grad_output_val = grad_output.cpu().item<scalar_t>();
        }
        // initialize output gradient pointers if they exist
        scalar_t* grad_input_img_ptr = grad_input_img.has_value() ? grad_input_img.value().data_ptr<scalar_t>() : nullptr;
        scalar_t* grad_target_img_ptr = grad_target_img.has_value() ? grad_target_img.value().data_ptr<scalar_t>() : nullptr;

        if (canUse32BitIndexMath(intermediates) && canUse32BitIndexMath(grad_output)) {
            cc3d_bwd_modify_interm_v1_kernel<scalar_t><<<gridSize, blockSize, 0, stream>>>(
                intermediates.data_ptr<scalar_t>(), input_img.data_ptr<scalar_t>(), target_img.data_ptr<scalar_t>(),  
                grad_input_img_ptr, grad_target_img_ptr, grad_output.data_ptr<scalar_t>(),
                static_cast<int>(kernel_size), nr, dr, reduction, compute_grad_input, compute_grad_target,
                static_cast<int>(batch_size), static_cast<int>(n_out_channels), static_cast<int>(height), static_cast<int>(width), static_cast<int>(depth), grad_output_val);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        else {
            cc3d_bwd_modify_interm_v1_kernel<scalar_t><<<gridSize, blockSize, 0, stream>>>(
                intermediates.data_ptr<scalar_t>(), input_img.data_ptr<scalar_t>(), target_img.data_ptr<scalar_t>(),  
                grad_input_img_ptr, grad_target_img_ptr, grad_output.data_ptr<scalar_t>(),
                kernel_size, nr, dr, reduction, compute_grad_input, compute_grad_target,
                batch_size, n_out_channels, height, width, depth, grad_output_val);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }));
    
    return torch::Tensor();
}

void cc3d_bwd_compute_grads(torch::Tensor intermediates, torch::Tensor input_img, torch::Tensor target_img, torch::Tensor grad_input_img, std::optional<torch::Tensor> grad_target_img) {
    /*
    intermediates: [batch_size, 5*n_channels, height, width, depth]
    input_img: [batch_size, n_channels, height, width, depth]
    target_img: [batch_size, n_channels, height, width, depth]
    grad_input_img: [batch_size, n_channels, height, width, depth]
    grad_target_img: [batch_size, n_channels, height, width, depth]
    */
    int64_t n = intermediates.numel() / 5;
    int64_t batch_size = intermediates.size(0);
    int64_t n_channels = intermediates.size(1);
    int64_t height = intermediates.size(2);
    int64_t width = intermediates.size(3);
    int64_t depth = intermediates.size(4);
    assert(n_channels % 5 == 0);
    int64_t n_out_channels = n_channels / 5;   // one output channel for each block of 5 input channels

    // device and stream guards
    c10::DeviceGuard guard(intermediates.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(intermediates.device().index());
    at::cuda::CUDAStreamGuard stream_guard(stream);

    CHECK_INPUT(intermediates);
    CHECK_INPUT(input_img);
    CHECK_INPUT(target_img);

    dim3 blockSize(BLOCKSIZE_3D);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, intermediates.scalar_type(), "cc3d_bwd_compute_grads", ([&] {
        // initialize output gradient pointers if they exist
        scalar_t* grad_input_img_ptr = grad_input_img.data_ptr<scalar_t>();
        scalar_t* grad_target_img_ptr = grad_target_img.has_value() ? grad_target_img.value().data_ptr<scalar_t>() : nullptr;

        if (canUse32BitIndexMath(intermediates) && canUse32BitIndexMath(grad_input_img)) {
            cc3d_bwd_compute_grads_kernel<scalar_t><<<gridSize, blockSize, 0, stream>>>(
                intermediates.data_ptr<scalar_t>(), input_img.data_ptr<scalar_t>(), target_img.data_ptr<scalar_t>(),  
                grad_input_img_ptr, grad_target_img_ptr, 
                static_cast<int>(batch_size), static_cast<int>(n_out_channels), static_cast<int>(height), static_cast<int>(width), static_cast<int>(depth)
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            cc3d_bwd_compute_grads_kernel<scalar_t><<<gridSize, blockSize, 0, stream>>>(
                intermediates.data_ptr<scalar_t>(), input_img.data_ptr<scalar_t>(), target_img.data_ptr<scalar_t>(),  
                grad_input_img_ptr, grad_target_img_ptr, 
                batch_size, n_out_channels, height, width, depth
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }));
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

    int64_t numdims = input_img.dim();

    // device and stream guards
    c10::DeviceGuard guard(input_img.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input_img.device().index());
    at::cuda::CUDAStreamGuard stream_guard(stream);

    CHECK_INPUT(target_img);
    CHECK_INPUT(input_img);
    CHECK_INPUT(intermediates);

    if (numdims == 5) {
        // 3D case
        int64_t batch_size = input_img.size(0);
        int64_t n_channels = input_img.size(1);
        int64_t height = input_img.size(2);
        int64_t width = input_img.size(3);
        int64_t depth = input_img.size(4);
        int64_t n = input_img.numel();

        // torch::Tensor intermediates = torch::zeros({batch_size, 5 * n_channels, height, width, depth}, input_img.scalar_type());
        // intermediates = intermediates.to(input_img.device());
        dim3 blockSize(BLOCKSIZE_3D);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_img.scalar_type(), "create_intermediates", ([&] {
            if (canUse32BitIndexMath(intermediates) && canUse32BitIndexMath(input_img)) {
                create_intermediates_kernel3d<scalar_t><<<gridSize, blockSize, 0, stream>>>(
                    input_img.data_ptr<scalar_t>(), target_img.data_ptr<scalar_t>(), intermediates.data_ptr<scalar_t>(),
                    static_cast<int>(batch_size), static_cast<int>(n_channels), static_cast<int>(height), static_cast<int>(width), static_cast<int>(depth)
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            } else {
                create_intermediates_kernel3d<scalar_t><<<gridSize, blockSize, 0, stream>>>(
                    input_img.data_ptr<scalar_t>(), target_img.data_ptr<scalar_t>(), intermediates.data_ptr<scalar_t>(),
                    batch_size, n_channels, height, width, depth);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }));
        return intermediates;
    }
    else if (numdims == 4) {
        // 2D case
        int64_t batch_size = input_img.size(0);
        int64_t n_channels = input_img.size(1);
        int64_t height = input_img.size(2);
        int64_t width = input_img.size(3);
        throw std::runtime_error("2D case not implemented yet");
    }
    else {
        throw std::runtime_error("Invalid input image dimensions");
    }
}