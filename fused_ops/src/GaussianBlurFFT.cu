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
#include <ATen/cuda/detail/KernelUtils.h>
// C++ standard headers
// #include <math.h>
#include <c10/macros/Macros.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <assert.h>
#include <cmath>
// Local headers
#include "GaussianBlurFFT.h"
#include "common.h"

using namespace at::cuda::detail;
using namespace at::native;

template <typename scalar_t>
__global__ void gaussian_blur_fft2_kernel(
    scalar_t *im_fft,
    int64_t batch_size, int64_t channel_size, int64_t H, int64_t W, int64_t n_elements,
    int64_t ys, int64_t xs, int64_t ye, int64_t xe, 
    float multiplier ) {
    // loop over
    using opmath_t = at::opmath_type<scalar_t>;

    CUDA_KERNEL_LOOP_TYPE(i, n_elements, int64_t) {
        int64_t w = i % W;
        int64_t h = (i / W) % H;
        // int64_t c = (i / (H * W)) % channel_size;
        // int64_t b = i / (channel_size * H * W);
        // get input value
        opmath_t inp = im_fft[i];
        // get exp
        float sigma_y = static_cast<float>(H) / 4.0f;
        float sigma_x = static_cast<float>(W) / 4.0f;
        float y_freq = static_cast<float>(ys + h) / sigma_y;
        float x_freq = static_cast<float>(xs + w) / sigma_x;
        opmath_t exp_val = exp(-0.5 * (y_freq * y_freq + x_freq * x_freq));
        // get value
        inp = inp * exp_val * static_cast<opmath_t>(multiplier);
        // store it
        im_fft[i] = inp;
    }
}

template <typename scalar_t>
__global__ void gaussian_blur_fft3_kernel(
    scalar_t *im_fft,
    int64_t batch_size, int64_t channel_size, int64_t D, int64_t H, int64_t W, int64_t n_elements,
    int64_t zs, int64_t ys, int64_t xs, int64_t ze, int64_t ye, int64_t xe, 
    float multiplier ) {

    using opmath_t = at::opmath_type<scalar_t>;
    // loop over
    CUDA_KERNEL_LOOP_TYPE(i, n_elements, int64_t) {
        int64_t w = i % W;
        int64_t h = (i / W) % H;
        int64_t d = (i / (H * W)) % D;
        // int64_t c = (i / (D * H * W)) % channel_size;
        // int64_t b = i / (channel_size * D * H * W);
        // get input value
        opmath_t inp = im_fft[i];
        // get exp
        float sigma_z = static_cast<float>(D) / 4.0f;
        float sigma_y = static_cast<float>(H) / 4.0f;
        float sigma_x = static_cast<float>(W) / 4.0f;
        float z_freq = static_cast<float>(zs + d) / sigma_z;
        float y_freq = static_cast<float>(ys + h) / sigma_y;
        float x_freq = static_cast<float>(xs + w) / sigma_x;
        opmath_t exp_val = exp(-0.5 * (y_freq * y_freq + x_freq * x_freq + z_freq * z_freq));
        // get value
        inp = inp * exp_val * static_cast<opmath_t>(multiplier);
        // store it
        im_fft[i] = inp;
    }
}

void gaussian_blur_fft2(torch::Tensor &im_fft,
                int64_t ys, int64_t xs, int64_t ye, int64_t xe, 
                float multiplier ) {
    // get device and stream
    c10::DeviceGuard guard(im_fft.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(im_fft.device().index());
    at::cuda::CUDAStreamGuard stream_guard(stream);

    // get number of elements
    int64_t batch_size = im_fft.size(0);
    int64_t channel_size = im_fft.size(1);
    int64_t H = im_fft.size(2);
    int64_t W = im_fft.size(3);
    int64_t n_elements = batch_size * channel_size * H * W;

    int64_t gridSize = std::min(static_cast<int64_t>(65536), GET_BLOCKS_v2(n_elements, BLOCKSIZE_3D));

    // launch kernel
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        im_fft.scalar_type(), "gaussian_blur_fft2", [&] {
            gaussian_blur_fft2_kernel<scalar_t><<<gridSize, BLOCKSIZE_3D, 0, stream>>>(
                im_fft.data_ptr<scalar_t>(),
                batch_size, channel_size, H, W, n_elements,
                ys, xs, ye, xe, multiplier);
            }
    );
}
                                
void gaussian_blur_fft3(torch::Tensor &im_fft,
                int64_t zs, int64_t ys, int64_t xs, int64_t ze, int64_t ye, int64_t xe, 
                float multiplier ) {
    // get device and stream
    c10::DeviceGuard guard(im_fft.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(im_fft.device().index());
    at::cuda::CUDAStreamGuard stream_guard(stream);

    // get number of elements
    int64_t batch_size = im_fft.size(0);
    int64_t channel_size = im_fft.size(1);
    int64_t D = im_fft.size(2);
    int64_t H = im_fft.size(3);
    int64_t W = im_fft.size(4);
    int64_t n_elements = batch_size * channel_size * D * H * W;

    int64_t gridSize = std::min(static_cast<int64_t>(65536), GET_BLOCKS_v2(n_elements, BLOCKSIZE_3D));

    // launch kernel
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        im_fft.scalar_type(), "gaussian_blur_fft3", [&] {
            gaussian_blur_fft3_kernel<scalar_t><<<gridSize, BLOCKSIZE_3D, 0, stream>>>(
                im_fft.data_ptr<scalar_t>(),
                batch_size, channel_size, D, H, W, n_elements,
                zs, ys, xs, ze, ye, xe, multiplier);
            }
    );
}
                    