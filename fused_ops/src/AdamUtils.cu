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
// #include <math.h>
#include <c10/macros/Macros.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <assert.h>
#include <cmath>
// Local headers
#include "common.h"

using namespace at::cuda::detail;
using namespace at::native;

template <typename scalar_t>
__global__ void adam_update_fused_kernel(
    scalar_t *grad,
    scalar_t *exp_avg,
    scalar_t *exp_avg_sq,
    float beta1,
    float beta2,
    float eps,
    int64_t n_elements
) {
    CUDA_KERNEL_LOOP_TYPE(i, n_elements, int64_t) {
        grad[i] = exp_avg[i] / (beta1) / (sqrt(exp_avg_sq[i] / (beta2)) + eps);
    }
}

void adam_update_fused(torch::Tensor &grad,
                      torch::Tensor exp_avg,
                      torch::Tensor exp_avg_sq,
                      float beta1,
                      float beta2,
                      float eps) {
    // get device
    int64_t device = grad.options().device().index();
    // get device and stream
    c10::DeviceGuard guard(grad.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(grad.device().index());
    at::cuda::CUDAStreamGuard stream_guard(stream);

    // get number of elements
    int64_t n_elements = grad.numel();

    // launch kernel
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad.scalar_type(), "adam_update_fused_kernel", [&] {
            adam_update_fused_kernel<scalar_t><<<GET_BLOCKS(n_elements, BLOCKSIZE_3D), BLOCKSIZE_3D, 0, stream>>>(
                grad.data_ptr<scalar_t>(),
                exp_avg.data_ptr<scalar_t>(),
                exp_avg_sq.data_ptr<scalar_t>(),
                beta1, beta2, eps, n_elements);
        });
}
                                