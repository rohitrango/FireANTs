// Core CUDA headers
#include <cuda_runtime.h>
#include <torch/extension.h>

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
#include <iostream>

// Local headers
#include "MutualInformation.h"
#include "common.h"

using namespace at::native;

bool is_power_of_two(int num) {
    return (num > 0) && ((num & (num - 1)) == 0);
}

template <typename scalar_t, typename index_t>
__global__ void mutual_information_histogram_fwd_kernel_basic(
    scalar_t* __restrict__ input_img, scalar_t* __restrict__ target_img, int64_t* __restrict__ pab, int64_t* __restrict__ pa, int64_t* __restrict__ pb,
    index_t batch_size, index_t channels, index_t num_aggregates, index_t num_bins, index_t num_samples
) {
    using opmath_t = at::opmath_type<scalar_t>;

    // get (b, c, n)
    index_t id = blockIdx.x * blockDim.x + threadIdx.x;
    index_t agg_idx = id % num_aggregates;
    index_t ch_idx = (id / num_aggregates) % channels;
    index_t b_idx = (id / (num_aggregates * channels)) % batch_size;
    
    for (index_t n = agg_idx; n < num_samples; n += num_aggregates) {
        // load i and j
        scalar_t iimg = input_img[b_idx * channels * num_samples + ch_idx * num_samples + n];
        scalar_t jimg = target_img[b_idx * channels * num_samples + ch_idx * num_samples + n];

        // compute bin indices
        index_t i_bin_idx = static_cast<index_t>(::floor(iimg * num_bins));
        index_t j_bin_idx = static_cast<index_t>(::floor(jimg * num_bins));

        // add to histogram
        pab[j_bin_idx + num_bins * (i_bin_idx + num_bins * (agg_idx + num_aggregates * (ch_idx + channels * b_idx)))] += 1;
        pa[i_bin_idx + num_bins * (agg_idx + num_aggregates * (ch_idx + channels * b_idx))] += 1;
        pb[j_bin_idx + num_bins * (agg_idx + num_aggregates * (ch_idx + channels * b_idx))] += 1;
    }
}


std::vector<torch::Tensor> mutual_information_histogram_fwd(torch::Tensor input_img, torch::Tensor target_img, int num_bins, KernelType kernel_type) {
    // input image: [batch_size, n_channels, *]
    // target image: [batch_size, n_channels, *]
    // num_bins: int
    // kernel_type: KernelType

    // device and stream guards
    c10::DeviceGuard guard(input_img.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input_img.device().index());
    at::cuda::CUDAStreamGuard stream_guard(stream);

    CHECK_INPUT(input_img);
    CHECK_INPUT(target_img);

    // we have verified that num_bins is a power of 2 and less than or equal to 512
    if (!is_power_of_two(num_bins)) {
        throw std::runtime_error("num_bins must be a power of 2");
    }
    if (num_bins > 512) {
        throw std::runtime_error("num_bins must be less than or equal to 512");
    }

    // define number of aggregates
    int64_t num_aggregates = 65536; 
    if (num_bins > 128) {
        num_aggregates = num_aggregates / 64;
    }

    // get batch_size, channel, and number of samples
    int64_t numel = input_img.numel();
    int64_t batch_size = input_img.size(0);
    int64_t channels = input_img.size(1);
    int64_t num_samples = numel / (batch_size * channels);

    /* TODO: Different implementations for small and large bins */
    // if num_bins * num_bins <= 4096, use small bins implementation using shared memory
    // otherwise, use large bins implementation

    // in "smallBinsImpl", we store the entire joint histogram in shared memory (because there will be contention)
    // thats why we need 4096 blocks of shared memory
    // each "block" is responsible for one (b, c, n) pair

    // in "largeBinsImpl", we will use simple atomicAdd operations because it is unlikely that
    bool useSmallBinsImpl = num_bins * num_bins <= 4096;

    // determine grid size and blocksize
    dim3 blockSize(BLOCKSIZE_3D);
    dim3 gridSize(batch_size * channels * num_aggregates / BLOCKSIZE_3D);

    torch::Tensor pab = torch::zeros({batch_size, channels, num_aggregates, num_bins, num_bins}, torch::TensorOptions().device(input_img.device()).dtype(torch::kLong));
    torch::Tensor pa = torch::zeros({batch_size, channels, num_aggregates, num_bins}, torch::TensorOptions().device(input_img.device()).dtype(torch::kLong));
    torch::Tensor pb = torch::zeros({batch_size, channels, num_aggregates, num_bins}, torch::TensorOptions().device(input_img.device()).dtype(torch::kLong));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_img.scalar_type(), "mutual_information_histogram_fwd", [&] {
        if (canUse32BitIndexMath(input_img) && canUse32BitIndexMath(pab)) {
            mutual_information_histogram_fwd_kernel_basic<<<gridSize, blockSize, 0, stream>>>(
                input_img.data_ptr<scalar_t>(),
                target_img.data_ptr<scalar_t>(),
                pab.data_ptr<int64_t>(),
                pa.data_ptr<int64_t>(),
                pb.data_ptr<int64_t>(),
                static_cast<int>(batch_size),
                static_cast<int>(channels),
                static_cast<int>(num_aggregates),
                static_cast<int>(num_bins),
                static_cast<int>(num_samples)
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        else {
            mutual_information_histogram_fwd_kernel_basic<<<gridSize, blockSize, 0, stream>>>(
                input_img.data_ptr<scalar_t>(),
                target_img.data_ptr<scalar_t>(),
                pab.data_ptr<int64_t>(),
                pa.data_ptr<int64_t>(),
                pb.data_ptr<int64_t>(),
                static_cast<int64_t>(batch_size),
                static_cast<int64_t>(channels),
                static_cast<int64_t>(num_aggregates),
                static_cast<int64_t>(num_bins),
                static_cast<int64_t>(num_samples)
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    });

    float num_samples_float = static_cast<float>(num_samples);
    pa = pa.sum({2}).to(torch::kFloat) / num_samples_float;
    pb = pb.sum({2}).to(torch::kFloat) / num_samples_float;
    pab = pab.sum({2}).to(torch::kFloat) / num_samples_float;

    return {pab, pa, pb};
}
