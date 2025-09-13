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
#define ONE_OVER_SQRT_2PI 0.3989422804014327 

using namespace at::native;

bool is_power_of_two(int num) {
    return (num > 0) && ((num & (num - 1)) == 0);
}

int div_ceil(int num, int div) {
    return ((num + div - 1) / div);
}

template <typename scalar_t, typename index_t>
__device__ scalar_t prob_func(scalar_t x, index_t bin_idx, index_t num_bins, KernelType kernel_type, float sigma_ratio) {
    float bin_width = 1.0 / num_bins;
    float bin_center = (bin_idx + 0.5) * bin_width;
    float sigma = bin_width / 2 * sigma_ratio;
    float abdiff = fabs((x - bin_center) / bin_width);
    switch (kernel_type) {
        case KernelType::GAUSSIAN:
            // return ONE_OVER_SQRT_2PI / sigma * __expf(-0.5 * powf((x - bin_center) / sigma, 2));
            return __expf(-0.5 * powf((x - bin_center) / sigma, 2));
        case KernelType::BSPLINE:
            if (abdiff < 1) 
                return (4 + abdiff * abdiff * (-6 + 3 * abdiff))/6.0;
            else if (abdiff < 2)
                return (2 - abdiff) * (2 - abdiff) * (2 - abdiff) / 6.0;
            return 0;
        case KernelType::DELTA:
            return fabsf((x - bin_center)) < bin_width ? 1 : 0;
        default:
            return 0;
    }
}

template <typename scalar_t, typename index_t>
__device__ scalar_t kernel_grad(scalar_t x, index_t bin_idx, index_t num_bins, KernelType kernel_type, float sigma_ratio) {
    float bin_width = 1.0 / num_bins;
    float bin_center = (bin_idx + 0.5) * bin_width;
    float sigma = bin_width / 2 * sigma_ratio;
    float abdiff = fabs((x - bin_center) / bin_width);
    switch (kernel_type) {
        case KernelType::GAUSSIAN:
            return prob_func(x, bin_idx, num_bins, kernel_type, sigma_ratio) * (bin_center - x) / sigma / sigma;

        case KernelType::BSPLINE:
            if (abdiff < 1) 
                return (-2 + 1.5*abdiff)*abdiff;
            else if (abdiff < 2)
                return -(2 - abdiff)*(2 - abdiff) / 2;
            return 0;
            
        case KernelType::DELTA:
            return fabsf((x - bin_center)) < bin_width ? 1 : 0;

        default:
            return 0;
    } 
}

template <typename scalar_t, typename index_t>
__global__ void mutual_information_histogram_bwd_kernel_basic(
    scalar_t* __restrict__ input_img, scalar_t* __restrict__ target_img, 
    scalar_t* __restrict__ grad_pab, scalar_t* __restrict__ grad_pa, scalar_t* __restrict__ grad_pb,
    scalar_t* __restrict__ grad_input_img, scalar_t* __restrict__ grad_target_img,
    index_t batch_size, index_t channels, index_t num_bins, index_t num_samples, KernelType kernel_type, float minval, float maxval,
    float sigma_ratio
) {
    using opmath_t = at::opmath_type<scalar_t>;

    opmath_t minval_op = static_cast<opmath_t>(minval);
    opmath_t maxval_op = static_cast<opmath_t>(maxval);
    
    // Shared memory for grad_pa, grad_pb and kernel values
    __shared__ opmath_t shared_grad_pa[BLOCKSIZE_3D];
    __shared__ opmath_t shared_grad_pb[BLOCKSIZE_3D];
    __shared__ opmath_t shared_prob_pa[BLOCKSIZE_3D];
    __shared__ opmath_t shared_prob_pb[BLOCKSIZE_3D];

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Get global thread index and decompose into bin_idx and (b,c,sample_idx)
    index_t bin_idx = static_cast<index_t>(idx % num_bins);
    idx = idx / num_bins;
    index_t sample_idx = static_cast<index_t>((idx) % num_samples);
    idx = idx / num_samples;
    index_t c_idx = static_cast<index_t>((idx / num_bins) % channels);
    idx = idx / channels;
    index_t b_idx = static_cast<index_t>(idx);
    // get batch index of starting thread
    index_t b_start = static_cast<index_t>((blockIdx.x * blockDim.x) / (num_samples * num_bins * channels));

    // while (b_start < batch_size) { 
    if (b_start < batch_size) {
        // if out of bounds, compute some other thing and we will not write to memory
        bool out_of_bounds = b_idx >= batch_size;
        b_idx -= (out_of_bounds ? 1 : 0);

        //  processing gradients for (b, c, n)
        opmath_t shared_grad_pa_j = grad_pa[bin_idx + num_bins * (c_idx + channels * b_idx)];  // (b, c, i)
        opmath_t shared_grad_pb_j = grad_pb[bin_idx + num_bins * (c_idx + channels * b_idx)];  // (b, c, i)
        // load I_k and J_k
        opmath_t Ik = static_cast<opmath_t>(input_img[sample_idx + num_samples * (c_idx + channels * b_idx)]);
        opmath_t Jk = static_cast<opmath_t>(target_img[sample_idx + num_samples * (c_idx + channels * b_idx)]);
        // normalize
        Ik = (Ik - minval_op) / (maxval_op - minval_op);
        Jk = (Jk - minval_op) / (maxval_op - minval_op);

        // compute grad and kernel values
        shared_prob_pa[threadIdx.x] = prob_func(Ik, bin_idx, num_bins, kernel_type, sigma_ratio);
        shared_prob_pb[threadIdx.x] = prob_func(Jk, bin_idx, num_bins, kernel_type, sigma_ratio);
        shared_grad_pa[threadIdx.x] = kernel_grad(Ik, bin_idx, num_bins, kernel_type, sigma_ratio);
        shared_grad_pb[threadIdx.x] = kernel_grad(Jk, bin_idx, num_bins, kernel_type, sigma_ratio);
        __syncthreads();

        // compute sum for prob func
        opmath_t wi, wj, wiprime, wjprime;
        wi = shared_prob_pa[threadIdx.x];
        wj = shared_prob_pb[threadIdx.x];
        wiprime = shared_grad_pa[threadIdx.x];
        wjprime = shared_grad_pb[threadIdx.x];
        for (index_t i = num_bins / 2; i > 0; i /= 2) {
            if (bin_idx < i) {
                shared_prob_pa[threadIdx.x] += shared_prob_pa[threadIdx.x + i];
                shared_prob_pb[threadIdx.x] += shared_prob_pb[threadIdx.x + i];
                shared_grad_pa[threadIdx.x] += shared_grad_pa[threadIdx.x + i];
                shared_grad_pb[threadIdx.x] += shared_grad_pb[threadIdx.x + i];
            }
            __syncthreads();
        }
        opmath_t wisum = shared_prob_pa[threadIdx.x - bin_idx];
        opmath_t wjsum = shared_prob_pb[threadIdx.x - bin_idx];
        opmath_t wiprimesum = shared_grad_pa[threadIdx.x - bin_idx];
        opmath_t wjprimesum = shared_grad_pb[threadIdx.x - bin_idx];
        // set pa = w/wsum
        shared_prob_pa[threadIdx.x] = wi / wisum;
        shared_prob_pb[threadIdx.x] = wj / wjsum;
        shared_grad_pa[threadIdx.x] = (wiprime - wi * wiprimesum / wisum) / wisum;
        shared_grad_pb[threadIdx.x] = (wjprime - wj * wjprimesum / wjsum) / wjsum;
        __syncthreads();

        // compute sub gradient w.r.t. Ik and Jk (for the jth bin)
        opmath_t Ik_grad;
        opmath_t Jk_grad;
        // get gradient w.r.t. pa and pb
        Ik_grad = shared_grad_pa_j * shared_grad_pa[threadIdx.x];
        Jk_grad = shared_grad_pb_j * shared_grad_pb[threadIdx.x];

        scalar_t shared_grad_pab;

        // load the jth bar
        for (index_t i = 0; i < num_bins; i++) {
            // load pab = g[i, j]
            shared_grad_pab = grad_pab[bin_idx + num_bins * (i + num_bins * (c_idx + channels * b_idx))];   // load [b, c, i, :]
            // __syncthreads();
            // gradI += g[i, j] w'(Ik, i) p(Jk, j)
            Ik_grad = Ik_grad + shared_grad_pab * shared_grad_pa[threadIdx.x - bin_idx + i] * shared_prob_pb[threadIdx.x];
            // gradJ += g[i, j] w'(Jk, j) p(Ik, i)
            Jk_grad = Jk_grad + shared_grad_pab * shared_grad_pb[threadIdx.x] * shared_prob_pa[threadIdx.x - bin_idx + i];
        }
        __syncthreads();

        // store sum of Ik_grad and Jk_grad (re-use shared_grad_pa and shared_grad_pb)
        shared_grad_pa[threadIdx.x] = Ik_grad;
        shared_grad_pb[threadIdx.x] = Jk_grad;
        __syncthreads();

        // log(n) reduction
        for (index_t i = num_bins / 2; i > 0; i /= 2) {
            if (bin_idx < i) {
                shared_grad_pa[threadIdx.x] += shared_grad_pa[threadIdx.x + i];
                shared_grad_pb[threadIdx.x] += shared_grad_pb[threadIdx.x + i];
            }
            __syncthreads();
        }

        // for bin_idx = 0, store the values
        if (bin_idx == 0 && !out_of_bounds) {
            grad_input_img[sample_idx + num_samples * (c_idx + channels * b_idx)] = shared_grad_pa[threadIdx.x] / num_samples;
            if (grad_target_img) {
                grad_target_img[sample_idx + num_samples * (c_idx + channels * b_idx)] = shared_grad_pb[threadIdx.x] / num_samples;
            }
        }
        __syncthreads();
            
        // update counter
        // idx += blockDim.x * gridDim.x;
        // bin_idx = idx % num_bins;
        // sample_idx = (idx / num_bins) % num_samples;
        // c_idx = (idx / (num_samples * num_bins)) % channels;
        // b_idx = idx / (num_samples * num_bins * channels);
        // b_start = (idx - threadIdx.x) / (num_samples * num_bins * channels);
    }
}

void mutual_information_histogram_bwd(torch::Tensor &input_img, torch::Tensor &target_img, torch::Tensor &grad_pab, torch::Tensor &grad_pa, torch::Tensor &grad_pb, int num_bins, torch::Tensor &grad_input_img, std::optional<torch::Tensor> &grad_target_img, KernelType kernel_type, float minval, float maxval, float sigma_ratio) {
    // input image: [batch_size, n_channels, *]
    // target_image: [batch_size, n_channels, *]
    // grad_pab: [batch_size, channels, num_bins, num_bins]
    // grad_pa: [batch_size, channels, num_bins]
    // grad_pb: [batch_size, channels, num_bins]
    // num_bins: int
    // kernel_type: KernelType
    // grad_input_img: [batch_size, n_channels, *]
    // grad_target_img: [batch_size, n_channels, *] if provided else none

    // device and stream guards
    c10::DeviceGuard guard(input_img.device());
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(input_img.device().index());
    at::cuda::CUDAStreamGuard stream_guard(stream);

    CHECK_INPUT(input_img);
    CHECK_INPUT(target_img);
    CHECK_INPUT(grad_pab);
    CHECK_INPUT(grad_pa);
    CHECK_INPUT(grad_pb);
    CHECK_INPUT(grad_input_img);
    if (grad_target_img.has_value()) {
        CHECK_INPUT(grad_target_img.value());
    }
    // we have verified that num_bins is a power of 2 and less than or equal to 512
    if (!is_power_of_two(num_bins)) {
        throw std::runtime_error("num_bins must be a power of 2");
    }
    if (num_bins > 512) {
        throw std::runtime_error("num_bins must be less than or equal to 512");
    }

    // define batch size, etc
    int64_t batch_size = input_img.size(0);
    int64_t channels = input_img.size(1);
    int64_t num_samples = input_img.numel() / (batch_size * channels);

    dim3 blockSize(BLOCKSIZE_3D);
    // int effective_thread_size = BLOCKSIZE_3D / num_bins;   // if there are 64 bins, then effective number of threads = 512 / 64 = 8 (8 individual samples are worked on within a block)
    // dim3 gridSize(std::min(GET_BLOCKS_v2(batch_size * channels * num_samples, effective_thread_size), static_cast<int64_t>(65536)));
    // dim3 gridSize(GET_BLOCKS_v2(batch_size * channels * num_samples, effective_thread_size));
    dim3 gridSize(GET_BLOCKS_v2(batch_size * channels * num_samples * num_bins, BLOCKSIZE_3D));

    // constexpr int64_t max_int = std::numeric_limits<int>::max();
    // bool isLargeIndex = (batch_size * channels * num_samples * num_bins) > max_int;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_img.scalar_type(), "mutual_information_histogram_bwd", [&] {
        if (canUse32BitIndexMath(input_img)) {
            mutual_information_histogram_bwd_kernel_basic<<<gridSize, blockSize, 0, stream>>>(
                input_img.data_ptr<scalar_t>(),
                target_img.data_ptr<scalar_t>(),
                grad_pab.data_ptr<scalar_t>(),
                grad_pa.data_ptr<scalar_t>(),
                grad_pb.data_ptr<scalar_t>(),
                grad_input_img.data_ptr<scalar_t>(),
                grad_target_img.has_value() ? grad_target_img.value().data_ptr<scalar_t>() : nullptr,
                static_cast<int>(batch_size),
                static_cast<int>(channels),
                static_cast<int>(num_bins),
                static_cast<int>(num_samples),
                kernel_type,
                minval, maxval,
                sigma_ratio
            );
        } else {
            mutual_information_histogram_bwd_kernel_basic<<<gridSize, blockSize, 0, stream>>>(
                input_img.data_ptr<scalar_t>(),
                target_img.data_ptr<scalar_t>(),
                grad_pab.data_ptr<scalar_t>(),
                grad_pa.data_ptr<scalar_t>(),
                grad_pb.data_ptr<scalar_t>(),
                grad_input_img.data_ptr<scalar_t>(),
                grad_target_img.has_value() ? grad_target_img.value().data_ptr<scalar_t>() : nullptr,
                static_cast<int64_t>(batch_size),
                static_cast<int64_t>(channels),
                static_cast<int64_t>(num_bins),
                static_cast<int64_t>(num_samples),
                kernel_type,
                minval, maxval,
                sigma_ratio
            );
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

/* ************************************************************************************************************ */
/* ************************************************************************************************************ */
/* ************************************************************************************************************ */
/* FORWARD KERNEL */
/* ************************************************************************************************************ */
/* ************************************************************************************************************ */
/* ************************************************************************************************************ */

template <typename scalar_t, typename index_t>
__global__ void mutual_information_histogram_fwd_kernel_basic_exact(
    scalar_t* __restrict__ input_img, scalar_t* __restrict__ target_img, scalar_t* __restrict__ pab, scalar_t* __restrict__ pa, scalar_t* __restrict__ pb,
    index_t batch_size, index_t channels, index_t num_aggregates, index_t num_bins, index_t num_samples,
    float minval, float maxval, float sigma_ratio, KernelType kernel_type
) {
    using opmath_t = at::opmath_type<scalar_t>;
    __shared__ opmath_t shared_prob_pa[BLOCKSIZE_3D];
    __shared__ opmath_t shared_prob_pb[BLOCKSIZE_3D];

    // get (b, c, n, bin_idx)
    // index_t id = blockIdx.x * blockDim.x + threadIdx.x;
    index_t bins_per_block = min(blockDim.x / num_bins, num_bins);
    // only the blockIdx is left 
    // index_t agg_idx = id % num_aggregates;
    // index_t ch_idx = (id / num_aggregates) % channels;
    // index_t b_idx = (id / (num_aggregates * channels)) % batch_size;
    index_t id = blockIdx.x;
    index_t sample_idx = id % num_samples;
    index_t ch_idx = (id / num_samples) % channels;
    index_t b_idx = (id / (num_samples * channels)) % batch_size;
    index_t agg_idx = sample_idx % num_aggregates;

    if (b_idx >= batch_size) {
        return;
    }

    opmath_t minval_op = static_cast<opmath_t>(minval);
    opmath_t maxval_op = static_cast<opmath_t>(maxval);

    // convert bin_bin_idx to bin_idx_j and bin_idx_i
    index_t bin_bin_idx = threadIdx.x;
    index_t bin_idx = bin_bin_idx % num_bins;
    index_t bin_idx_outer = bin_bin_idx / num_bins;

    // index
    index_t pa_idx = num_bins * (agg_idx + num_aggregates * (ch_idx + channels * b_idx));  // [b, c, agg, ?]
    
    // load i and j
    opmath_t iimg = static_cast<opmath_t>(input_img[b_idx * channels * num_samples + ch_idx * num_samples + sample_idx]);
    opmath_t jimg = static_cast<opmath_t>(target_img[b_idx * channels * num_samples + ch_idx * num_samples + sample_idx]);
    // normalize
    iimg = (iimg - minval_op) / (maxval_op - minval_op);
    jimg = (jimg - minval_op) / (maxval_op - minval_op);

    // compute unnormalized probability of beting in `bin_idx`
    shared_prob_pa[threadIdx.x] = prob_func(iimg, bin_idx, num_bins, kernel_type, sigma_ratio);
    shared_prob_pb[threadIdx.x] = prob_func(jimg, bin_idx, num_bins, kernel_type, sigma_ratio);

    scalar_t prob_Ik = shared_prob_pa[threadIdx.x];
    scalar_t prob_Jk = shared_prob_pb[threadIdx.x];

    __syncthreads();
    // compute O(log(n)) sum of unnormalized probabilities
    for (index_t i = num_bins / 2; i > 0; i >>= 1) {
        if (bin_idx < i) {
            shared_prob_pa[threadIdx.x] += shared_prob_pa[threadIdx.x + i];
            shared_prob_pb[threadIdx.x] += shared_prob_pb[threadIdx.x + i];
        }
        __syncthreads();
    }
    // normalize (bin 0 contains the sum of all w(j) for this image intensity)
    prob_Ik = prob_Ik / shared_prob_pa[threadIdx.x - bin_idx];
    prob_Jk = prob_Jk / shared_prob_pb[threadIdx.x - bin_idx];
    __syncthreads();
    shared_prob_pa[threadIdx.x] = prob_Ik;  // compute this for pab computation
    __syncthreads();

    // add these to aggregate only for first outer bin
    if(bin_idx_outer == 0) {
        // pa[bin_idx + pa_idx] += prob_Ik;
        // pb[bin_idx + pa_idx] += prob_Jk;
        gpuAtomicAdd(pa + bin_idx + pa_idx, prob_Ik);
        gpuAtomicAdd(pb + bin_idx + pa_idx, prob_Jk);
    }

    for (index_t i = bin_idx_outer; i < num_bins; i += bins_per_block) {
        // pab[bin_idx + num_bins * (i + pa_idx)] += shared_prob_pa[threadIdx.x - bin_idx + i] * prob_Jk;
        gpuAtomicAdd(pab + bin_idx + num_bins * (i + pa_idx), shared_prob_pa[threadIdx.x - bin_idx + i] * prob_Jk);
    }

}

template <typename scalar_t, typename index_t>
__global__ void mutual_information_histogram_fwd_kernel_basic_approximate(
    scalar_t* __restrict__ input_img, scalar_t* __restrict__ target_img, scalar_t* __restrict__ pab, scalar_t* __restrict__ pa, scalar_t* __restrict__ pb,
    index_t batch_size, index_t channels, index_t num_aggregates, index_t num_bins, index_t num_samples,
    float minval, float maxval, float sigma_ratio, KernelType kernel_type
) {
    using opmath_t = at::opmath_type<scalar_t>;
    __shared__ opmath_t shared_prob_pa[BLOCKSIZE_3D];
    __shared__ opmath_t shared_prob_pb[BLOCKSIZE_3D];

    // get (b, c, n, bin_idx)
    index_t id = blockIdx.x * blockDim.x + threadIdx.x;
    // index_t agg_idx = id % num_aggregates;
    // index_t ch_idx = (id / num_aggregates) % channels;
    // index_t b_idx = (id / (num_aggregates * channels)) % batch_size;
    index_t sample_idx = id % num_samples;
    index_t ch_idx = (id / num_samples) % channels;
    index_t b_idx = (id / (num_samples * channels)) % batch_size;
    index_t agg_idx = sample_idx % num_aggregates;

    // some threads have overflown
    if (b_idx >= batch_size) {
        return;
    }

    opmath_t minval_op = static_cast<opmath_t>(minval);
    opmath_t maxval_op = static_cast<opmath_t>(maxval);

    // index
    index_t pa_idx = num_bins * (agg_idx + num_aggregates * (ch_idx + channels * b_idx));  // [b, c, agg, ?]
    
    // load i and j
    opmath_t iimg = static_cast<opmath_t>(input_img[b_idx * channels * num_samples + ch_idx * num_samples + sample_idx]);
    opmath_t jimg = static_cast<opmath_t>(target_img[b_idx * channels * num_samples + ch_idx * num_samples + sample_idx]);
    // normalize
    iimg = (iimg - minval_op) / (maxval_op - minval_op);
    jimg = (jimg - minval_op) / (maxval_op - minval_op);

    // compute bin indices (approximate reduction)
    index_t i_bin_idx = static_cast<index_t>(::floor(iimg * num_bins));
    index_t j_bin_idx = static_cast<index_t>(::floor(jimg * num_bins));
    if (i_bin_idx < 0) i_bin_idx = 0;
    if (j_bin_idx < 0) j_bin_idx = 0;
    if (i_bin_idx >= num_bins) i_bin_idx = num_bins - 1;
    if (j_bin_idx >= num_bins) j_bin_idx = num_bins - 1;

    // add to histogram
    gpuAtomicAdd(pab + j_bin_idx + num_bins * (i_bin_idx + pa_idx), 1);
    gpuAtomicAdd(pa + i_bin_idx + pa_idx, 1);
    gpuAtomicAdd(pb + j_bin_idx + pa_idx, 1);
    // pab[j_bin_idx + num_bins * (i_bin_idx + pa_idx)] += 1;
    // pa[i_bin_idx + pa_idx] += 1;
    // pb[j_bin_idx + pa_idx] += 1;
}


std::vector<torch::Tensor> mutual_information_histogram_fwd(torch::Tensor &input_img, torch::Tensor &target_img, int num_bins, KernelType kernel_type, float minval, float maxval, float sigma_ratio, bool approximate_reduction) {
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

    // get batch_size, channel, and number of samples
    int64_t numel = input_img.numel();
    int64_t batch_size = input_img.size(0);
    int64_t channels = input_img.size(1);
    int64_t num_samples = numel / (batch_size * channels);

    // define number of aggregates
    int64_t num_aggregates = min(num_samples / num_bins / num_bins , static_cast<int64_t>(65536)); 
    if (num_bins > 32 && num_aggregates == 65536) {
        int64_t factor = num_bins / 32;
        num_aggregates = num_aggregates / factor / factor;
    }
    num_aggregates = ((num_aggregates + BLOCKSIZE_3D - 1) / BLOCKSIZE_3D) * BLOCKSIZE_3D;

    int bins_per_block = min(BLOCKSIZE_3D / num_bins, num_bins);

    // determine grid size and blocksize
    int64_t grid_size_blocks;
    if (approximate_reduction) {
        grid_size_blocks = div_ceil(batch_size * channels * num_samples, BLOCKSIZE_3D);
    }
    else {
        grid_size_blocks = batch_size * channels * num_samples;
    }
    dim3 blockSize(approximate_reduction ? BLOCKSIZE_3D : min(BLOCKSIZE_3D, num_bins * num_bins));
    dim3 gridSize(grid_size_blocks);
    // dim3 gridSize(batch_size * channels * num_aggregates * (approximate_reduction ? 1 : num_bins) / BLOCKSIZE_3D);

    torch::Tensor pab = torch::zeros({batch_size, channels, num_aggregates, num_bins, num_bins}, torch::TensorOptions().device(input_img.device()).dtype(torch::kFloat));
    torch::Tensor pa = torch::zeros({batch_size, channels, num_aggregates, num_bins}, torch::TensorOptions().device(input_img.device()).dtype(torch::kFloat));
    torch::Tensor pb = torch::zeros({batch_size, channels, num_aggregates, num_bins}, torch::TensorOptions().device(input_img.device()).dtype(torch::kFloat));


    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input_img.scalar_type(), "mutual_information_histogram_fwd", [&] {
        if (canUse32BitIndexMath(input_img) && canUse32BitIndexMath(pab)) {
            if (approximate_reduction) {
                mutual_information_histogram_fwd_kernel_basic_approximate<<<gridSize, blockSize, 0, stream>>>(
                    input_img.data_ptr<scalar_t>(),
                    target_img.data_ptr<scalar_t>(),
                    pab.data_ptr<scalar_t>(),
                    pa.data_ptr<scalar_t>(),
                    pb.data_ptr<scalar_t>(),
                    static_cast<int>(batch_size),
                    static_cast<int>(channels),
                    static_cast<int>(num_aggregates),
                    static_cast<int>(num_bins),
                    static_cast<int>(num_samples),
                    minval, maxval,
                    sigma_ratio,
                    kernel_type
                );
            } else {
                mutual_information_histogram_fwd_kernel_basic_exact<<<gridSize, blockSize, 0, stream>>>(
                    input_img.data_ptr<scalar_t>(),
                    target_img.data_ptr<scalar_t>(),
                    pab.data_ptr<scalar_t>(),
                    pa.data_ptr<scalar_t>(),
                    pb.data_ptr<scalar_t>(),
                    static_cast<int>(batch_size),
                    static_cast<int>(channels),
                    static_cast<int>(num_aggregates),
                    static_cast<int>(num_bins),
                    static_cast<int>(num_samples),
                    minval, maxval,
                    sigma_ratio,
                    kernel_type
                );
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        else {
            if (approximate_reduction) {
                mutual_information_histogram_fwd_kernel_basic_approximate<<<gridSize, blockSize, 0, stream>>>(
                input_img.data_ptr<scalar_t>(),
                target_img.data_ptr<scalar_t>(),
                pab.data_ptr<scalar_t>(),
                pa.data_ptr<scalar_t>(),
                pb.data_ptr<scalar_t>(),
                static_cast<int64_t>(batch_size),
                static_cast<int64_t>(channels),
                static_cast<int64_t>(num_aggregates),
                static_cast<int64_t>(num_bins),
                static_cast<int64_t>(num_samples),
                minval, maxval,
                sigma_ratio,
                kernel_type);
            } else {
            mutual_information_histogram_fwd_kernel_basic_exact<<<gridSize, blockSize, 0, stream>>>(
                input_img.data_ptr<scalar_t>(),
                target_img.data_ptr<scalar_t>(),
                pab.data_ptr<scalar_t>(),
                pa.data_ptr<scalar_t>(),
                pb.data_ptr<scalar_t>(),
                static_cast<int64_t>(batch_size),
                static_cast<int64_t>(channels),
                static_cast<int64_t>(num_aggregates),
                static_cast<int64_t>(num_bins),
                static_cast<int64_t>(num_samples),
                minval, maxval,
                sigma_ratio,
                kernel_type
            );
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    });

    float num_samples_float = static_cast<float>(num_samples);
    pa = pa.sum({2}).to(torch::kFloat) / num_samples_float;
    pb = pb.sum({2}).to(torch::kFloat) / num_samples_float;
    pab = pab.sum({2}).to(torch::kFloat) / num_samples_float;

    return {pab, pa, pb};
}
