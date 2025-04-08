// define constants and other common utilities
#include <torch/torch.h>

#define BLOCKSIZE_3D 512
#define WARP_SIZE 32
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/* All other utilities go here */

void adam_update_fused(torch::Tensor &grad,
                      torch::Tensor exp_avg,
                      torch::Tensor exp_avg_sq,
                      float beta1,
                      float beta2,
                      float eps);


static inline int64_t GET_BLOCKS_v2(const int64_t N, const int64_t max_threads_per_block=BLOCKSIZE_3D) {
    TORCH_INTERNAL_ASSERT(N > 0, "CUDA kernel launch blocks must be positive, but got N=", N);
    constexpr int64_t max_int = std::numeric_limits<int>::max();
  
    // Round up division for positive number that cannot cause integer overflow
    auto block_num = (N - 1) / max_threads_per_block + 1;
    TORCH_INTERNAL_ASSERT(block_num <= max_int, "Can't schedule too many blocks on CUDA device");
    return block_num;
}
