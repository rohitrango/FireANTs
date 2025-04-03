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