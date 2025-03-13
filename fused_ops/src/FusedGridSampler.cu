// #include <cuda_runtime.h>
// #include <ATen/ATen.h>
// #include <torch/extension.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/detail/KernelUtils.h>
// #include <torch/torch.h>
// #include <c10/cuda/CUDAGuard.h>
// // #include <c10/cuda/CUDAGuard.h>

// // #include "common.h"
// #define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// #ifndef BLOCKADAM_COMMON_H
// #define BLOCKADAM_COMMON_H

// // global constants for 3d computation
// #define EFF_THREADSIZE_3D 4
// #define BLOCKSIZE_3D 512

// // global constants for 2d computation

// #endif

// template <typename scalar_t>
// __global__ void accum_grad_3d_kernel(scalar_t* hess, scalar_t* grad, float alpha, scalar_t* out, int numVox, bool normalize) {
//     /* kernel for accumulating the gradients into tril part of the hessian

//     * hess: input hessian matrix [..., 6]
//     * grad: input gradient matrix [..., 3]
//     * Alpha: momentum parameter
//     * out: output hessian matrix [..., 6]  (can be a placeholder for hess or a new output)
//     * numVox: number of voxels

//     */
    
//     // we will do all ops in float format and cast it back
//     __shared__ float hessdata[2*BLOCKSIZE_3D];
//     __shared__ float graddata[BLOCKSIZE_3D];
    
//     float oneminusalpha = 1.0 - alpha;

//     // index is computed by dividing the threadsize and threadindex by the effective threadsize  
//     int idx = blockIdx.x * (BLOCKSIZE_3D/EFF_THREADSIZE_3D) + threadIdx.x/EFF_THREADSIZE_3D;
//     if (idx >= numVox) {
//         return;
//     }

//     // load data into shared memory
//     int offset = threadIdx.x % EFF_THREADSIZE_3D;
//     int start = (threadIdx.x / EFF_THREADSIZE_3D) * EFF_THREADSIZE_3D;   // start of the threads

//     if (offset < 3) {
//         graddata[threadIdx.x] = static_cast<float>(grad[idx*3 + offset]);  // 
//         hessdata[2*threadIdx.x]= alpha * static_cast<float>(hess[idx*6 + 2*offset]);
//         hessdata[2*threadIdx.x + 1] = alpha * static_cast<float>(hess[idx*6 + 2*offset + 1]);
//     }
//     __syncthreads();

//     // collect max of the gradients
//     float maxgrad;
//     if (normalize) {
//         maxgrad = 0;
//         #pragma unroll
//         for(int i = 0; i < 3; i++) {
//             maxgrad = fmaxf(maxgrad, graddata[start + i]);
//         }
//     }
//     else {
//         maxgrad = 1;
//     }
//     // divide here to avoid more flops
//     maxgrad = 1/maxgrad;

//     if(offset == 0) {
//         hessdata[2*threadIdx.x] += oneminusalpha * graddata[threadIdx.x] * maxgrad * graddata[threadIdx.x];   // sigma_xx
//         hessdata[2*threadIdx.x + 1] += oneminusalpha * graddata[threadIdx.x] * maxgrad * graddata[threadIdx.x + 1];   // sigma_xy
//     }
//     else if (offset == 1) {
//         hessdata[2*threadIdx.x] += oneminusalpha * graddata[threadIdx.x] * maxgrad * graddata[threadIdx.x];  // sigma_yy
//         hessdata[2*threadIdx.x + 1] += oneminusalpha * graddata[threadIdx.x - 1] * maxgrad * graddata[threadIdx.x + 1];   // sigma_xz
//     }
//     else if (offset == 2) {
//         hessdata[2*threadIdx.x] += oneminusalpha * graddata[threadIdx.x - 1] * maxgrad * graddata[threadIdx.x];  // sigma_yz
//         hessdata[2*threadIdx.x + 1] += oneminusalpha * graddata[threadIdx.x] * maxgrad * graddata[threadIdx.x];  // sigma_zz
//     }

//     __syncthreads(); 

//     // write to memory
//     if (offset < 3) {
//         out[idx*6 + 2*offset] = hessdata[2*threadIdx.x];
//         out[idx*6 + 2*offset + 1] = hessdata[2*threadIdx.x + 1];
//     }

// }

// template <typename scalar_t>
// __global__ void accum_grad_3d_kernel_v2(scalar_t* hess, scalar_t* grad, float alpha, scalar_t* out, int numVox, bool normalize) {
//     /* kernel for accumulating the gradients into tril part of the hessian

//     * hess: input hessian matrix [..., 6]
//     * grad: input gradient matrix [..., 3]
//     * Alpha: momentum parameter
//     * out: output hessian matrix [..., 6]  (can be a placeholder for hess or a new output)
//     * numVox: number of voxels

//     */
    
//     // we will do all ops in float format and cast it back
//     float hessdata[6];
//     float graddata[3];
    
//     float oneminusalpha = 1.0 - alpha;

//     // index is computed by dividing the threadsize and threadindex by the effective threadsize  
//     int idx = blockIdx.x * BLOCKSIZE_3D + threadIdx.x;
//     if (idx >= numVox) {
//         return;
//     }

//     // load data into shared memory
//     #pragma unroll
//     for(int i = 0; i < 3; i++) {
//         graddata[i] = static_cast<float>(grad[idx*3 + i]);
//     }
//     #pragma unroll
//     for(int i = 0; i < 6; i++) {
//         hessdata[i] = alpha * static_cast<float>(hess[idx*6 + i]);
//     }

//     // collect max of the gradients
//     float maxgrad;
//     if (normalize) {
//         maxgrad = 1e-6;
//         #pragma unroll
//         for(int i = 0; i < 3; i++) {
//             maxgrad = fmaxf(maxgrad, fabs(graddata[i]));
//         }
//     }
//     else {
//         maxgrad = 1;
//     }
//     // invert it to multiply
//     maxgrad = 1/maxgrad;

//     hessdata[0] += oneminusalpha * graddata[0] * maxgrad * graddata[0];   // sigma_xx
//     hessdata[1] += oneminusalpha * graddata[0] * maxgrad * graddata[1];   // sigma_xy
//     hessdata[2] += oneminusalpha * graddata[1] * maxgrad * graddata[1];  // sigma_yy
//     hessdata[3] += oneminusalpha * graddata[0] * maxgrad * graddata[2];   // sigma_xz
//     hessdata[4] += oneminusalpha * graddata[1] * maxgrad * graddata[2];  // sigma_yz
//     hessdata[5] += oneminusalpha * graddata[2] * maxgrad * graddata[2];  // sigma_zz

//     // write to memory
//     #pragma unroll
//     for(int i = 0; i < 6; i++) {
//         out[idx*6 + i] = static_cast<scalar_t>(hessdata[i]);
//     }
// }

// __device__ void compute_cholesky_3d(float* A, float eps) {
//     /* 
//     function to compute the cholesky decomposition of a 3x3 matrix where only tril part is stored
//     */
//     A[0] = sqrtf(A[0] + eps);

//     A[1] = A[1] / A[0];
//     A[3] = A[3] / A[0];
//     A[2] = sqrtf(A[2] - A[1]*A[1] + eps);

//     A[4] = (A[4] - A[1]*A[3]) / A[2];
//     A[5] = sqrtf(A[5] - A[3]*A[3] - A[4]*A[4] + eps);
// }

// __device__ void solve_cholesky_3d(float* A, float* b, float* x, int step) {
//     /* 
//     function to solve the equation Ax = b for a 3x3 matrix A whose tril part is stored

//     step = 0: solve for Lx = b
//     step > 0: solve for L^T x = b
//     */
//     if (step == 0) {
//         x[0] = b[0] / A[0];
//         x[1] = (b[1] - A[1]*x[0]) / A[2];
//         x[2] = (b[2] - A[3]*x[0] - A[4]*x[1]) / A[5];
//     }
//     else {
//         x[2] = b[2] / A[5];
//         x[1] = (b[1] - A[4]*x[2]) / A[2];
//         x[0] = (b[0] - A[1]*x[1] - A[3]*x[2]) / A[0];
//     }
// } 

// template <typename scalar_t>
// __global__ void solve_pinv_3d_kernel(scalar_t* A, scalar_t* b, float eps, scalar_t* out, int numVox, bool normalize) {
//     /* kernel for accumulating the gradients into tril part of the hessian

//     * hess: input hessian matrix [..., 6]
//     * grad: input gradient matrix [..., 3]
//     * Alpha: momentum parameter
//     * out: output hessian matrix [..., 6]  (can be a placeholder for hess or a new output)
//     * numVox: number of voxels

//     */
    
//     // we will do all ops in float format and cast it back
//     float A_[6];
//     float b_[3];
//     float x_[3];
    
//     // index is computed by dividing the threadsize and threadindex by the effective threadsize  
//     int idx = blockIdx.x * BLOCKSIZE_3D + threadIdx.x;
//     if (idx >= numVox) {
//         return;
//     }

//     // load data into shared memory
//     #pragma unroll
//     for(int i = 0; i < 3; i++) {
//         b_[i] = static_cast<float>(b[idx*3 + i]);
//     }
//     #pragma unroll
//     for(int i = 0; i < 6; i++) {
//         A_[i] = static_cast<float>(A[idx*6 + i]);
//     }

//     // compute L
//     compute_cholesky_3d(A_, eps);

//     // normalize matrix to have trace 1 (eigenvalue control)
//     if(normalize) {
//         float trace = (A_[0] + A_[2] + A_[5])/3;
//         #pragma unroll
//         for(int i = 0; i < 6; i++) {
//             A_[i] /= trace;
//         }
//     }
//     solve_cholesky_3d(A_, b_, x_, 0);
//     solve_cholesky_3d(A_, x_, b_, 1);

//     // write to memory
//     #pragma unroll
//     for(int i = 0; i < 3; i++) {
//         out[idx*3 + i] = static_cast<scalar_t>(b_[i]);
//     }
// }

// // C++ function to accumulate gradient into Hessian
// torch::Tensor accum_grad_3d(torch::Tensor hess, torch::Tensor grad, float alpha, bool inplace, bool normalize) {

//     TORCH_CHECK(hess.device().type() == torch::kCUDA, "hessian must be on CUDA");
//     TORCH_CHECK(grad.device().type() == torch::kCUDA, "grad must be on CUDA");
//     TORCH_CHECK(hess.device() == grad.device(), "hessian and grad must be on the same device");

//     const at::cuda::OptionalCUDAGuard device_guard_hess(device_of(hess));
//     const at::cuda::OptionalCUDAGuard device_guard_grad(device_of(grad));

//     auto out = hess;
//     if (inplace) {
//        ;
//     } else {
//         out = torch::zeros_like(hess);
//     }

//     int n = hess.numel() / 6;      // number of voxels
//     const int blockSize = BLOCKSIZE_3D;
//     const int effectiveBlockSize = blockSize / EFF_THREADSIZE_3D;     // number of elements per block
//     const int gridSize = (n + effectiveBlockSize - 1) / effectiveBlockSize;

//     // support for half and bfloat16
//     AT_DISPATCH_FLOATING_TYPES_AND2(
//     at::ScalarType::Half, at::ScalarType::BFloat16, 
//     hess.type(), "accum_grad_3d_kernel", ([&] {
//         accum_grad_3d_kernel<<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(hess.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), alpha, out.data_ptr<scalar_t>(), n, normalize);
//     }));

//     return out;
// }

// // C++ function to accumulate gradient into Hessian
// torch::Tensor accum_grad_3d_v2(torch::Tensor hess, torch::Tensor grad, float alpha, bool inplace, bool normalize) {
//     TORCH_CHECK(hess.device().type() == torch::kCUDA, "hessian must be on CUDA");
//     TORCH_CHECK(grad.device().type() == torch::kCUDA, "grad must be on CUDA");
//     TORCH_CHECK(hess.device() == grad.device(), "hessian and grad must be on the same device");

//     const at::cuda::OptionalCUDAGuard device_guard_hess(device_of(hess));
//     const at::cuda::OptionalCUDAGuard device_guard_grad(device_of(grad));

//     auto out = hess;
//     if (inplace) {
//        ;
//     } else {
//         out = torch::zeros_like(hess);
//     }

//     int n = hess.numel() / 6;      // number of voxels
//     const int blockSize = BLOCKSIZE_3D;
//     const int gridSize = (n + blockSize - 1) / blockSize;

//     // support for half and bfloat16
//     AT_DISPATCH_FLOATING_TYPES_AND2(
//     at::ScalarType::Half, at::ScalarType::BFloat16, 
//     hess.type(), "accum_grad_3d_kernel_v2", ([&] {
//         accum_grad_3d_kernel_v2<<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(hess.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), alpha, out.data_ptr<scalar_t>(), n, normalize);
//     }));

//     return out;
// }

// // C++ function to solve the equation Ax = b for PSD matrix A whose tril is stored
// torch::Tensor solve_pinv_3d(torch::Tensor A, torch::Tensor b, float eps, bool inplace, bool normalize) {

//     TORCH_CHECK(A.device().type() == torch::kCUDA, "hessian must be on CUDA");
//     TORCH_CHECK(b.device().type() == torch::kCUDA, "grad must be on CUDA");
//     TORCH_CHECK(A.device() == b.device(), "hessian and grad must be on the same device");

//     const at::cuda::OptionalCUDAGuard device_guard_hess(device_of(A));
//     const at::cuda::OptionalCUDAGuard device_guard_grad(device_of(b));

//     auto out = b;
//     if (inplace) {
//        ;
//     } else {
//         out = torch::zeros_like(b);
//     }

//     int n = A.numel() / 6;      // number of voxels
//     const int blockSize = BLOCKSIZE_3D;
//     const int gridSize = (n + blockSize - 1) / blockSize;

//     // support for half and bfloat16
//     AT_DISPATCH_FLOATING_TYPES_AND2(
//     at::ScalarType::Half, at::ScalarType::BFloat16, 
//     A.type(), "solve_pinv_3d_kernel", ([&] {
//         solve_pinv_3d_kernel<<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(A.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), eps, out.data_ptr<scalar_t>(), n, normalize);
//     }));

//     return out;

// }