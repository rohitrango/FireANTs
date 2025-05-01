#include <iostream>
#include <torch/torch.h>
#include <string>

enum class KernelType {
    GAUSSIAN,
    BSPLINE, 
    DELTA,
};

std::vector<torch::Tensor> mutual_information_histogram_fwd(torch::Tensor &input_img, torch::Tensor &target_img, int num_bins, KernelType kernel_type);

void mutual_information_histogram_bwd(torch::Tensor &input_img, torch::Tensor &target_img, torch::Tensor &grad_pab, torch::Tensor &grad_pa, torch::Tensor &grad_pb, int num_bins, torch::Tensor &grad_input_img, std::optional<torch::Tensor> &grad_target_img, KernelType kernel_type);