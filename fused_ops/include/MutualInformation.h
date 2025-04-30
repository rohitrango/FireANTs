#include <iostream>
#include <torch/torch.h>
#include <string>

enum class KernelType {
    GAUSSIAN,
    BSPLINE, 
    DELTA,
};

std::vector<torch::Tensor> mutual_information_histogram_fwd(torch::Tensor input_img, torch::Tensor target_img, int num_bins, KernelType kernel_type);

