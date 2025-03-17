#include <torch/torch.h>
#include <string>

enum class Reduction {
    MEAN,
    SUM, 
    NONE
};

torch::Tensor create_intermediates(torch::Tensor input_img, torch::Tensor target_img, torch::Tensor intermediates);

torch::Tensor cc3d_fwd_interm_v1(torch::Tensor intermediates, int kernel_volume, Reduction reduction, float nr, float dr);

torch::Tensor cc3d_bwd_modify_interm_v1(torch::Tensor intermediates, torch::Tensor input_img, torch::Tensor target_img, 
    torch::Tensor grad_output, 
    std::optional<torch::Tensor> grad_input_img, std::optional<torch::Tensor> grad_target_img, 
    int kernel_size, float nr, float dr, Reduction reduction);

void cc3d_bwd_compute_grads(torch::Tensor intermediates, torch::Tensor input_img, torch::Tensor target_img, torch::Tensor grad_input_img, std::optional<torch::Tensor> grad_target_img);