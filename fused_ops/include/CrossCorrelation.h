#include <torch/torch.h>
#include <string>

enum class Reduction {
    MEAN,
    SUM, 
    NONE
};

torch::Tensor cc3d_fwd_interm_v1(torch::Tensor intermediates, int kernel_volume, Reduction reduction, float nr, float dr);
