#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "CrossCorrelation.h"
#include "FusedGridSampler.h"
#include "FusedGridComposer.h"
#include "FusedGenerateGrid.h"
#include "GaussianBlurFFT.h"
#include "MutualInformation.h"
#include "common.h"

PYBIND11_MODULE(fireants_fused_ops, m) {
    // Reduction enum
    // usable as integers
    py::enum_<Reduction>(m, "Reduction", py::arithmetic())
            .value("MEAN", Reduction::MEAN)
            .value("SUM", Reduction::SUM)
            .value("NONE", Reduction::NONE).export_values();

    py::enum_<KernelType>(m, "KernelType", py::arithmetic())
            .value("GAUSSIAN", KernelType::GAUSSIAN)
            .value("BSPLINE", KernelType::BSPLINE)
            .value("DELTA", KernelType::DELTA).export_values();

    // cross-correlation forward
    m.def("cc3d_fwd_interm_v1", &cc3d_fwd_interm_v1, "Forward pass of cross-correlation given intermediates", 
    py::arg("intermediates"), py::arg("kernel_volume"), py::arg("reduction") = Reduction::MEAN, py::arg("nr") = 0.0, py::arg("dr") = 1e-5);

    m.def("cc3d_bwd_modify_interm_v1", &cc3d_bwd_modify_interm_v1, "Backward pass of cross-correlation given intermediates", 
    py::arg("intermediates"), py::arg("input_img"), py::arg("target_img"), py::arg("grad_output"), py::arg("grad_input_img"), py::arg("grad_target_img"), py::arg("kernel_size"), py::arg("nr") = 0.0, py::arg("dr") = 1e-5, py::arg("reduction") = Reduction::MEAN);

    m.def("create_intermediates", &create_intermediates, "Create intermediates for cross-correlation",
        py::arg("input_img"), py::arg("target_img"), py::arg("intermediates"));

    m.def("cc3d_bwd_compute_grads", &cc3d_bwd_compute_grads, "Compute gradients for cross-correlation",
        py::arg("intermediates"), py::arg("input_img"), py::arg("target_img"), py::arg("grad_input_img"), py::arg("grad_target_img"));
    
    // grid sampler utils
    m.def("fused_grid_sampler_3d_forward", &fused_grid_sampler_3d_forward_impl, "Forward pass for fused grid sample",
       py::arg("input"), py::arg("affine_3d"), py::arg("grid"), py::arg("affine_3d_pregrid"), py::arg("output"), py::arg("out_D"), py::arg("out_H"), py::arg("out_W"),
       py::arg("grid_xmin"), py::arg("grid_ymin"), py::arg("grid_zmin"), py::arg("grid_xmax"), py::arg("grid_ymax"), py::arg("grid_zmax"),
       py::arg("is_displacement"), py::arg("interpolation_mode"), py::arg("padding_mode"), py::arg("align_corners"));
    
    m.def("fused_grid_sampler_3d_backward", &fused_grid_sampler_3d_backward_impl, "Backward pass for fused grid sample",
        py::arg("input"), py::arg("affine_3d"), py::arg("grid"), py::arg("affine_3d_pregrid"), py::arg("grad_output"), py::arg("grad_input"), py::arg("grad_affine"), py::arg("grad_grid"),
        py::arg("out_D"), py::arg("out_H"), py::arg("out_W"), py::arg("grid_xmin"), py::arg("grid_ymin"), py::arg("grid_zmin"), py::arg("grid_xmax"), py::arg("grid_ymax"), py::arg("grid_zmax"),
        py::arg("is_displacement"), py::arg("interpolation_mode"), py::arg("padding_mode"), py::arg("align_corners"));

    // grid composer utils
    m.def("fused_grid_composer_3d_forward", &fused_grid_composer_3d_forward_impl, "Forward pass for fused grid composer",
        py::arg("input"), py::arg("affine_3d"), py::arg("grid"), py::arg("grid_xmin"), py::arg("grid_ymin"), py::arg("grid_zmin"), py::arg("grid_xmax"), py::arg("grid_ymax"), py::arg("grid_zmax"),
        py::arg("align_corners"), py::arg("output"));

    m.def("fused_grid_composer_3d_backward", &fused_grid_composer_3d_backward_impl, "Backward pass for fused grid composer",
        py::arg("input"), py::arg("affine_3d"), py::arg("grid"), py::arg("grad_output"), py::arg("grad_input"), py::arg("grad_affine"), py::arg("grad_grid"),
        py::arg("grid_xmin"), py::arg("grid_ymin"), py::arg("grid_zmin"), py::arg("grid_xmax"), py::arg("grid_ymax"), py::arg("grid_zmax"), py::arg("align_corners"));
    
    m.def("fused_warp_create_3d_forward", &fused_warp_create_3d_forward_impl, "Forward pass for fused warp create",
        py::arg("affine"), py::arg("grid"), py::arg("grid_xmin"), py::arg("grid_ymin"), py::arg("grid_zmin"), py::arg("grid_xmax"), py::arg("grid_ymax"), py::arg("grid_zmax"));

    m.def("fused_warp_create_3d_backward", &fused_warp_create_3d_backward_impl, "Backward pass for fused warp create",
        py::arg("affine"), py::arg("grid"), py::arg("grad_output"), py::arg("grad_affine"), py::arg("grad_grid"),
        py::arg("grid_xmin"), py::arg("grid_ymin"), py::arg("grid_zmin"), py::arg("grid_xmax"), py::arg("grid_ymax"), py::arg("grid_zmax"));
    
    m.def("adam_update_fused", &adam_update_fused, "Adam update fused",
        py::arg("grad"), py::arg("exp_avg"), py::arg("exp_avg_sq"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"));
    
    // gaussian blur in fft space
    m.def("gaussian_blur_fft2", &gaussian_blur_fft2, "Gaussian blur in fft space",
        py::arg("im_fft"), py::arg("ys"), py::arg("xs"), py::arg("ye"), py::arg("xe"), py::arg("multiplier"));
    
    m.def("gaussian_blur_fft3", &gaussian_blur_fft3, "Gaussian blur in fft space",
        py::arg("im_fft"), py::arg("zs"), py::arg("ys"), py::arg("xs"), py::arg("ze"), py::arg("ye"), py::arg("xe"), py::arg("multiplier"));
    
    m.def("mutual_information_histogram_fwd", &mutual_information_histogram_fwd, "Mutual information histogram forward", py::arg("input_img"), py::arg("target_img"), py::arg("num_bins"), py::arg("kernel_type") = KernelType::GAUSSIAN, py::arg("minval") = 0.0, py::arg("maxval") = 1.0, py::arg("sigma_ratio") = 1.0, py::arg("approximate_reduction") = false);

    m.def("mutual_information_histogram_bwd", &mutual_information_histogram_bwd, "Mutual information histogram backward", py::arg("input_img"), py::arg("target_img"), py::arg("grad_pab"), py::arg("grad_pa"), py::arg("grad_pb"), py::arg("num_bins"), py::arg("grad_input_img"), py::arg("grad_target_img"), py::arg("kernel_type") = KernelType::GAUSSIAN, py::arg("minval") = 0.0, py::arg("maxval") = 1.0, py::arg("sigma_ratio") = 1.0);

}
