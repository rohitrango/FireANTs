#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "CrossCorrelation.h"
#include "FusedGridSampler.h"

PYBIND11_MODULE(fireants_fused_ops, m) {
    // Reduction enum
    // usable as integers
    py::enum_<Reduction>(m, "Reduction", py::arithmetic())
            .value("MEAN", Reduction::MEAN)
            .value("SUM", Reduction::SUM)
            .value("NONE", Reduction::NONE).export_values();

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
       py::arg("input"), py::arg("affine_3d"), py::arg("grid"), py::arg("out_D"), py::arg("out_H"), py::arg("out_W"),
       py::arg("grid_xmin"), py::arg("grid_ymin"), py::arg("grid_zmin"), py::arg("grid_xmax"), py::arg("grid_ymax"), py::arg("grid_zmax"),
       py::arg("is_displacement"), py::arg("interpolation_mode"), py::arg("padding_mode"), py::arg("align_corners"));
    
    m.def("fused_grid_sampler_3d_backward", &fused_grid_sampler_3d_backward_impl, "Backward pass for fused grid sample",
        py::arg("input"), py::arg("affine_3d"), py::arg("grid"), py::arg("grad_output"), py::arg("grad_input"), py::arg("grad_affine"), py::arg("grad_grid"),
        py::arg("out_D"), py::arg("out_H"), py::arg("out_W"), py::arg("grid_xmin"), py::arg("grid_ymin"), py::arg("grid_zmin"), py::arg("grid_xmax"), py::arg("grid_ymax"), py::arg("grid_zmax"),
        py::arg("is_displacement"), py::arg("interpolation_mode"), py::arg("padding_mode"), py::arg("align_corners"));
}
