#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "CrossCorrelation.h"

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


}
