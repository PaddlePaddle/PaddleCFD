#pragma once
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace fused_segment_csr {
pybind11::object select_segment_csr_mean(pybind11::object src,
                                         pybind11::object map,
                                         pybind11::object indptr);

pybind11::object select_segment_csr_sum(pybind11::object src,
                                        pybind11::object map,
                                        pybind11::object indptr);

pybind11::object select_segment_csr_mean_bwd(
    const std::vector<int64_t>& src_shape, pybind11::object grad_output,
    pybind11::object map, pybind11::object indptr);

pybind11::object select_segment_csr_sum_bwd(
    const std::vector<int64_t>& src_shape, pybind11::object grad_output,
    pybind11::object map, pybind11::object indptr);
}  // namespace fused_segment_csr
