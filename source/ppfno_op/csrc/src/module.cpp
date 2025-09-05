/*
Copyright 2022-2025 TheCoreTeam. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
================================================================================
*/

#include <fused_segment_csr/fused_segment_csr.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace fused_segment_csr {
void BuildModule(pybind11::module_& mod) {
  // NOLINT[runtime/references]

  mod.doc() = "";
  mod.attr("Py_TPFLAGS_BASETYPE") = pybind11::int_(Py_TPFLAGS_BASETYPE);
#ifdef _GLIBCXX_USE_CXX11_ABI
  // NOLINTNEXTLINE[modernize-use-bool-literals]
  mod.attr("GLIBCXX_USE_CXX11_ABI") =
      pybind11::bool_(static_cast<bool>(_GLIBCXX_USE_CXX11_ABI));
#else
  mod.attr("GLIBCXX_USE_CXX11_ABI") = pybind11::bool_(false);
#endif

  mod.def("select_segment_csr_mean", &select_segment_csr_mean, "",
          pybind11::arg("src"), pybind11::arg("idx_map"),
          pybind11::arg("indptr"));

  mod.def("select_segment_csr_sum", &select_segment_csr_sum, "",
          pybind11::arg("src"), pybind11::arg("idx_map"),
          pybind11::arg("indptr"));

  mod.def("select_segment_csr_mean_bwd", &select_segment_csr_mean_bwd, "",
          pybind11::arg("src_shape"), pybind11::arg("grad_output"),
          pybind11::arg("idx_map"), pybind11::arg("indptr"));

  mod.def("select_segment_csr_sum_bwd", &select_segment_csr_sum_bwd, "",
          pybind11::arg("src_shape"), pybind11::arg("grad_output"),
          pybind11::arg("idx_map"), pybind11::arg("indptr"));
}
}  // namespace fused_segment_csr

#if PYBIND11_VERSION_HEX >= 0x020D00F0  // pybind11 2.13.0
// NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-type-vararg]
PYBIND11_MODULE(_C, mod, pybind11::mod_gil_not_used()) {
  fused_segment_csr::BuildModule(mod);
}
#else
// NOLINTNEXTLINE[cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-type-vararg]
PYBIND11_MODULE(_C, mod) { fused_segement_csr::BuildModule(mod); }
#endif
