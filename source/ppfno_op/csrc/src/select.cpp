#include "fused_segment_csr/select.h"

#include <fused_segment_csr/fused_segment_csr.h>
#include <paddle/phi/api/ext/tensor_compat.h>
#include <paddle/phi/api/include/tensor.h>
#include <paddle/utils/pybind.h>

namespace fused_segment_csr {
pybind11::object select_segment_csr_sum(pybind11::object src_,
                                        pybind11::object map_,
                                        pybind11::object indptr_) {
  const auto src = pybind11::cast<const paddle::Tensor>(src_);
  const auto map = pybind11::cast<const paddle::Tensor>(map_);
  const auto indptr = pybind11::cast<const paddle::Tensor>(indptr_);

  PD_CHECK(map.shape().size() == 1);
  PD_CHECK(indptr.shape().size() == 1);

  PD_CHECK(src.is_gpu());
  PD_CHECK(map.is_gpu());
  PD_CHECK(indptr.is_gpu());

  PD_CHECK(src.is_contiguous());
  PD_CHECK(map.is_contiguous());
  PD_CHECK(indptr.is_contiguous());

  paddle::Tensor out;
  if (src.shape().size() == 1) {
    out = paddle::empty({indptr.shape()[0] - 1}, src.dtype(), src.place());

    impl::select_segment_csr_sum_1d(out, src, map, indptr);
  } else if (src.shape().size() == 2) {
    out = paddle::empty({indptr.shape()[0] - 1, src.shape()[1]}, src.dtype(),
                        src.place());

    impl::select_segment_csr_sum_2d(out, src, map, indptr);
  } else {
    throw std::invalid_argument(
        "select_segment_csr_sum only supports 1D or 2D tensors, but got " +
        std::to_string(src.shape().size()) + "D tensor.");
  }

  return pybind11::cast(out);
}

pybind11::object select_segment_csr_mean(pybind11::object src_,
                                         pybind11::object map_,
                                         pybind11::object indptr_) {
  const auto src = pybind11::cast<const paddle::Tensor>(src_);
  const auto map = pybind11::cast<const paddle::Tensor>(map_);
  const auto indptr = pybind11::cast<const paddle::Tensor>(indptr_);

  PD_CHECK(map.shape().size() == 1);
  PD_CHECK(indptr.shape().size() == 1);

  PD_CHECK(src.is_gpu());
  PD_CHECK(map.is_gpu());
  PD_CHECK(indptr.is_gpu());

  PD_CHECK(src.is_contiguous());
  PD_CHECK(map.is_contiguous());
  PD_CHECK(indptr.is_contiguous());

  paddle::Tensor out;

  if (src.shape().size() == 1) {
    out = paddle::empty({indptr.shape()[0] - 1}, src.dtype(), src.place());

    impl::select_segment_csr_mean_1d(out, src, map, indptr);
  } else if (src.shape().size() == 2) {
    out = paddle::empty({indptr.shape()[0] - 1, src.shape()[1]}, src.dtype(),
                        src.place());

    impl::select_segment_csr_mean_2d(out, src, map, indptr);
  } else {
    throw std::invalid_argument(
        "select_segment_csr_mean only supports 1D or 2D tensors, but got " +
        std::to_string(src.shape().size()) + "D tensor.");
  }

  return pybind11::cast(out);
}

pybind11::object select_segment_csr_mean_bwd(
    const std::vector<int64_t>& src_shape, pybind11::object grad_output_,
    pybind11::object map_, pybind11::object indptr_) {
  const auto grad_output = pybind11::cast<const paddle::Tensor>(grad_output_);
  const auto map = pybind11::cast<const paddle::Tensor>(map_);
  const auto indptr = pybind11::cast<const paddle::Tensor>(indptr_);

  PD_CHECK(map.shape().size() == 1);
  PD_CHECK(indptr.shape().size() == 1);

  PD_CHECK(map.is_gpu());
  PD_CHECK(indptr.is_gpu());

  // For grad_output
  PD_CHECK(grad_output.shape()[0] == indptr.shape()[0] - 1);
  PD_CHECK(grad_output.is_gpu());

  PD_CHECK(map.is_contiguous());
  PD_CHECK(indptr.is_contiguous());
  PD_CHECK(grad_output.is_contiguous());

  paddle::Tensor grad_input =
      paddle::zeros(src_shape, grad_output.dtype(), grad_output.place());

  if (src_shape.size() == 1) {
    PD_CHECK(grad_output.shape().size() == 1);

    impl::select_segment_csr_mean_bwd_1d(grad_input, grad_output, map, indptr);
  } else if (src_shape.size() == 2) {
    PD_CHECK(grad_output.shape().size() == 2);
    PD_CHECK(grad_output.shape()[1] == src_shape[1]);

    impl::select_segment_csr_mean_bwd_2d(grad_input, grad_output, map, indptr);
  } else {
    throw std::invalid_argument(
        "select_segment_csr_mean_bwd only supports 1D or 2D tensors, but got " +
        std::to_string(src_shape.size()) + "D tensor.");
  }

  return pybind11::cast(grad_input);
}

pybind11::object select_segment_csr_sum_bwd(
    const std::vector<int64_t>& src_shape, pybind11::object grad_output_,
    pybind11::object map_, pybind11::object indptr_) {
  const auto grad_output = pybind11::cast<const paddle::Tensor>(grad_output_);
  const auto map = pybind11::cast<const paddle::Tensor>(map_);
  const auto indptr = pybind11::cast<const paddle::Tensor>(indptr_);

  PD_CHECK(map.shape().size() == 1);
  PD_CHECK(indptr.shape().size() == 1);

  PD_CHECK(map.is_gpu());
  PD_CHECK(indptr.is_gpu());

  // For grad_output
  PD_CHECK(grad_output.shape()[0] == indptr.shape()[0] - 1);
  PD_CHECK(grad_output.is_gpu());

  PD_CHECK(map.is_contiguous());
  PD_CHECK(indptr.is_contiguous());
  PD_CHECK(grad_output.is_contiguous());

  paddle::Tensor grad_input =
      paddle::zeros(src_shape, grad_output.dtype(), grad_output.place());

  if (src_shape.size() == 1) {
    PD_CHECK(grad_output.shape().size() == 1);

    impl::select_segment_csr_sum_bwd_1d(grad_input, grad_output, map, indptr);
  } else if (src_shape.size() == 2) {
    PD_CHECK(grad_output.shape().size() == 2);
    PD_CHECK(grad_output.shape()[1] == src_shape[1]);

    impl::select_segment_csr_sum_bwd_2d(grad_input, grad_output, map, indptr);
  } else {
    throw std::invalid_argument(
        "select_segment_csr_sum_bwd only supports 1D or 2D tensors, but got " +
        std::to_string(src_shape.size()) + "D tensor.");
  }

  return pybind11::cast(grad_input);
}
}  // namespace fused_segment_csr
