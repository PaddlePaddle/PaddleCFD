#pragma once
#include <paddle/phi/api/include/tensor.h>

namespace fused_segment_csr::impl {
void select_segment_csr_mean_1d(paddle::Tensor& out, const paddle::Tensor& src,
                                const paddle::Tensor& map,
                                const paddle::Tensor& indptr);

void select_segment_csr_sum_1d(paddle::Tensor& out, const paddle::Tensor& src,
                               const paddle::Tensor& map,
                               const paddle::Tensor& indptr);

void select_segment_csr_mean_bwd_1d(paddle::Tensor& grad_input,
                                    const paddle::Tensor& grad_output,
                                    const paddle::Tensor& map,
                                    const paddle::Tensor& indptr);

void select_segment_csr_sum_bwd_1d(paddle::Tensor& grad_input,
                                   const paddle::Tensor& grad_output,
                                   const paddle::Tensor& map,
                                   const paddle::Tensor& indptr);

void select_segment_csr_mean_2d(paddle::Tensor& out, const paddle::Tensor& src,
                                const paddle::Tensor& map,
                                const paddle::Tensor& indptr);

void select_segment_csr_sum_2d(paddle::Tensor& out, const paddle::Tensor& src,
                               const paddle::Tensor& map,
                               const paddle::Tensor& indptr);

void select_segment_csr_mean_bwd_2d(paddle::Tensor& grad_input,
                                    const paddle::Tensor& grad_output,
                                    const paddle::Tensor& map,
                                    const paddle::Tensor& indptr);

void select_segment_csr_sum_bwd_2d(paddle::Tensor& grad_input,
                                   const paddle::Tensor& grad_output,
                                   const paddle::Tensor& map,
                                   const paddle::Tensor& indptr);
}  // namespace fused_segment_csr::impl
