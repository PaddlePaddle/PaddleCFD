#include <cuda_runtime.h>
#include <paddle/phi/api/ext/dispatch.h>
#include <paddle/phi/api/include/tensor.h>

static constexpr int64_t BLOCK_SIZE = 128;

template <typename T1, typename T2>
auto ceil_div(const T1 a, const T2 b) {
  return (a + b - 1) / b;
}

namespace fused_segment_csr::impl {
namespace {
template <typename T>
__global__ __launch_bounds__(BLOCK_SIZE) void select_segment_csr_sum_1d_kernel(
    T* __restrict__ out, const size_t num_rows, const T* __restrict__ src,
    const int64_t* __restrict__ map, const int64_t* __restrict__ indptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto row = tid;
  if (row >= num_rows) {
    return;
  }
  const auto row_start = indptr[row];
  const auto row_end = indptr[row + 1];
  double sum = 0;
  for (auto i = row_start; i < row_end; ++i) {
    const auto selected_row = map[i];
    sum += src[selected_row];
  }
  out[row] = sum;
}

template <typename T>
__global__ __launch_bounds__(BLOCK_SIZE) void select_segment_csr_mean_1d_kernel(
    T* __restrict__ out, const size_t num_rows, const T* __restrict__ src,
    const int64_t* __restrict__ map, const int64_t* __restrict__ indptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto row = tid;
  if (row >= num_rows) {
    return;
  }
  const auto row_start = indptr[row];
  const auto row_end = indptr[row + 1];
  double sum = 0;
  for (auto i = row_start; i < row_end; ++i) {
    const auto selected_row = map[i];
    sum += src[selected_row];
  }
  const auto row_length = row_end - row_start;
  if (row_length == 0) {
    out[row] = 0;
  } else {
    out[row] = sum / row_length;
  }
}
}  // namespace

void select_segment_csr_sum_1d(paddle::Tensor& out, const paddle::Tensor& src,
                               const paddle::Tensor& map,
                               const paddle::Tensor& indptr) {
  const auto num_rows = out.shape()[0];

  const auto total_count = num_rows;
  const dim3 block(min(BLOCK_SIZE, total_count));
  const dim3 grid(ceil_div(total_count, block.x));

  PD_DISPATCH_FLOATING_TYPES(out.type(), "select_segment_csr_sum", [&] {
    select_segment_csr_sum_1d_kernel<<<grid, block, 0, src.stream()>>>(
        out.data<data_t>(), num_rows, src.data<data_t>(), map.data<int64_t>(),
        indptr.data<int64_t>());
  });
}

void select_segment_csr_mean_1d(paddle::Tensor& out, const paddle::Tensor& src,
                                const paddle::Tensor& map,
                                const paddle::Tensor& indptr) {
  const auto num_rows = out.shape()[0];

  const auto total_count = num_rows;
  const dim3 block(min(BLOCK_SIZE, total_count));
  const dim3 grid(ceil_div(total_count, block.x));

  PD_DISPATCH_FLOATING_TYPES(out.type(), "select_segment_csr_mean", [&] {
    select_segment_csr_mean_1d_kernel<<<grid, block, 0, src.stream()>>>(
        out.data<data_t>(), num_rows, src.data<data_t>(), map.data<int64_t>(),
        indptr.data<int64_t>());
  });
}

namespace {
template <typename T>
__global__
__launch_bounds__(BLOCK_SIZE) void select_segment_csr_mean_bwd_1d_kernel(
    T* __restrict__ grad_input, const size_t num_rows,
    const T* __restrict__ grad_output, const int64_t* __restrict__ map,
    const int64_t* __restrict__ indptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto row = tid;
  if (row >= num_rows) {
    return;
  }
  const auto row_start = indptr[row];
  const auto row_end = indptr[row + 1];
  const auto scale = row_end - row_start;
  for (auto i = row_start; i < row_end; ++i) {
    const auto selected_row = map[i];
    const auto grad = grad_output[row];
    atomicAdd(&grad_input[selected_row], grad / scale);
  }
}

template <typename T>
__global__
__launch_bounds__(BLOCK_SIZE) void select_segment_csr_sum_bwd_1d_kernel(
    T* __restrict__ grad_input, const size_t num_rows,
    const T* __restrict__ grad_output, const int64_t* __restrict__ map,
    const int64_t* __restrict__ indptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto row = tid;
  if (row >= num_rows) {
    return;
  }
  const auto row_start = indptr[row];
  const auto row_end = indptr[row + 1];
  for (auto i = row_start; i < row_end; ++i) {
    const auto selected_row = map[i];
    const auto grad = grad_output[row];
    atomicAdd(&grad_input[selected_row], grad);
  }
}
}  // namespace

void select_segment_csr_mean_bwd_1d(paddle::Tensor& grad_input,
                                    const paddle::Tensor& grad_output,
                                    const paddle::Tensor& map,
                                    const paddle::Tensor& indptr) {
  const auto num_rows = grad_output.shape()[0];

  const auto total_count = num_rows;
  const dim3 block(min(BLOCK_SIZE, total_count));
  const dim3 grid(ceil_div(total_count, block.x));

  PD_DISPATCH_FLOATING_TYPES(
      grad_output.type(), "select_segment_csr_mean_bwd", [&] {
        select_segment_csr_mean_bwd_1d_kernel<<<grid, block, 0,
                                                grad_output.stream()>>>(
            grad_input.data<data_t>(), num_rows, grad_output.data<data_t>(),
            map.data<int64_t>(), indptr.data<int64_t>());
      });
}

void select_segment_csr_sum_bwd_1d(paddle::Tensor& grad_input,
                                   const paddle::Tensor& grad_output,
                                   const paddle::Tensor& map,
                                   const paddle::Tensor& indptr) {
  const auto num_rows = grad_output.shape()[0];

  const auto total_count = num_rows;
  const dim3 block(min(BLOCK_SIZE, total_count));
  const dim3 grid(ceil_div(total_count, block.x));

  PD_DISPATCH_FLOATING_TYPES(
      grad_output.type(), "select_segment_csr_sum_bwd", [&] {
        select_segment_csr_sum_bwd_1d_kernel<<<grid, block, 0,
                                               grad_output.stream()>>>(
            grad_input.data<data_t>(), num_rows, grad_output.data<data_t>(),
            map.data<int64_t>(), indptr.data<int64_t>());
      });
}

namespace {
template <typename T>
__global__ __launch_bounds__(BLOCK_SIZE) void select_segment_csr_sum_2d_kernel(
    T* __restrict__ out, const size_t num_rows, const size_t num_cols,
    const T* __restrict__ src, const int64_t* __restrict__ map,
    const int64_t* __restrict__ indptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto row = tid / num_cols;
  const auto col = tid % num_cols;
  if (row >= num_rows || col >= num_cols) {
    return;
  }
  const auto row_start = indptr[row];
  const auto row_end = indptr[row + 1];
  double sum = 0;
  for (auto i = row_start; i < row_end; ++i) {
    const auto selected_row = map[i];
    sum += src[selected_row * num_cols + col];
  }
  out[row * num_cols + col] = sum;
}

template <typename T>
__global__ __launch_bounds__(BLOCK_SIZE) void select_segment_csr_mean_2d_kernel(
    T* __restrict__ out, const size_t num_rows, const size_t num_cols,
    const T* __restrict__ src, const int64_t* __restrict__ map,
    const int64_t* __restrict__ indptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto row = tid / num_cols;
  const auto col = tid % num_cols;
  if (row >= num_rows || col >= num_cols) {
    return;
  }
  const auto row_start = indptr[row];
  const auto row_end = indptr[row + 1];
  double sum = 0;
  for (auto i = row_start; i < row_end; ++i) {
    const auto selected_row = map[i];
    sum += src[selected_row * num_cols + col];
  }
  const auto row_length = row_end - row_start;
  if (row_length == 0) {
    out[row * num_cols + col] = 0;
  } else {
    out[row * num_cols + col] = sum / row_length;
  }
}
}  // namespace

void select_segment_csr_sum_2d(paddle::Tensor& out, const paddle::Tensor& src,
                               const paddle::Tensor& map,
                               const paddle::Tensor& indptr) {
  const auto num_rows = out.shape()[0];
  const auto num_cols = out.shape()[1];

  const auto total_count = num_rows * num_cols;
  const dim3 block(min(BLOCK_SIZE, total_count));
  const dim3 grid(ceil_div(total_count, block.x));

  PD_DISPATCH_FLOATING_TYPES(out.type(), "select_segment_csr_sum", [&] {
    select_segment_csr_sum_2d_kernel<<<grid, block, 0, src.stream()>>>(
        out.data<data_t>(), num_rows, num_cols, src.data<data_t>(),
        map.data<int64_t>(), indptr.data<int64_t>());
  });
}

void select_segment_csr_mean_2d(paddle::Tensor& out, const paddle::Tensor& src,
                                const paddle::Tensor& map,
                                const paddle::Tensor& indptr) {
  const auto num_rows = out.shape()[0];
  const auto num_cols = out.shape()[1];

  const auto total_count = num_rows * num_cols;
  const dim3 block(min(BLOCK_SIZE, total_count));
  const dim3 grid(ceil_div(total_count, block.x));

  PD_DISPATCH_FLOATING_TYPES(out.type(), "select_segment_csr_mean", [&] {
    select_segment_csr_mean_2d_kernel<<<grid, block, 0, src.stream()>>>(
        out.data<data_t>(), num_rows, num_cols, src.data<data_t>(),
        map.data<int64_t>(), indptr.data<int64_t>());
  });
}

namespace {
template <typename T>
__global__
__launch_bounds__(BLOCK_SIZE) void select_segment_csr_mean_bwd_2d_kernel(
    T* __restrict__ grad_input, const size_t num_rows, const size_t num_cols,
    const T* __restrict__ grad_output, const int64_t* __restrict__ map,
    const int64_t* __restrict__ indptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto row = tid / num_cols;
  const auto col = tid % num_cols;
  if (row >= num_rows || col >= num_cols) {
    return;
  }
  const auto row_start = indptr[row];
  const auto row_end = indptr[row + 1];
  const auto scale = row_end - row_start;
  for (auto i = row_start; i < row_end; ++i) {
    const auto selected_row = map[i];
    const auto grad = grad_output[row * num_cols + col];
    atomicAdd(&grad_input[selected_row * num_cols + col], grad / scale);
  }
}

template <typename T>
__global__
__launch_bounds__(BLOCK_SIZE) void select_segment_csr_sum_bwd_2d_kernel(
    T* __restrict__ grad_input, const size_t num_rows, const size_t num_cols,
    const T* __restrict__ grad_output, const int64_t* __restrict__ map,
    const int64_t* __restrict__ indptr) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  const auto row = tid / num_cols;
  const auto col = tid % num_cols;
  if (row >= num_rows || col >= num_cols) {
    return;
  }
  const auto row_start = indptr[row];
  const auto row_end = indptr[row + 1];
  for (auto i = row_start; i < row_end; ++i) {
    const auto selected_row = map[i];
    const auto grad = grad_output[row * num_cols + col];
    atomicAdd(&grad_input[selected_row * num_cols + col], grad);
  }
}
}  // namespace

void select_segment_csr_mean_bwd_2d(paddle::Tensor& grad_input,
                                    const paddle::Tensor& grad_output,
                                    const paddle::Tensor& map,
                                    const paddle::Tensor& indptr) {
  const auto num_rows = grad_output.shape()[0];
  const auto num_cols = grad_output.shape()[1];

  const auto total_count = num_rows * num_cols;
  const dim3 block(min(BLOCK_SIZE, total_count));
  const dim3 grid(ceil_div(total_count, block.x));

  PD_DISPATCH_FLOATING_TYPES(
      grad_output.type(), "select_segment_csr_mean_bwd", [&] {
        select_segment_csr_mean_bwd_2d_kernel<<<grid, block, 0,
                                                grad_output.stream()>>>(
            grad_input.data<data_t>(), num_rows, num_cols,
            grad_output.data<data_t>(), map.data<int64_t>(),
            indptr.data<int64_t>());
      });
}

void select_segment_csr_sum_bwd_2d(paddle::Tensor& grad_input,
                                   const paddle::Tensor& grad_output,
                                   const paddle::Tensor& map,
                                   const paddle::Tensor& indptr) {
  const auto num_rows = grad_output.shape()[0];
  const auto num_cols = grad_output.shape()[1];

  const auto total_count = num_rows * num_cols;
  const dim3 block(min(BLOCK_SIZE, total_count));
  const dim3 grid(ceil_div(total_count, block.x));

  PD_DISPATCH_FLOATING_TYPES(
      grad_output.type(), "select_segment_csr_sum_bwd", [&] {
        select_segment_csr_sum_bwd_2d_kernel<<<grid, block, 0,
                                               grad_output.stream()>>>(
            grad_input.data<data_t>(), num_rows, num_cols,
            grad_output.data<data_t>(), map.data<int64_t>(),
            indptr.data<int64_t>());
      });
}
}  // namespace fused_segment_csr::impl
