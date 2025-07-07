# include <iostream>
# include <math.h>
# include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"
#include "paddle/phi/backends/all_context.h"
// #include "/workspace/Paddle/Paddle/paddle/phi/kernels/funcs/layer_norm_impl.cu.h"


#define FULL_WARP_MASK 0xFFFFFFFF
#define CREATE_SHFL_MASK(mask, predicate) \
  mask = __ballot_sync(FULL_WARP_MASK, (predicate))
#define HOSTDEVICE __host__ __device__

#define FIXED_BLOCK_DIM_CASE_BASE(log2_block_dim, ...)  \
  case (1 << (log2_block_dim)): {                       \
    constexpr auto kBlockDim = (1 << (log2_block_dim)); \
    __VA_ARGS__;                                        \
  } break

#define FIXED_BLOCK_DIM_CASE(...)              \
  FIXED_BLOCK_DIM_CASE_BASE(9, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(8, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(7, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(6, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(5, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(4, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(3, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(2, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(1, ##__VA_ARGS__)

template <typename T>
__forceinline__ __device__ T
CudaShuffleDownSync(unsigned mask, T val, int delta, int width = warpSize) {
  return __shfl_down_sync(mask, val, static_cast<unsigned>(delta), width);
}


template <typename U>
static __forceinline__ __device__ U WarpReduceSum(U val) {
  unsigned mask = 0u;
  CREATE_SHFL_MASK(mask, true);
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += CudaShuffleDownSync(mask, val, offset);
  }
  return val;
}

template <typename T>
inline HOSTDEVICE T roundWithTiesToEven(T x) {
  T xLower = floor(x);
  T xUpper = ceil(x);
  // x is in interval [xl,xu]. Choose closest of two bounds, breaking ties to
  // even.
  T dLower = x - xLower;
  T dUpper = xUpper - x;
  return static_cast<T>(
      (dLower == dUpper ? fmod(xLower, 2.0F) == 0.0F : dLower < dUpper)
          ? xLower
          : xUpper);
}



template <typename T>
__forceinline__ __device__ int8_t quant_helper(const T input,
                                               const float scale,
                                               const int round_type,
                                               const float max_bound,
                                               const float min_bound) {
  float quant_value = max_bound * scale * static_cast<float>(input);

  if (round_type == 0) {
    quant_value = static_cast<float>(roundWithTiesToEven(quant_value));
  } else {
    quant_value = static_cast<float>(round(quant_value));
  }
  quant_value = quant_value > max_bound ? max_bound : quant_value;
  quant_value = quant_value < min_bound ? min_bound : quant_value;
  return static_cast<int8_t>(quant_value);
}

template <typename U>
__forceinline__ __device__ U BlockReduceSum(U val, U *shared) {
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = WarpReduceSum(val);          // Each warp performs partial reduction
  if (lane == 0) shared[wid] = val;  // Write reduced value to shared memory
  __syncthreads();                   // Wait for all partial reductions
  // read from shared memory only if that warp existed
  val =
      (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : static_cast<U>(0);

  if (wid == 0) val = WarpReduceSum(val);  // Final reduce within first warp

  return val;
}

template <typename T>
__inline__ __device__ T rsqrt_(const T val) {
  return static_cast<T>(1) / sqrt(val);
}

template <>
__inline__ __device__ float rsqrt_(const float val) {
  return rsqrtf(val);
}

template <>
__inline__ __device__ double rsqrt_(const double val) {
  return ::rsqrt(val);
}


inline static int GetDesiredBlockDim(int dim) {
  if (dim > 128) {
    return 256;
  } else if (dim > 64) {
    return 128;
  } else if (dim > 32) {
    return 64;
  } else {
    return 32;
  }
}



template <typename T,
          typename U,
          int BlockDim,
          bool ScaleBiasWithSameTypeX = false,
          typename InType = T,
          typename OutType = T>
__global__ void LayerNormForward(
    InType *x,
    const float *scale,
    const float *bias,
    U *mean,
    U *var,
    float epsilon,
    int64_t feature_size,
    const float *dequant_out_scale_data = nullptr,
    const int quant_out_scale_offset = 0,
    const float quant_in_scale = 1.0,
    const int quant_round_type = 1,
    const float quant_max_bound = 127.0,
    const float quant_min_bound = -127.0) {
    __shared__ U mean_share;
    __shared__ U var_share;
    __shared__ U shared_mean[32];  // threadIdx.x / warpSize <= kMaxBlockDim /
                                    // warpSize <= 1024/32 = 32;
    __shared__ U shared_var[32];

    int64_t beg_idx = blockIdx.x * feature_size + threadIdx.x;
    int64_t end_idx = (blockIdx.x + 1) * feature_size;

    // Step 1: Reduce to calculate mean and var
    U mean_val = 0;
    U var_val = 0;
    for (int64_t i = beg_idx; i < end_idx; i += BlockDim) {
    U tmp = static_cast<U>(x[i]);
    mean_val += tmp;
    var_val += (tmp * tmp);
    }

    mean_val = BlockReduceSum<U>(mean_val, shared_mean);
    var_val = BlockReduceSum<U>(var_val, shared_var);

    if (threadIdx.x == 0) {
    auto scale = static_cast<U>(static_cast<float>(1.) /
                                static_cast<float>(feature_size));
    auto tmp = mean_val * scale;
    mean[blockIdx.x] = mean_share = static_cast<U>(tmp);
    var_share = static_cast<U>(var_val * scale - mean_share * mean_share);
    var_share = var_share > U(0) ? var_share : U(0);
    var[blockIdx.x] = var_share;
    }
    __syncthreads();

    mean_val = mean_share;
    // U mean_val_ = *mean;
    // U* mean_share_ = &mean_val_;
    
    // T* mean_share_ = nullptr;
    extern __shared__ U shared_mem[];
    T* mean_share_ = (T*)shared_mem;
    if (threadIdx.x == 0){
      // mean_share_ = (T*)malloc(sizeof(T));
      *mean_share_ = *mean;
    }
    __syncthreads();


    U invvar = rsqrt_<U>(var_share + static_cast<U>(epsilon));

  // Step 2: Calculate y
  if (scale != nullptr) {
    if (bias != nullptr) {
      for (int64_t i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        if (std::is_same<OutType, int8_t>::value) {
          mean_share_[i] = quant_helper(
              static_cast<T>(static_cast<U>(scale[j]) *
                                 (static_cast<U>(x[i]) - mean_val) * invvar +
                             static_cast<U>(bias[j])),
              quant_in_scale,
              quant_round_type,
              quant_max_bound,
              quant_min_bound);
        } else {
          x[i] = static_cast<OutType>(static_cast<U>(scale[j]) *
                                          (static_cast<U>(x[i]) - mean_val) *
                                          invvar +
                                      static_cast<U>(bias[j]));
        }
      }
    } else {
      for (int64_t i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        if (std::is_same<OutType, int8_t>::value) {
          var[i] = quant_helper(
              static_cast<T>(static_cast<U>(scale[j]) *
                             (static_cast<U>(x[i]) - mean_val) * invvar),
              quant_in_scale,
              quant_round_type,
              quant_max_bound,
              quant_min_bound);
        } else {
          x[i] =
              static_cast<OutType>(static_cast<U>(scale[j]) *
                                   (static_cast<U>(x[i]) - mean_val) * invvar);
        }
      }
    }
  } else {  // scale == nullptr
    if (bias != nullptr) {
      for (int64_t i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        if (std::is_same<OutType, int8_t>::value) {
          mean_share_[i] = quant_helper(
              static_cast<T>((static_cast<U>(x[i]) - mean_val) * invvar +
                             static_cast<U>(bias[j])),
              quant_in_scale,
              quant_round_type,
              quant_max_bound,
              quant_min_bound);
        } else {
          mean_share_[i] =
              static_cast<OutType>((static_cast<U>(x[i]) - mean_val) * invvar +
                                   static_cast<U>(bias[j]));
        }
      }
    } else {
      for (int64_t i = beg_idx, j = threadIdx.x; i < end_idx;
           i += BlockDim, j += BlockDim) {
        if (std::is_same<OutType, int8_t>::value) {
          mean[i] = quant_helper(
              static_cast<T>((static_cast<U>(x[i]) - mean_val) * invvar),
              quant_in_scale,
              quant_round_type,
              quant_max_bound,
              quant_min_bound);
        } else {
          mean_share_[i] =
              static_cast<OutType>((static_cast<U>(x[i]) - mean_val) * invvar);
        }
      }
    }
  }
}

// #define CHECK_CUSTOM_INPUT(x) \
//   PD_CHECK(x.is_custom_device(), #x " must be a custom Tensor.")

// void* GetStream(const paddle::Tensor& x) {
//   CHECK_CUSTOM_INPUT(x);
//   auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(x.place());
//   auto custom_ctx = static_cast<const phi::CustomContext*>(dev_ctx);
//   void* stream = custom_ctx->stream();
//   PD_CHECK(stream != nullptr);
//   return stream;
// }


#include "paddle/extension.h"
void layernorm_fwd_inplace(
    paddle::Tensor& x,
    const paddle::Tensor& weight,
    const paddle::Tensor& bias
    ) {
    std::vector<int64_t> shape = x.shape();
    int64_t B = shape[0], T = shape[1], C = shape[2];
    float* var_data;
    float* mean_data;
    cudaMalloc(&var_data, B * T * sizeof(float));
    cudaMalloc(&mean_data, B * T * sizeof(float));
    std::cout << "GetDesiredBlockDim(T)" << GetDesiredBlockDim(T) << std::endl;
    
    // void* stream = GetStream(x);
    int64_t feature_size = C;
    int64_t batch_size = B * T;
    auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(x.place());
    auto custom_ctx = static_cast<const phi::GPUContext*>(dev_ctx);
    auto stream = custom_ctx->stream();
    const auto *void_scale_data = weight.data<float>();
    const auto *void_bias_data = bias.data<float>();
    auto *x_data = x.data<float>();
    float epsilon = 1e-5;



#define PADDLE_LAUNCH_LAYERNORM_FWD(ScaleBiasT, IsScaleBiasSameDTypeWithX) \
  do {                                                                     \
    switch (GetDesiredBlockDim(feature_size)) {                \
      FIXED_BLOCK_DIM_CASE(                                                \
          LayerNormForward<T, float, kBlockDim, IsScaleBiasSameDTypeWithX> \
          <<<batch_size, kBlockDim, 0, stream>>>(                          \
              x_data,                                                      \
              static_cast<const ScaleBiasT *>(void_scale_data),            \
              static_cast<const ScaleBiasT *>(void_bias_data),             \
              mean_data,                                                   \
              var_data,                                                    \
              epsilon,                                                     \
              feature_size));                                              \
      default:                                                             \
        break;                                                             \
    }                                                                      \
  } while (0)

  // PADDLE_LAUNCH_LAYERNORM_FWD(float, true);
  switch (GetDesiredBlockDim(T)) {
    FIXED_BLOCK_DIM_CASE(
        LayerNormForward<float, float, kBlockDim, true>
        <<<B*T, kBlockDim, 0, stream>>>(
            x.data<float>(),
            static_cast<const float *>(void_scale_data),
            static_cast<const float *>(void_bias_data),
            mean_data,
            var_data,
            epsilon,
            feature_size));
    default:
        printf(
            "Product from begin_norm_axis to end in layer_norm must be larger "
            "than 1");
    break;
}


    cudaFree (var_data);
    cudaFree (mean_data);
}

PD_BUILD_OP(layernorm_inplace)
    .Inputs({"input", "weight", "bias"})
    .Outputs({})
    .Attrs({})
    .SetKernelFn(PD_KERNEL(layernorm_fwd_inplace));
