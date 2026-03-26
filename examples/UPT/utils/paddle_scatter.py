# 文件路径: UPT/src/utils/paddle_scatter.py
import paddle

# def segment_csr(src, indptr, reduce="sum", out=None):
#     """
#     用 Paddle 模拟 torch_scatter.segment_csr
#     src: [N, D] 输入特征
#     indptr: [M+1] CSR 格式的索引指针 (定义了 M 个段)
#     reduce: 'sum' | 'mean' | 'add'
#     """
#     # 1. 计算每个段包含多少个元素 (counts)
#     # indptr: [0, 2, 5] -> diff: [2, 3] -> segment 0 有 2 个元素, segment 1 有 3 个
#     diff = indptr[1:] - indptr[:-1]
    
#     # 2. 将 CSR 指针展开为 COO 格式的 segment_ids
#     # 效果: [0, 0, 1, 1, 1]
#     # 注意：确保 indptr 是 int 类型 (int32 或 int64)
#     if indptr.dtype not in [paddle.int32, paddle.int64]:
#         indptr = indptr.cast('int64')
        
#     segment_ids = paddle.repeat_interleave(
#         paddle.arange(len(diff), dtype=indptr.dtype), 
#         diff
#     )
    
#     # 3. 确定输出形状 [M, D]
#     if out is None:
#         out_shape = list(src.shape)
#         out_shape[0] = len(diff)
#         out = paddle.zeros(out_shape, dtype=src.dtype)
    
#     # 4. 执行聚合
#     if reduce == "sum" or reduce == "add":
#         # scatter_nd_add 需要 index 为 [N, 1] (针对 dim 0)
#         index = segment_ids.unsqueeze(-1)
#         # 如果 src 和 index 设备不一致，强制同步
#         if index.place != src.place:
#             index = index.to(src.place)
            
#         out = paddle.scatter_nd_add(out, index, src)
#         return out
        
#     elif reduce == "mean":
#         # 先求和
#         out_sum = segment_csr(src, indptr, reduce="sum")
        
#         # 再除以计数 (注意处理除零，防止 NaN)
#         counts = diff.cast(src.dtype).unsqueeze(-1)
#         # 将 count=0 的地方设为 1，避免除以 0 (反正分子也是 0)
#         counts = paddle.where(counts == 0, paddle.ones_like(counts), counts)
        
#         return out_sum / counts
        
#     else:
#         raise NotImplementedError(f"Current Paddle workaround does not support reduce='{reduce}'")

import paddle
from typing import Optional, Tuple

def indptr_to_segment_ids(indptr: paddle.Tensor) -> paddle.Tensor:
    """将 CSR 的 indptr 转换为 COO 的 segment_ids"""
    # 计算每个 segment 的长度
    lengths = indptr[1:] - indptr[:-1]
    # 使用 repeat_interleave 生成 ids
    # 例如 indptr=[0, 2, 5] -> lengths=[2, 3] -> ids=[0, 0, 1, 1, 1]
    return paddle.repeat_interleave(
        paddle.arange(len(lengths), dtype=indptr.dtype), 
        lengths
    )

def segment_sum_csr(src: paddle.Tensor, indptr: paddle.Tensor,
                    out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    # 注意：Paddle 的 segment_sum 目前不支持直接传入 out 进行 in-place 操作
    segment_ids = indptr_to_segment_ids(indptr.reshape([-1]))
    return paddle.geometric.segment_sum(src, segment_ids)

def segment_add_csr(src: paddle.Tensor, indptr: paddle.Tensor,
                    out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    return segment_sum_csr(src, indptr, out)

def segment_mean_csr(src: paddle.Tensor, indptr: paddle.Tensor,
                     out: Optional[paddle.Tensor] = None) -> paddle.Tensor:
    segment_ids = indptr_to_segment_ids(indptr.reshape([-1]))
    return paddle.geometric.segment_mean(src, segment_ids)

def segment_min_csr(src: paddle.Tensor, indptr: paddle.Tensor,
                    out: Optional[paddle.Tensor] = None) -> Tuple[paddle.Tensor, paddle.Tensor]:
    segment_ids = indptr_to_segment_ids(indptr.reshape([-1]))
    # Paddle 的 segment_min 返回的是 (output, arg_min)
    return paddle.geometric.segment_min(src, segment_ids), None

def segment_max_csr(src: paddle.Tensor, indptr: paddle.Tensor,
                    out: Optional[paddle.Tensor] = None) -> Tuple[paddle.Tensor, paddle.Tensor]:
    segment_ids = indptr_to_segment_ids(indptr.reshape([-1]))
    # Paddle 的 segment_max 返回的是 (output, arg_max)
    return paddle.geometric.segment_max(src, segment_ids), None

def segment_csr(src: paddle.Tensor, indptr: paddle.Tensor,
                out: Optional[paddle.Tensor] = None,
                reduce: str = "sum") -> paddle.Tensor:
    """
    老爷，这是通用的调度函数
    """
    # 处理 indptr 的广播和维度问题（Paddle 的 segment_ids 需为 1D）
    if indptr.ndim > 1:
        # 假设 indptr 在所有 batch 上是一致的，取第一行
        indptr = indptr[0]

    if reduce in ['sum', 'add']:
        return segment_sum_csr(src, indptr, out)
    elif reduce == 'mean':
        return segment_mean_csr(src, indptr, out)
    elif reduce == 'min':
        return segment_min_csr(src, indptr, out)[0]
    elif reduce == 'max':
        return segment_max_csr(src, indptr, out)[0]
    else:
        raise ValueError(f"Unsupported reduce type: {reduce}")

