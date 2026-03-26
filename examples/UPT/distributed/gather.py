import einops
import paddle

from .config import get_world_size, is_distributed
from .functional.all_gather_grad_autograd import AllGatherGradAutograd
from .functional.all_gather_grad_overwrite import AllGatherGradOverwrite


# def get_device_and_bfloat16supported():
#     if not is_distributed():
#         return paddle.device("cpu"), True
#     if paddle.distributed.get_backend() == "nccl":
#         return paddle.device("cuda"), True
#     if paddle.distributed.get_backend() == "gloo":
#         return paddle.device("cpu"), False
#     raise NotImplementedError

import paddle

def get_device_and_bfloat16supported():
    # 1. 检查是否为分布式环境 (Paddle 常用 get_world_size > 1 判断)
    is_dist = paddle.distributed.get_world_size() > 1
    
    if not is_dist:
        # 非分布式模式：返回 CPU 设备对象和 True (3090环境建议默认支持bf16)
        return paddle.CPUPlace(), True

    # 2. 获取当前分布式后端
    # Paddle 在 GPU 训练时默认就是 nccl
    backend = paddle.distributed.get_backend()

    if backend == "nccl":
        # 注意：Paddle 使用 Place 对象或字符串来表示设备
        # 这里返回当前进程对应的 GPU Place
        return paddle.CUDAPlace(paddle.distributed.get_rank()), True
    
    if backend == "gloo":
        # Gloo 通常用于 CPU 分布式
        return paddle.CPUPlace(), False

    # 3. 兜底逻辑：如果走到这里，说明是未知的后端
    # 建议返回当前默认设备，而不是直接报 NotImplementedError 导致程序崩掉
    return paddle.get_device(), False


def get_bool_gather_supported():
    if not is_distributed():
        return True
    if paddle.distributed.get_backend() == "nccl":
        return True
    if paddle.distributed.get_backend() == "gloo":
        return False
    raise NotImplementedError


def _prepare_tensor(x):
    """
    prepare for distributed communication
    - wrap primitive types into tensors
    - push tensor onto supported device
    - convert bool to float if bool gathering is not supported
    - call .contiguous if x is not in a contiguous memory block
    """
    device, bfloat16_supported = get_device_and_bfloat16supported()
    if isinstance(x, bool):
        raise RuntimeError
    if isinstance(x, (float, int, list, tuple)):
        x = paddle.to_tensor(x, place=device)
        og_device = paddle.CPUPlace()
    else:
        og_device = x.place
    if x.dtype == paddle.bfloat16 and not bfloat16_supported:
        x = x.astype(paddle.float32)
    if x.dtype == paddle.bool and not get_bool_gather_supported():
        x = x.astype(paddle.float32)
        to_bool = True
    else:
        to_bool = False
    if not x.is_contiguous():
        x = x.contiguous()
    return x.to(device), og_device, to_bool


def _all_gather_grad(x, all_gather_fn, batch_dim=0):
    x, og_device, to_bool = _prepare_tensor(x)
    if is_distributed():
        result = all_gather_fn(x)
        if result[0].ndim == 0:
            result = [r.unsqueeze(0) for r in result]
        result = paddle.concat(result, dim=batch_dim).to(og_device)
    else:
        result = _all_gather_nondistributed(x, og_device)
    if to_bool:
        result = result.bool()
    return result


def all_gather_grad(x, batch_dim=0):
    return _all_gather_grad(x, AllGatherGradAutograd.apply, batch_dim=batch_dim)


def all_gather_grad_autograd(x):
    return _all_gather_grad(x, AllGatherGradAutograd.apply)


def all_gather_grad_overwrite(x):
    return _all_gather_grad(x, AllGatherGradOverwrite.apply)


@paddle.no_grad()
def all_gather_nograd(x):
    x, og_device, to_bool = _prepare_tensor(x)
    if is_distributed():
        result = [paddle.zeros_like(x) for _ in range(get_world_size())]
        paddle.distributed.all_gather(tensor_list=result, tensor=x)
        if result[0].ndim == 0:
            result = paddle.to_tensor(result, place=og_device)
        else:
            result = paddle.concat(result).to(og_device)
    else:
        result = _all_gather_nondistributed(x, og_device).detach()
    if to_bool:
        result = result.bool()
    return result


def _all_gather_nondistributed(x, og_device):
    if x.ndim == 0:
        x = x.unsqueeze(0)
    return x.to(og_device)


def all_gather_nograd_clipped(x, max_length):
    result = all_gather_nograd(x)
    if is_distributed():
        result = einops.rearrange(
            result,
            "(num_gpus len_per_gpu) ... -> (len_per_gpu num_gpus) ...",
            num_gpus=get_world_size(),
        )
        return result[:max_length]
    return result


def all_reduce_sum_nograd(x):
    with paddle.no_grad():
        return all_reduce_sum_grad(x)


def all_reduce_sum_grad(x):
    x, og_device, to_bool = _prepare_tensor(x)
    if is_distributed():
        paddle.distributed.all_reduce(tensor=x, op=paddle.distributed.ReduceOp.SUM)
    x = x.to(og_device)
    if to_bool:
        x = x.bool()
    return x


def all_reduce_mean_grad(x):
    x, og_device, to_bool = _prepare_tensor(x)
    if is_distributed():
        x = all_reduce_sum_grad(x) / get_world_size()
    x = x.to(og_device)
    if to_bool:
        x = x.bool()
    return x
