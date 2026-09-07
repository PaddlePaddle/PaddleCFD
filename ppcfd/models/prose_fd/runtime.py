"""Runtime device helpers for PROSE-FD models."""

import paddle


RUNTIME_DEVICE = "cpu"


def set_runtime_device(device: str):
    global RUNTIME_DEVICE
    RUNTIME_DEVICE = device


def get_runtime_device() -> str:
    return RUNTIME_DEVICE


def to_device(*args, use_cpu=False, device: str | None = None):
    target = "cpu" if use_cpu else (device or RUNTIME_DEVICE)
    moved = [None if x is None else x.to(target) for x in args]
    if len(args) == 1:
        return moved[0]
    return moved


def to_cuda(*args, use_cpu=False):
    return to_device(*args, use_cpu=use_cpu)


def get_amp_device_type() -> str:
    return RUNTIME_DEVICE.split(":")[0]


def max_memory_allocated_mb():
    if RUNTIME_DEVICE.startswith("gpu"):
        return paddle.device.cuda.max_memory_allocated() / 1024**2
    return None


def sync_tensor(t):
    if not paddle.distributed.is_initialized():
        return t
    source_place = t.place
    t_sync = t.to(RUNTIME_DEVICE)
    paddle.distributed.barrier()
    paddle.distributed.all_reduce(t_sync, op=paddle.distributed.ReduceOp.SUM)
    return t_sync.to(source_place)
