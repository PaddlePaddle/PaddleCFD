# TODO this should be moved into a seperate library
import paddle.distributed as dist


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    return 1
