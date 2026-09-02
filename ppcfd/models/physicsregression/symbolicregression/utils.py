import argparse
import errno
import math
import os
import signal
import time
from functools import partial, wraps

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}
CUDA = True


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def to_cuda(*args, use_cpu=False, device=None):
    """
    将张量移动到设备（统一使用 .to() 方法）

    注意：虽然函数名为 to_cuda，但实际支持所有 PaddlePaddle 设备类型，
    包括 gpu, cpu, iluvatar, npu, xpu 等。

    参数：
        *args: 可变数量的张量或其他对象
            - paddle.Tensor: 将被移动到目标设备
            - None: 保持为 None（不处理）
            - 其他对象: 原样返回
        use_cpu: bool, 默认 False
            如果为 True，跳过设备移动，直接返回原对象
        device: str/int/None, 默认 None
            目标设备字符串或ID：
            - None: 使用默认GPU（gpu:0）
            - "cuda:0": 自动转换为 "gpu:0"
            - "gpu:0": 直接使用
            - "iluvatar:0": Custom device
            - 0: 转换为 "gpu:0"

    返回：
        tuple - 移动到目标设备的对象元组（None值保持不变）

    示例：
        >>> x = paddle.randn([10, 5])
        >>> x_gpu, = to_cuda(x, device="cuda:0")
        >>> str(x_gpu.place)
        'Place(gpu:0)'

        >>> y, none_val, z = to_cuda(paddle.randn([5]), None, paddle.randn([3]))
        >>> none_val is None
        True
    """
    if not CUDA or use_cpu:
        return args

    # 标准化设备字符串
    from ..paddle_utils import device2str

    if device is None:
        device_str = 'gpu:0'  # 默认设备
    else:
        device_str = device2str(device)

    # 使用 .to() 方法移动所有张量
    return tuple(
        (None if x is None else x.to(device_str))
        for x in args
    )



class MyTimeoutError(BaseException):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(repeat_id, signum, frame):
            signal.signal(signal.SIGALRM, partial(_handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            raise MyTimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator
