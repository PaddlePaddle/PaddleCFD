import paddle
from paddle.amp import GradScaler

FLOAT32_ALIASES = ["float32", 32]
FLOAT16_ALIASES = ["float16", 16]
BFLOAT16_ALIASES = ["bfloat16", "bf16"]
VALID_PRECISIONS = FLOAT32_ALIASES + FLOAT16_ALIASES + BFLOAT16_ALIASES


def _normalize_device(device):
    if device is None:
        return paddle.device.get_device()
    if isinstance(device, str):
        return device
    if hasattr(device, 'gpu_device_id'):
        return f"gpu:{device.gpu_device_id()}"
    if hasattr(device, 'is_gpu_place') and device.is_gpu_place():
        return 'gpu'
    if hasattr(device, 'is_cpu_place') and device.is_cpu_place():
        return 'cpu'
    return str(device)


def get_supported_precision(desired_precision, device, log_fn=print):
    assert desired_precision in VALID_PRECISIONS
    if desired_precision in FLOAT32_ALIASES:
        return paddle.float32
    if desired_precision in FLOAT16_ALIASES:
        desired_precision = 'float16'
    if desired_precision in BFLOAT16_ALIASES:
        desired_precision = 'bfloat16'

    if desired_precision == 'bfloat16':
        if is_bfloat16_compatible(device):
            return paddle.bfloat16
        if is_float16_compatible(device):
            log_fn('bfloat16 not supported -> using float32 (float16 could lead to under-/overflows)')
            return paddle.float32

    if desired_precision == 'float16':
        if is_float16_compatible(device):
            return paddle.float16
        if is_bfloat16_compatible(device):
            log_fn('float16 not supported -> using bfloat16')
            return paddle.bfloat16

    log_fn('float16/bfloat16 not supported -> using float32')
    return paddle.float32


def _is_compatible(device, dtype):
    device = _normalize_device(device)
    if not device.startswith('gpu'):
        return False
    if not paddle.device.is_compiled_with_cuda():
        return False
    if dtype == paddle.float16:
        return True
    if dtype == paddle.bfloat16:
        major, _ = paddle.device.cuda.get_device_capability()
        return major >= 8
    return False


def is_bfloat16_compatible(device):
    return _is_compatible(device, paddle.bfloat16)


def is_float16_compatible(device):
    return _is_compatible(device, paddle.float16)


class NoopGradScaler:
    @staticmethod
    def scale(loss):
        return loss

    @staticmethod
    def unscale_(optimizer):
        pass

    @staticmethod
    def step(optimizer, *args, **kwargs):
        optimizer.step(*args, **kwargs)

    @staticmethod
    def update():
        pass


class NoopContext:
    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        return False


def get_grad_scaler_and_autocast_context(precision, device):
    if precision == paddle.float32:
        return NoopGradScaler(), NoopContext()
    if precision == paddle.bfloat16:
        return NoopGradScaler(), paddle.amp.auto_cast(dtype='bfloat16')
    if precision == paddle.float16:
        return GradScaler(), paddle.amp.auto_cast(dtype='float16')
    raise NotImplementedError
