import os

import paddle

############################## 相关utils函数，如下 ##############################
############################ PaConvert 自动生成的代码 ###########################

def _set_num_threads(int):
    os.environ['CPU_NUM'] = str(int)

def _Tensor_split(self, split_size, dim=0):
    if isinstance(split_size, int):
        return paddle.split(self, self.shape[dim] // split_size, dim)
    else:
        return paddle.split(self, split_size, dim)

setattr(paddle.Tensor, "split", _Tensor_split)

def device2int(device):
    if isinstance(device, str):
        device = device.replace('cuda', 'gpu')
        device = device.replace('gpu:', '')
    return int(device)


def normalize_device_spec(device):
    if device is None:
        return None
    if not isinstance(device, str):
        raise TypeError(f"device must be a string or None, got {type(device).__name__}")
    device = device.strip()
    if not device:
        raise ValueError("device must not be empty")
    if device.startswith("cuda"):
        return device.replace("cuda", "gpu", 1)
    return device


def _iter_auto_device_candidates():
    current_device = normalize_device_spec(paddle.device.get_device())
    if current_device != "cpu":
        yield current_device

    get_custom_device_types = getattr(paddle.device, "get_all_custom_device_type", None)
    if callable(get_custom_device_types):
        for custom_device in get_custom_device_types() or []:
            yield normalize_device_spec(custom_device)

    yield "gpu"
    yield "xpu"
    yield "cpu"


def resolve_runtime_device(device="auto"):
    normalized_device = normalize_device_spec(device or "auto")
    if normalized_device != "auto":
        return normalized_device

    seen = set()
    last_error = None
    for candidate in _iter_auto_device_candidates():
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            paddle.set_device(candidate)
            return candidate
        except ValueError as err:
            last_error = err

    if last_error is not None:
        raise last_error
    return "cpu"


def set_runtime_device(device="auto"):
    resolved_device = resolve_runtime_device(device)
    if normalize_device_spec(device or "auto") != "auto":
        paddle.set_device(resolved_device)
    return resolved_device


def move_to_device(obj, device):
    normalized_device = normalize_device_spec(device)
    if normalized_device is None:
        return obj
    if not hasattr(obj, "to"):
        raise TypeError(f"object of type {type(obj).__name__} cannot be moved to a device")
    return obj.to(device=normalized_device)
############################## 相关utils函数，如上 ##############################
