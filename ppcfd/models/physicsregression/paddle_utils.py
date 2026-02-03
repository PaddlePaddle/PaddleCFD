
import paddle

############################## 相关utils函数，如下 ##############################
############################ PaConvert 自动生成的代码 ###########################

def device2str(device):
    """
    标准化设备字符串为 PaddlePaddle 格式

    支持的输入格式：
        - "cuda:0" / "cuda:1" → "gpu:0" / "gpu:1"（PyTorch 兼容）
        - "gpu:0" → "gpu:0"（保持不变）
        - "iluvatar:0" → "iluvatar:0"（Custom device）
        - "cpu" → "cpu"（CPU 模式）
        - 0 / 1 → "gpu:0" / "gpu:1"（整数索引）

    返回：str - PaddlePaddle 兼容的设备字符串

    示例：
        >>> device2str("cuda:0")
        'gpu:0'
        >>> device2str("iluvatar:1")
        'iluvatar:1'
        >>> device2str(2)
        'gpu:2'
    """
    if isinstance(device, int):
        return f"gpu:{device}"

    if not isinstance(device, str):
        raise TypeError(f"device must be str or int, got {type(device)}")

    device = device.strip().lower()

    # 处理 cuda → gpu 转换
    if device.startswith('cuda'):
        device = device.replace('cuda', 'gpu')

    # 验证格式
    valid_prefixes = ['gpu', 'cpu', 'iluvatar', 'npu', 'xpu']
    if not any(device.startswith(prefix) for prefix in valid_prefixes):
        raise ValueError(
            f"Invalid device string: {device}. "
            f"Supported: {', '.join(valid_prefixes)}"
        )

    return device


def device2int(device):
    """
    将设备字符串转换为整数ID

    ⚠️ DEPRECATED: 此函数主要用于兼容旧代码中的 Module.cuda(device=id)
    新代码应使用 device2str() + .to(device)，因为：
    1. .to() 支持所有设备类型（包括 custom device）
    2. Module.cuda() 不支持 custom device

    参数：
        device: str 或 int
            - "cuda:0" / "gpu:0" / "iluvatar:0" → 0
            - "cuda:1" / "gpu:1" → 1
            - 0 → 0（直接返回）

    返回：int - 设备ID
    """
    if isinstance(device, str):
        # 移除设备类型前缀,只保留ID
        device = device.replace('cuda', 'gpu')
        if ':' in device:
            device = device.split(':')[-1]  # 提取冒号后的ID部分
        else:
            # 如果没有冒号(如"cpu"),返回0
            device = '0'
    return int(device)

def _Tensor_max(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "_max", _Tensor_max)
############################## 相关utils函数，如上 ##############################

