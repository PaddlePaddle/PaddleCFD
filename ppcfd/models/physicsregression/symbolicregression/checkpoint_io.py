from argparse import Namespace
from pathlib import Path

import paddle


MODEL_BUNDLE_KEYS = ("embedder", "encoder", "decoder", "params")


def normalize_params(raw_params):
    if isinstance(raw_params, Namespace):
        return raw_params
    if isinstance(raw_params, dict):
        return Namespace(**raw_params)
    raise TypeError(f"Unsupported params type: {type(raw_params)!r}")


def load_paddle_payload(path):
    try:
        data = paddle.load(path=str(path))
    except Exception as exc:
        raise ValueError(
            f"仅支持 Paddle 原生模型文件: {path}。"
            "Torch 权重请先运行 examples/physics_regression/convert_torch_to_paddle.py。"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"模型文件结构非法: {path}")
    return dict(data)


def require_keys(data, required_keys, path):
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"文件缺少必要字段 {missing}: {path}")


def load_paddle_model_bundle(path):
    path = Path(path)
    if path.suffix in {".pt", ".pkl"}:
        raise ValueError(
            f"仅支持 Paddle 原生模型文件: {path}。"
            "Torch 权重请先运行 examples/physics_regression/convert_torch_to_paddle.py。"
        )
    data = load_paddle_payload(path)
    require_keys(data, MODEL_BUNDLE_KEYS, path)
    data["params"] = normalize_params(data["params"])
    return data


def set_layer_state(layer, state_dict, layer_name):
    try:
        layer.set_state_dict(state_dict)
    except Exception as exc:
        raise ValueError(f"{layer_name} 状态恢复失败") from exc


def set_modules_state(modules, payload, module_names=("embedder", "encoder", "decoder")):
    for module_name in module_names:
        set_layer_state(modules[module_name], payload[module_name], module_name)


def set_optimizer_state(optimizer, state_dict):
    optimizer.set_state_dict(state_dict)


def set_grad_scaler_state(scaler, state_dict):
    scaler.load_state_dict(state_dict)
