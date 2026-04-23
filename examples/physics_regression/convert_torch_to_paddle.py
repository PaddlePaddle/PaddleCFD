#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import paddle
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ppcfd.models.physicsregression.symbolicregression.checkpoint_io import normalize_params
from ppcfd.models.physicsregression.symbolicregression.envs import build_env
from ppcfd.models.physicsregression.symbolicregression.model import build_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch checkpoint to PaddlePaddle native model bundle"
    )
    parser.add_argument("--torch-model", default="./model.pt")
    parser.add_argument("--paddle-model", default="./model.pdparams")
    return parser.parse_args()


def validate_state_dict(torch_state_dict, paddle_state_dict, module_name):
    torch_keys = list(torch_state_dict.keys())
    paddle_keys = list(paddle_state_dict.keys())
    if torch_keys != paddle_keys:
        raise ValueError(f"{module_name} key mismatch")
    for key in torch_keys:
        if tuple(torch_state_dict[key].shape) != tuple(paddle_state_dict[key].shape):
            raise ValueError(
                f"{module_name}.{key} shape mismatch: "
                f"{tuple(torch_state_dict[key].shape)} != {tuple(paddle_state_dict[key].shape)}"
            )


def convert_torch_state_dict(torch_state_dict):
    converted = {}
    for key, tensor in torch_state_dict.items():
        converted[key] = paddle.to_tensor(tensor.detach().cpu().numpy())
    return converted


def main():
    args = parse_args()
    torch_model_path = Path(args.torch_model).resolve()
    paddle_model_path = Path(args.paddle_model).resolve()
    paddle_model_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(torch_model_path, map_location="cpu", weights_only=False)
    params = normalize_params(checkpoint["params"])
    params.cpu = True

    env = build_env(params)
    modules = build_modules(env, params)

    save_dict = {"params": vars(params).copy()}
    for module_name in ("embedder", "encoder", "decoder"):
        paddle_state_dict = modules[module_name].state_dict()
        torch_state_dict = checkpoint[module_name]
        validate_state_dict(torch_state_dict, paddle_state_dict, module_name)
        converted = convert_torch_state_dict(torch_state_dict)
        modules[module_name].set_state_dict(converted)
        save_dict[module_name] = modules[module_name].state_dict()
        print(f"[OK] {module_name}: {len(converted)} tensors verified and loaded")

    paddle.save(save_dict, str(paddle_model_path))
    print(f"[DONE] Saved converted checkpoint to: {paddle_model_path}")


if __name__ == "__main__":
    main()
