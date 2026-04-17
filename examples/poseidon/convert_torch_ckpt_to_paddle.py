#!/usr/bin/env python3
"""
Convert a local Torch Poseidon/scOT checkpoint directory into a Paddle checkpoint.

This script is intentionally project-specific:
- It reads the Torch-side ScOT model definition from ppcfd.models.poseidon.
- It preserves parameter names.
- It only transposes weights that belong to torch.nn.Linear modules, because
  Paddle stores Linear weights with the opposite layout.

Usage:
    python examples/poseidon/convert_torch_ckpt_to_paddle.py --src /path/to/torch_ckpt --dst ./models_paddle/my_model
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

import paddle
import torch
from safetensors.torch import load_file as load_safetensors

REPO_ROOT = Path(__file__).resolve().parents[2]

from ppcfd.models.poseidon import ScOT, ScOTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a local Torch Poseidon/scOT checkpoint to Paddle."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source Torch checkpoint directory.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Destination Paddle checkpoint directory. Defaults to ./models_paddle/<src_name>.",
    )
    return parser.parse_args()


def resolve_paths(src_dir: Path, dst_dir: Path | None) -> tuple[Path, Path]:
    src_dir = src_dir.expanduser().resolve()
    if dst_dir is None:
        dst_dir = REPO_ROOT / "models_paddle" / src_dir.name
    else:
        dst_dir = dst_dir.expanduser().resolve()
    return src_dir, dst_dir


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def instantiate_torch_model(config_dict: dict[str, Any]) -> ScOT:
    config = ScOTConfig(**config_dict)
    model = ScOT(config)
    model.eval()
    return model


def collect_linear_weight_names(model: torch.nn.Module) -> set[str]:
    linear_weight_names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prefix = f"{module_name}." if module_name else ""
            linear_weight_names.add(f"{prefix}weight")
    return linear_weight_names


def load_torch_checkpoint(src_dir: Path) -> tuple[OrderedDict[str, torch.Tensor], Path]:
    safetensors_path = src_dir / "model.safetensors"
    pytorch_bin_path = src_dir / "pytorch_model.bin"

    if safetensors_path.exists():
        state_dict = load_safetensors(str(safetensors_path), device="cpu")
        return OrderedDict(state_dict.items()), safetensors_path

    if pytorch_bin_path.exists():
        state_dict = torch.load(str(pytorch_bin_path), map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError(
                f"Unsupported checkpoint format in {pytorch_bin_path}: {type(state_dict)!r}"
            )
        return OrderedDict(state_dict.items()), pytorch_bin_path

    raise FileNotFoundError(
        f"No weights file found in {src_dir}. Expected model.safetensors or pytorch_model.bin."
    )


def validate_state_dict_keys(
    state_dict: OrderedDict[str, torch.Tensor], model: torch.nn.Module
) -> None:
    expected_keys = set(model.state_dict().keys())
    actual_keys = set(state_dict.keys())

    missing_keys = sorted(expected_keys - actual_keys)
    unexpected_keys = sorted(actual_keys - expected_keys)
    if missing_keys or unexpected_keys:
        raise ValueError(
            "Checkpoint keys do not match the project model definition. "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}"
        )


def convert_torch_state_dict_to_paddle_state_dict(
    state_dict: OrderedDict[str, torch.Tensor],
    linear_weight_names: set[str],
) -> tuple[OrderedDict[str, Any], list[str]]:
    converted_state_dict: OrderedDict[str, Any] = OrderedDict()
    transposed_names: list[str] = []

    for name, tensor in state_dict.items():
        tensor = tensor.detach().cpu()
        if name in linear_weight_names:
            if tensor.ndim != 2:
                raise ValueError(
                    f"Expected 2D weight for linear layer {name}, got shape {tuple(tensor.shape)}."
                )
            converted_state_dict[name] = tensor.transpose(0, 1).contiguous().numpy()
            transposed_names.append(name)
        else:
            converted_state_dict[name] = tensor.numpy()

    return converted_state_dict, transposed_names


def save_converted_checkpoint(
    converted_state_dict: OrderedDict[str, Any],
    src_config_path: Path,
    dst_dir: Path,
    weights_path: Path,
    transposed_names: list[str],
) -> dict[str, Any]:
    dst_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_config_path, dst_dir / "config.json")
    paddle_state_dict = {
        name: paddle.to_tensor(value) for name, value in converted_state_dict.items()
    }
    paddle.save(paddle_state_dict, str(dst_dir / "model_state.pdparams"))

    summary = {
        "source_directory": str(src_config_path.parent),
        "source_weights": str(weights_path),
        "output_directory": str(dst_dir),
        "num_tensors": len(converted_state_dict),
        "num_total_parameters": int(
            sum(int(torch.tensor(value.shape).prod().item()) for value in converted_state_dict.values())
        ),
        "num_transposed_linear_weights": len(transposed_names),
        "transposed_linear_weights": transposed_names,
    }
    with (dst_dir / "conversion_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def convert_checkpoint(src_dir: Path, dst_dir: Path | None = None) -> dict[str, Any]:
    src_dir, dst_dir = resolve_paths(src_dir, dst_dir)

    config_path = src_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    config_dict = load_config(config_path)
    model = instantiate_torch_model(config_dict)
    linear_weight_names = collect_linear_weight_names(model)
    state_dict, weights_path = load_torch_checkpoint(src_dir)
    validate_state_dict_keys(state_dict, model)
    converted_state_dict, transposed_names = convert_torch_state_dict_to_paddle_state_dict(
        state_dict, linear_weight_names
    )
    return save_converted_checkpoint(
        converted_state_dict=converted_state_dict,
        src_config_path=config_path,
        dst_dir=dst_dir,
        weights_path=weights_path,
        transposed_names=transposed_names,
    )


def main() -> None:
    args = parse_args()
    summary = convert_checkpoint(args.src, args.dst)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
