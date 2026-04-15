from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import paddle
import torch
from omegaconf import OmegaConf

from ppcfd.models.prose_fd import PROSE_2to1, SymbolicEnvironment


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]


class Params:
    cpu = 1
    compile = 0
    input_len = 10
    reload_model = None


def clean_torch_key(key: str) -> str:
    if key.startswith("module."):
        key = key[len("module."):]
    if key.startswith("_orig_mod."):
        key = key[len("_orig_mod."):]
    # torch decoder uses nn.Sequential (.layers.), paddle uses ModuleList (direct index)
    key = key.replace(".transformer_decoder.layers.", ".transformer_decoder.")
    # torch stores final norm inside transformer_decoder, paddle as decoder_norm
    if key == "data_decoder.transformer_decoder.norm.weight":
        key = "data_decoder.decoder_norm.weight"
    return key


def maybe_convert_array(name: str, array: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if tuple(array.shape) == tuple(target_shape):
        return array
    if array.ndim == 2 and tuple(array.T.shape) == tuple(target_shape):
        return array.T
    raise ValueError(f"Shape mismatch for {name}: src={array.shape}, dst={target_shape}")


def build_paddle_model():
    model_cfg = OmegaConf.load(EXAMPLE_ROOT / "configs" / "model" / "prose_2to1.yaml")
    data_cfg = OmegaConf.load(EXAMPLE_ROOT / "configs" / "data" / "fluids.yaml")
    symbol_cfg = OmegaConf.load(EXAMPLE_ROOT / "configs" / "symbol" / "symbol.yaml")
    OmegaConf.resolve(model_cfg)
    OmegaConf.resolve(data_cfg)
    OmegaConf.resolve(symbol_cfg)
    symbol_env = SymbolicEnvironment(symbol_cfg)
    model = PROSE_2to1(model_cfg, symbol_env, data_cfg.x_num, data_cfg.max_output_dimension, data_cfg.t_num - 10)
    return model


def convert_checkpoint(torch_ckpt: Path, paddle_ckpt: Path):
    torch_payload = torch.load(torch_ckpt, map_location="cpu")
    torch_state = {clean_torch_key(k): v.detach().cpu().numpy() for k, v in torch_payload["model"].items()}
    paddle_model = build_paddle_model()
    paddle_state = paddle_model.state_dict()
    converted = {}
    missing = []
    for name, tensor in paddle_state.items():
        if name not in torch_state:
            missing.append(name)
            continue
        converted[name] = paddle.to_tensor(maybe_convert_array(name, torch_state[name], tuple(tensor.shape)))
    if missing:
        raise KeyError(f"Missing keys: {missing}")
    paddle.save({"model": converted}, str(paddle_ckpt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-ckpt", type=Path, default=EXAMPLE_ROOT / "models" / "prose_fd_torch" / "prose_fd.pth")
    parser.add_argument("--paddle-ckpt", type=Path, default=EXAMPLE_ROOT / "models" / "prose_fd_paddle" / "prose_fd_converted.pdparams")
    args = parser.parse_args()
    args.paddle_ckpt.parent.mkdir(parents=True, exist_ok=True)
    convert_checkpoint(args.torch_ckpt, args.paddle_ckpt)
    print(args.paddle_ckpt)


if __name__ == "__main__":
    main()
