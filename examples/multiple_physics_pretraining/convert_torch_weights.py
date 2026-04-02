import argparse
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml_config(yaml_path, config_name):
    with open(yaml_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if config_name not in payload:
        raise KeyError(f"Config '{config_name}' not found in {yaml_path}")
    return argparse.Namespace(**payload[config_name])


def extract_model_state_dict(payload):
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a mapping")
    if "model_state" in payload:
        return payload["model_state"]
    return payload


def strip_module_prefix(state_dict):
    keys = list(state_dict.keys())
    if keys and all(key.startswith("module.") for key in keys):
        return {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def build_default_output_path(torch_weights_path):
    source_path = Path(torch_weights_path)
    return REPO_ROOT / "models_paddle" / f"{source_path.stem}.pdparams"


def probe_paddle_load(weight_path):
    import paddle

    try:
        paddle.load(str(weight_path))
        return True, ""
    except Exception as exc:
        return False, str(exc)


def to_numpy_array(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def convert_array_for_target(source_array, target_shape, target_dtype, key):
    source_array = np.asarray(source_array)
    if tuple(source_array.shape) == tuple(target_shape):
        return source_array.astype(target_dtype, copy=False)
    if source_array.ndim == 2 and tuple(source_array.T.shape) == tuple(target_shape):
        return source_array.T.astype(target_dtype, copy=False)
    raise ValueError(
        f"shape mismatch for {key}: source {tuple(source_array.shape)} vs target {tuple(target_shape)}"
    )


def get_source_value_for_target_key(source_state_dict, target_key):
    if target_key in source_state_dict:
        return source_state_dict[target_key]
    if target_key.endswith(".scale"):
        fallback_key = f"{target_key[:-6]}.weight"
        if fallback_key in source_state_dict:
            return source_state_dict[fallback_key]
    raise KeyError(target_key)


def convert_state_dict(source_state_dict, target_metadata):
    source_keys = set(source_state_dict.keys())
    missing_keys = []
    used_source_keys = set()

    for target_key in target_metadata:
        try:
            source_value = get_source_value_for_target_key(source_state_dict, target_key)
            for candidate_key, candidate_value in source_state_dict.items():
                if candidate_value is source_value:
                    used_source_keys.add(candidate_key)
                    break
        except KeyError:
            missing_keys.append(target_key)

    extra_keys = sorted(source_keys - used_source_keys)

    if missing_keys:
        raise ValueError(f"Missing keys: {missing_keys}")
    if extra_keys:
        raise ValueError(f"Unexpected keys: {extra_keys}")

    converted = {}
    for key, meta in target_metadata.items():
        converted[key] = convert_array_for_target(
            source_array=to_numpy_array(
                get_source_value_for_target_key(source_state_dict, key)
            ),
            target_shape=meta["shape"],
            target_dtype=meta["dtype"],
            key=key,
        )
    return converted


def collect_target_metadata(state_dict):
    metadata = {}
    for key, value in state_dict.items():
        array = to_numpy_array(value)
        metadata[key] = {"shape": tuple(array.shape), "dtype": str(array.dtype)}
    return metadata


def build_paddle_model(params):
    from ppcfd.models.multiple_physics_pretraining.avit import build_avit

    return build_avit(params)


def convert_weights(yaml_config, config_name, torch_weights, output_path):
    import paddle
    import torch

    is_paddle_file, _ = probe_paddle_load(torch_weights)
    if is_paddle_file:
        raise ValueError(f"{torch_weights} is already a Paddle-serializable file")

    params = load_yaml_config(yaml_config, config_name)
    paddle_model = build_paddle_model(params)
    target_metadata = collect_target_metadata(paddle_model.state_dict())

    checkpoint = torch.load(torch_weights, map_location="cpu")
    source_state = strip_module_prefix(extract_model_state_dict(checkpoint))
    converted_state = convert_state_dict(source_state, target_metadata)
    paddle_state = {
        key: paddle.to_tensor(value, dtype=target_metadata[key]["dtype"])
        for key, value in converted_state.items()
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paddle.save(paddle_state, str(output_path))
    return output_path


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", required=True, type=Path)
    parser.add_argument("--config", default="basic_config", type=str)
    parser.add_argument("--torch_weights", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    output_path = args.output or build_default_output_path(args.torch_weights)
    converted_path = convert_weights(
        yaml_config=args.yaml_config,
        config_name=args.config,
        torch_weights=args.torch_weights,
        output_path=output_path,
    )
    print(f"Saved Paddle weights to {converted_path}")


if __name__ == "__main__":
    main()
