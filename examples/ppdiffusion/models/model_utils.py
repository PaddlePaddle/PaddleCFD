from typing import Any, Callable, Optional

import paddle.nn as nn


def default(val: Optional[Any], d: Callable[[], Any] | Any) -> Any:
    return val if val is not None else (d() if callable(d) else d)


def get_normalization_layer(name, dims, num_groups=None, *args, **kwargs):
    if not isinstance(name, str) or name.lower() == "none":
        return None
    elif "batch_norm" == name:
        return nn.BatchNorm2D(num_features=dims, *args, **kwargs)
    elif "layer_norm" == name:
        return nn.LayerNorm(dims, *args, **kwargs)
    elif "instance" in name:
        return nn.InstanceNorm1D(num_features=dims, *args, **kwargs)
    elif "group" in name:
        if num_groups is None:
            pos_groups = [int(dims / N) for N in range(2, 17) if dims % N == 0]
            if len(pos_groups) == 0:
                raise NotImplementedError(f"Group norm could not infer the number of groups for dim={dims}")
            num_groups = max(pos_groups)
        return nn.GroupNorm(num_groups=num_groups, num_channels=dims)
    else:
        raise ValueError("Unknown normalization name", name)
