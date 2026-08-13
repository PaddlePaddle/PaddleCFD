from logging import getLogger

import os

import paddle
from tabulate import tabulate

from ppcfd.models.prose_fd.paddle_utils import *
from ppcfd.models.prose_fd.runtime import get_runtime_device
from ppcfd.models.prose_fd.transformer_wrappers import PROSE_1to1, PROSE_2to1

logger = getLogger()

_UNPORTED_BASELINE_MODELS = frozenset({"fno", "vit", "unet", "deeponet"})
_UNPORTED_BASELINE_ERROR = (
    "Paddle side currently only supports prose_1to1 / prose_2to1. "
    "Baseline models are not ported to avoid torch/neuralop dependencies."
)


def build_model(params, model_config, data_config, symbol_env):
    modules = {}
    name = model_config.name
    if name == "prose_1to1":
        modules["model"] = PROSE_1to1(
            model_config,
            symbol_env,
            data_config.x_num,
            data_config.max_output_dimension,
            data_config.t_num - params.input_len,
        )
    elif name == "prose_2to1":
        modules["model"] = PROSE_2to1(
            model_config,
            symbol_env,
            data_config.x_num,
            data_config.max_output_dimension,
            data_config.t_num - params.input_len,
        )
    elif name in _UNPORTED_BASELINE_MODELS:
        raise NotImplementedError(_UNPORTED_BASELINE_ERROR)
    else:
        raise NotImplementedError(
            "Paddle side currently only supports prose_1to1 / prose_2to1."
        )
    if params.reload_model:
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = paddle.load(path=str(params.reload_model))
        for k, v in modules.items():
            assert k in reloaded, f"{k} not in save"
            if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {
                    k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()
                }
            if all([k2.startswith("_orig_mod.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {
                    k2[len("_orig_mod.") :]: v2 for k2, v2 in reloaded[k].items()
                }
            v.load_state_dict(reloaded[k])
    for k, v in modules.items():
        logger.info(f"{k}: {v}")
    for k, v in modules.items():
        s = f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if not p.stop_gradient]):,}"
        if hasattr(v, "summary"):
            s += v.summary()
        logger.info(s)
    if not params.cpu:
        for v in modules.values():
            v.to(get_runtime_device())
    if os.environ.get("PROSE_TO_STATIC", "0") == "1":
        from paddle.jit import to_static

        logger.info("Wrapping model with paddle.jit.to_static (CINN) ...")
        for k in list(modules.keys()):
            modules[k] = to_static(modules[k], full_graph=True)
    return modules
