from copy import deepcopy
from functools import partial

from optimizers.interleaved_optimizer import InterleavedOptimizer
from optimizers.optimizer_wrapper import OptimizerWrapper
from utils.factory import get_ctor


def optim_ctor_from_kwargs(kind, **kwargs):
    kwargs = deepcopy(kwargs)
    if kind == "interleaved_optimizer":
        optim_ctors = [
            optim_ctor_from_kwargs(**optim) for optim in kwargs.pop("optims")
        ]
        return partial(InterleavedOptimizer, optim_ctors=optim_ctors, **kwargs)
    wrapped_optim_kwargs = {}
    wrapped_optim_kwargs_keys = [
        "schedule",
        "weight_decay_schedule",
        "clip_grad_value",
        "clip_grad_norm",
        "exclude_bias_from_wd",
        "exclude_norm_from_wd",
        "param_group_modifiers",
        "lr_scaler",
    ]
    for key in wrapped_optim_kwargs_keys:
        if key in kwargs:
            wrapped_optim_kwargs[key] = kwargs.pop(key)
    paddle_optim_ctor = get_ctor(
        module_names=["paddle.optimizer", f"optimizers.custom.{kind}"],
        type_names=["AdamW"],
        **kwargs,
    )
    return partial(
        _optimizer_wrapper_ctor,
        paddle_optim_ctor=paddle_optim_ctor,
        **wrapped_optim_kwargs,
    )


def _optimizer_wrapper_ctor(model, paddle_optim_ctor, **wrapped_optim_kwargs):
    return OptimizerWrapper(
        model=model, paddle_optim_ctor=paddle_optim_ctor, **wrapped_optim_kwargs
    )
