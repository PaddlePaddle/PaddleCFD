import logging
from copy import deepcopy
from functools import partial

import paddle
import yaml
from initializers import initializer_from_kwargs
from utils.factory import instantiate


def model_from_kwargs(kind=None, path_provider=None, data_container=None, **kwargs):
    update_counter = kwargs.pop("update_counter", None)
    static_ctx = kwargs.pop("static_ctx", None)
    dynamic_ctx = kwargs.pop("dynamic_ctx", None)
    kwargs = deepcopy(kwargs)
    kwargs_from_yaml = kwargs.pop("kwargs", {})
    kwargs = {**kwargs_from_yaml, **kwargs}
    if "initializers" in kwargs:
        initializer_kwargs = kwargs["initializers"][0]
        assert all(
            obj.get("use_checkpoint_kwargs", None) is None
            for obj in kwargs["initializers"][1:]
        )
        use_checkpoint_kwargs = initializer_kwargs.pop("use_checkpoint_kwargs", False)
        initializer = initializer_from_kwargs(
            **initializer_kwargs, path_provider=path_provider
        )
        if use_checkpoint_kwargs:
            ckpt_kwargs = initializer.get_model_kwargs()
            if kind is None and "kind" in ckpt_kwargs:
                kind = ckpt_kwargs.pop("kind")
            else:
                ckpt_kwargs.pop("kind", None)
            ckpt_kwargs.pop("initializers", None)
            ckpt_kwargs.pop("optim_ctor", None)
            kwargs_intersection = set(kwargs.keys()).intersection(
                set(ckpt_kwargs.keys())
            )
            if len(kwargs_intersection) > 0:
                logging.info(
                    f"checkpoint_kwargs overlap with kwargs (intersection={kwargs_intersection})"
                )
                for intersecting_kwarg in kwargs_intersection:
                    ckpt_kwargs.pop(intersecting_kwarg)
            kwargs.update(ckpt_kwargs)
            if "input_shape" in kwargs and not isinstance(kwargs["input_shape"], tuple):
                kwargs["input_shape"] = tuple(kwargs["input_shape"])
            logging.info(
                f"""postprocessed checkpoint kwargs:
{yaml.safe_dump(kwargs, sort_keys=False)[:-1]}"""
            )
        else:
            logging.info(f"not loading checkpoint kwargs")
    else:
        logging.info(
            f"model has no initializers -> not loading a checkpoint or an optimizer state"
        )
    assert (
        kind is not None
    ), "model has no kind (maybe use_checkpoint_kwargs=True is missing in the initializer?)"
    optim = kwargs.pop("optim", None)
    if optim is not None:
        kwargs["optim_ctor"] = optim
    ctor_kwargs_filtered = {
        k: v for k, v in kwargs.items() if not isinstance(v, paddle.nn.Layer)
    }
    ctor_kwargs = deepcopy(ctor_kwargs_filtered)
    ctor_kwargs["kind"] = kind
    ctor_kwargs.pop("input_shape", None)
    ctor_kwargs.pop("output_shape", None)
    ctor_kwargs.pop("optim_ctor", None)
    return instantiate(
        module_names=[f"models.{kind}", f"models.composite.{kind}"],
        type_names=[kind.split(".")[-1]],
        update_counter=update_counter,
        path_provider=path_provider,
        data_container=data_container,
        static_ctx=static_ctx,
        dynamic_ctx=dynamic_ctx,
        ctor_kwargs=ctor_kwargs,
        **kwargs,
    )


def prepare_momentum_kwargs(kwargs):
    kwargs = deepcopy(kwargs)
    _prepare_momentum_kwargs(kwargs)
    return kwargs


def _prepare_momentum_kwargs(kwargs):
    if isinstance(kwargs, dict):
        kwargs.pop("optim", None)
        kwargs.pop("freezers", None)
        kwargs.pop("initializers", None)
        kwargs.pop("is_frozen", None)
        for v in kwargs.values():
            _prepare_momentum_kwargs(v)
    elif isinstance(kwargs, partial):
        kwargs.keywords.pop("optim_ctor", None)
        kwargs.keywords.pop("freezers", None)
        kwargs.keywords.pop("initializers", None)
        kwargs.keywords.pop("is_frozen", None)
        for v in kwargs.keywords.values():
            _prepare_momentum_kwargs(v)
