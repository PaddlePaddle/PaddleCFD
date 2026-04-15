from __future__ import annotations

import math
from typing import Callable

import paddle


def cosine_schedule_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
    min_lr_rate: float = 0.0,
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    factor = factor * (1.0 - min_lr_rate) + min_lr_rate
    return max(0.0, factor)


def inverse_sqrt_schedule_lambda(
    current_step: int, *, num_warmup_steps: int, timescale: int | None = None
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if timescale is None:
        timescale = num_warmup_steps or 10_000
    shift = timescale - num_warmup_steps
    return 1.0 / math.sqrt((current_step + shift) / timescale)


def warmup_stable_decay_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float,
    num_cycles: float,
) -> float:
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step < num_warmup_steps + num_stable_steps:
        return 1.0
    if current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
        progress = float(
            current_step - num_warmup_steps - num_stable_steps
        ) / float(max(1, num_decay_steps))
        value = max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )
        return (1.0 - min_lr_ratio) * value + min_lr_ratio
    return min_lr_ratio


def _build_lr_lambda(name: str, num_warmup_steps: int, num_training_steps: int, **kwargs) -> Callable[[int], float]:
    if name == "cosine":
        return lambda step: cosine_schedule_lambda(
            step,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=kwargs.get("num_cycles", 0.5),
            min_lr_rate=0.0,
        )
    if name == "cosine_with_restarts":
        return lambda step: cosine_schedule_lambda(
            step,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=kwargs.get("num_cycles", 1),
            min_lr_rate=0.0,
        )
    if name == "cosine_with_min_lr":
        if "min_lr_rate" in kwargs:
            min_lr_rate = kwargs["min_lr_rate"]
        elif "min_lr" in kwargs:
            min_lr_rate = kwargs["min_lr"] / kwargs["base_learning_rate"]
        else:
            raise ValueError("cosine_with_min_lr requires min_lr or min_lr_rate")
        return lambda step: cosine_schedule_lambda(
            step,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=kwargs.get("num_cycles", 0.5),
            min_lr_rate=min_lr_rate,
        )
    if name == "inverse_sqrt":
        return lambda step: inverse_sqrt_schedule_lambda(
            step,
            num_warmup_steps=num_warmup_steps,
            timescale=kwargs.get("timescale"),
        )
    if name == "warmup_stable_decay":
        return lambda step: warmup_stable_decay_lambda(
            step,
            num_warmup_steps=num_warmup_steps,
            num_stable_steps=kwargs["num_stable_steps"],
            num_decay_steps=kwargs["num_decay_steps"],
            min_lr_ratio=kwargs.get("min_lr_ratio", 0.0),
            num_cycles=kwargs.get("num_cycles", 0.5),
        )
    raise ValueError(f"Unsupported scheduler type: {name}")


def build_lr_scheduler(
    *,
    scheduler_type: str,
    base_learning_rate: float,
    num_warmup_steps: int,
    num_training_steps: int,
    scheduler_specific_kwargs: dict | None = None,
):
    scheduler_specific_kwargs = scheduler_specific_kwargs or {}
    lr_lambda = _build_lr_lambda(
        scheduler_type,
        num_warmup_steps,
        num_training_steps,
        base_learning_rate=base_learning_rate,
        **scheduler_specific_kwargs,
    )
    return paddle.optimizer.lr.LambdaDecay(base_learning_rate, lr_lambda)
