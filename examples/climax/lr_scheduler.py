import math
from typing import List


class LinearWarmupCosineAnnealingLR:
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between
    warmup_start_lr and base_lr followed by a cosine annealing schedule between base_lr and
    eta_min.

    Standalone scheduler that replicates the recursive semantics of torch's
    ``_LRScheduler`` (each step's value is derived from the previous step's
    ``group['lr']``). It intentionally does not inherit
    ``paddle.optimizer.lr.LRScheduler`` to avoid that base class' different
    base_lrs/last_epoch mechanics.

    Args:
        optimizer (paddle.optimizer.Optimizer): Wrapped optimizer.
        warmup_epochs (int): Maximum number of iterations for linear warmup
        max_epochs (int): Maximum number of iterations
        warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        # Paddle keeps dict-style parameter groups in `_param_groups`.
        self.param_groups = optimizer._param_groups
        self.base_lrs = [
            g.get("learning_rate", optimizer._learning_rate) for g in self.param_groups
        ]
        self.last_epoch = last_epoch
        self._last_lr = [warmup_start_lr] * len(self.base_lrs)
        self.step()  # mirror torch: constructor steps once (last_epoch -1 -> 0)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.param_groups)
            ]
        if (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.param_groups)
            ]
        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.param_groups
        ]

    def step(self) -> None:
        self.last_epoch += 1
        values = self.get_lr()
        for group, value in zip(self.param_groups, values):
            # paddle's param-group lr key is `learning_rate`; keep a torch-style
            # `lr` key too, since the recursion above reads the previous value
            # from `group['lr']` exactly like the torch scheduler does.
            group["learning_rate"] = value
            group["lr"] = value
        # A paddle optimizer does not re-read the group dict's `learning_rate`
        # at step time (it is only consumed at construction as a scale of the
        # base lr, see paddle/optimizer/optimizer.py `_add_param_group`), so
        # propagate the scheduled value through the optimizer itself.
        if len(set(values)) == 1 and isinstance(self.optimizer._learning_rate, float):
            self.optimizer.set_lr(values[0])
        self._last_lr = values

    def get_last_lr(self) -> List[float]:
        """Return the learning rate computed at the last step."""
        return self._last_lr
