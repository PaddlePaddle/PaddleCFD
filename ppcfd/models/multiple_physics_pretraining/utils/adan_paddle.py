import math

import paddle

from .custom_optimizer_base import (
    TorchStylePaddleOptimizer,
    divide_inplace,
    scale_inplace,
)


def exists(val):
    return val is not None


class Adan(TorchStylePaddleOptimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.02, 0.08, 0.01),
        eps=1e-08,
        weight_decay=0,
        restart_cond: callable = None,
    ):
        assert len(betas) == 3
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            restart_cond=restart_cond,
        )
        super().__init__(params, lr, defaults)

    def step(self, closure=None):
        self._sync_group_lr()
        loss = None
        if exists(closure):
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            restart_cond = group["restart_cond"]
            for p in group["params"]:
                if not exists(p.grad):
                    continue
                data, grad = p.data, p.grad.data
                assert not grad.is_sparse()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["prev_grad"] = paddle.zeros_like(grad)
                    state["m"] = paddle.zeros_like(grad)
                    state["v"] = paddle.zeros_like(grad)
                    state["n"] = paddle.zeros_like(grad)
                step, m, v, n, prev_grad = (
                    state["step"],
                    state["m"],
                    state["v"],
                    state["n"],
                    state["prev_grad"],
                )
                if step > 0:
                    prev_grad = state["prev_grad"]
                    scale_inplace(m, 1 - beta1).add_(grad, alpha=beta1)
                    grad_diff = grad - prev_grad
                    scale_inplace(v, 1 - beta2).add_(grad_diff, alpha=beta2)
                    next_n = (grad + (1 - beta2) * grad_diff) ** 2
                    scale_inplace(n, 1 - beta3).add_(next_n, alpha=beta3)
                step += 1
                correct_m, correct_v, correct_n = map(
                    lambda n: 1 / (1 - (1 - n) ** step), (beta1, beta2, beta3)
                )

                def grad_step_(data, m, v, n):
                    weighted_step_size = lr / ((n * correct_n).sqrt() + eps)
                    denom = 1 + weight_decay * lr
                    update = weighted_step_size * (
                        m * correct_m + (1 - beta2) * v * correct_v
                    )
                    data.add_(update, alpha=-1.0)
                    divide_inplace(data, denom)

                grad_step_(data, m, v, n)
                if exists(restart_cond) and restart_cond(state):
                    m.data.copy_(grad)
                    v.zero_()
                    n.data.copy_(grad**2)
                    grad_step_(data, m, v, n)
                prev_grad.copy_(grad)
                state["step"] = step
        return loss
