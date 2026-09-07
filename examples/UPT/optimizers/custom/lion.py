import paddle
from paddle.optimizer import Optimizer

class Lion(Optimizer):
    """https://raw.githubusercontent.com/lucidrains/lion-paddle/main/lion_paddle/lion_paddle.py"""

    def __init__(self, params, lr=0.0001, betas=(0.9, 0.99), weight_decay=0.0):
        assert lr > 0.0
        assert all([(0.0 <= beta <= 1.0) for beta in betas])
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @paddle.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with paddle.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in filter(lambda pp: pp.grad is not None, group["params"]):
                grad = p.grad
                lr = group["lr"]
                wd = group["weight_decay"]
                beta1, beta2 = group["betas"]
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = paddle.zeros_like(p)
                exp_avg = state["exp_avg"]
                p.data.mul_(1 - lr * wd)
                update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign()
                p.add_(update, alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss
