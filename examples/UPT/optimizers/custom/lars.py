import paddle
from paddle.optimizer import Optimizer

class LARS(Optimizer):
    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        trust_coefficient=0.001,
        eps=1e-08,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            trust_coefficient=trust_coefficient,
            eps=eps,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @paddle.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with paddle.enable_grad():
                loss = closure()
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                p_norm = paddle.norm(p.data)
                g_norm = paddle.norm(p.grad.data)
                if weight_decay != 0:
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = (
                            group["trust_coefficient"]
                            * p_norm
                            / (g_norm + p_norm * weight_decay + group["eps"])
                        )
                        grad.add_(p, alpha=weight_decay)
                        grad.mul_(lars_lr)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = paddle.clone(
                            x=grad
                        ).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                p.add_(grad, alpha=-group["lr"])
        return loss
