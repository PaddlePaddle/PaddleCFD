import inspect
import math
import re

import paddle


class Adam(paddle.optimizer.Optimizer):
    """
    Converted to paddle from PyTorch's Adam optimizer.
    Same as https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py,
    without amsgrad, with step in a tensor, and states initialization in __init__.
    It was important to add `.item()` in `state['step'].item()`.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        # 保存原始参数列表和超参数 (PaddlePaddle不使用param_groups字典结构)
        self._params_list = list(params)
        self.betas = betas
        self.eps = eps

        # 使用命名参数调用 PaddlePaddle 父类
        super().__init__(
            learning_rate=lr,
            parameters=self._params_list,
            weight_decay=weight_decay if weight_decay != 0 else None
        )

        # PaddlePaddle不会自动初始化state,需要手动创建
        self.state = {}

        # 使用保存的参数列表初始化状态
        for p in self._params_list:
            self.state[p] = {
                "step": 0,
                "exp_avg": paddle.zeros_like(p),
                "exp_avg_sq": paddle.zeros_like(p)
            }

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        """
        Step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # 获取当前学习率
        lr = self.get_lr()

        # PaddlePaddle要求优化器更新在no_grad上下文中执行
        with paddle.no_grad():
            # 直接遍历参数列表
            for p in self._params_list:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse():
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = self.betas  # 使用实例变量

                state["step"] += 1

                # 指数移动平均更新 (PaddlePaddle API)
                exp_avg.scale_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.scale_(beta2).add_(grad.multiply(grad), alpha=1 - beta2)

                denom = exp_avg_sq.sqrt() + self.eps  # PaddlePaddle不支持add_(scalar)

                # 偏差修正
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # 权重衰减
                if hasattr(self, '_weight_decay') and self._weight_decay is not None and self._weight_decay != 0:
                    p.add_(p, alpha=-self._weight_decay * lr)

                # 参数更新
                p.add_(exp_avg.divide(denom), alpha=-step_size)

        return loss


class AdamWithWarmup(Adam):
    """
    Adam with a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`).
    During warmup:
        lrs = paddle.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = lr
    """

    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        warmup_updates=10000,
        warmup_init_lr=1e-07,
    ):
        # 调用父类 Adam 的 __init__，使用命名参数
        super().__init__(
            params=params,
            lr=warmup_init_lr,  # 使用初始学习率
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        self.warmup_end_lr = lr
        self.lr_step = (lr - warmup_init_lr) / warmup_updates
        self.num_updates = 0  # 使用单一更新计数器

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.warmup_end_lr

    def step(self, closure=None):
        super().step(closure)
        self.num_updates += 1
        # 更新学习率
        self._learning_rate = self.get_lr_for_step(self.num_updates)


class AdamInverseSqrtWithWarmup(Adam):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup:
        lrs = paddle.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
        decay_factor = lr * sqrt(warmup_updates)
    """

    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        warmup_updates=10000,
        warmup_init_lr=1e-07,
        exp_factor=0.5,
    ):
        # 调用父类 Adam 的 __init__，使用命名参数
        super().__init__(
            params=params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates
        self.exp_factor = exp_factor
        self.decay_factor = warmup_end_lr * warmup_updates**self.exp_factor
        self.num_updates = 0  # 使用单一更新计数器

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.decay_factor * num_updates**-self.exp_factor

    def step(self, closure=None):
        super().step(closure)
        self.num_updates += 1
        # 更新学习率
        self._learning_rate = self.get_lr_for_step(self.num_updates)


class AdamCosineWithWarmup(Adam):
    """
    Assign LR based on a cyclical schedule that follows the cosine function.
    See https://arxiv.org/pdf/1608.03983.pdf for details.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (``--warmup-init-lr``) until the configured
    learning rate (``--lr``).
    During warmup::
      lrs = paddle.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]
    After warmup::
      lr = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(t_curr / t_i))
    where ``t_curr`` is current percentage of updates within the current period
    range and ``t_i`` is the current period range, which is scaled by ``t_mul``
    after every iteration.
    """

    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        warmup_updates=10000,
        warmup_init_lr=1e-07,
        min_lr=1e-09,
        init_period=100000000,
        period_mult=1,
        lr_shrink=0.75,
        smooth=False,
    ):
        # 调用父类 Adam 的 __init__，使用命名参数
        super().__init__(
            params=params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        self.smooth = smooth
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates
        self.min_lr = min_lr
        self.max_lr = lr
        self.period = init_period
        self.period_mult = period_mult
        self.lr_shrink = lr_shrink
        assert not self.smooth or self.period_mult == 1
        self.num_updates = 0  # 使用单一更新计数器

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            t = num_updates - self.warmup_updates
            if self.period_mult == 1:
                if self.smooth:
                    pid = math.floor(t / self.period - 1 / 2)
                else:
                    pid = math.floor(t / self.period)
                t_i = self.period
                t_curr = t - self.period * pid
            else:
                pid = math.floor(
                    math.log(
                        1 - t / self.period * (1 - self.period_mult), self.period_mult
                    )
                )
                t_i = self.period * self.period_mult**pid
                t_curr = (
                    t
                    - (1 - self.period_mult**pid)
                    / (1 - self.period_mult)
                    * self.period
                )
            lr_shrink = self.lr_shrink**pid
            min_lr = self.min_lr * lr_shrink
            max_lr = self.max_lr * lr_shrink
            if self.smooth:
                return min_lr + 0.5 * (max_lr - min_lr) * (
                    1 + math.cos(2 * math.pi * t_curr / t_i)
                )
            else:
                return min_lr + 0.5 * (max_lr - min_lr) * (
                    1 + math.cos(math.pi * t_curr / t_i)
                )

    def step(self, closure=None):
        super().step(closure)
        self.num_updates += 1
        # 更新学习率
        self._learning_rate = self.get_lr_for_step(self.num_updates)


def get_optimizer(parameters, lr, s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[: s.find(",")]
        optim_params = {}
        for x in s[s.find(",") + 1 :].split(","):
            split = x.split("=")
            assert len(split) == 2
            assert re.match("^[+-]?(\\d+(\\.\\d*)?|\\.\\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}
    if method == "adadelta":
        optim_fn = paddle.optimizer.Adadelta
    elif method == "adagrad":
        optim_fn = paddle.optimizer.Adagrad
    elif method == "adam":
        optim_fn = Adam
        optim_params["betas"] = optim_params.get("beta1", 0.9), optim_params.get(
            "beta2", 0.999
        )
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adam_warmup":
        optim_fn = AdamWithWarmup
        optim_params["betas"] = optim_params.get("beta1", 0.9), optim_params.get(
            "beta2", 0.999
        )
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adam_inverse_sqrt":
        optim_fn = AdamInverseSqrtWithWarmup
        optim_params["betas"] = optim_params.get("beta1", 0.9), optim_params.get(
            "beta2", 0.999
        )
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adam_cosine":
        optim_fn = AdamCosineWithWarmup
        optim_params["smooth"] = False
        optim_params["betas"] = optim_params.get("beta1", 0.9), optim_params.get(
            "beta2", 0.999
        )
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adam_smooth_cosine":
        optim_fn = AdamCosineWithWarmup
        optim_params["smooth"] = True
        optim_params["betas"] = optim_params.get("beta1", 0.9), optim_params.get(
            "beta2", 0.999
        )
        optim_params.pop("beta1", None)
        optim_params.pop("beta2", None)
    elif method == "adamax":
        optim_fn = paddle.optimizer.Adamax
    elif method == "asgd":
        optim_fn = paddle.optimizer.ASGD
    elif method == "rmsprop":
        optim_fn = paddle.optimizer.RMSProp
    elif method == "rprop":
        optim_fn = paddle.optimizer.Rprop
    elif method == "sgd":
        optim_fn = paddle.optimizer.SGD
        assert "lr" in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ["self", "params"]
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception(
            'Unexpected parameters: expected "%s", got "%s"'
            % (str(expected_args[2:]), str(optim_params.keys()))
        )
    return optim_fn(parameters, lr=lr, **optim_params)
