from collections import defaultdict

import paddle


def assign_inplace(tensor, value):
    tensor.copy_(value)
    return tensor


def scale_inplace(tensor, scalar):
    return assign_inplace(tensor, tensor * scalar)


def divide_inplace(tensor, value):
    return assign_inplace(tensor, tensor / value)


class TorchStylePaddleOptimizer(paddle.optimizer.Optimizer):
    """Minimal compatibility layer for Torch-style custom optimizers."""

    def __init__(self, params, lr, defaults):
        super().__init__(
            learning_rate=lr,
            parameters=params,
            weight_decay=defaults.get("weight_decay", 0.0),
        )
        self.defaults = defaults.copy()
        if self._param_groups and isinstance(self._param_groups[0], dict):
            self.param_groups = self._param_groups
        else:
            self.param_groups = [{"params": list(self._param_groups or [])}]
        self.state = defaultdict(dict)
        self._init_param_groups()

    def _init_param_groups(self):
        self._sync_group_lr()
        for group in self.param_groups:
            for key, value in self.defaults.items():
                if key == "lr":
                    continue
                group.setdefault(key, value)

    def _sync_group_lr(self):
        base_lr = float(self.get_lr())
        for group in self.param_groups:
            group["lr"] = base_lr * float(group.get("learning_rate", 1.0))

    @staticmethod
    def _clone_value(value):
        if isinstance(value, paddle.Tensor):
            return value.clone()
        return value

    @paddle.base.framework.dygraph_only
    def state_dict(self):
        state = {}
        for group in self.param_groups:
            for param in group["params"]:
                param_state = self.state.get(param, {})
                if not param_state:
                    continue
                state[param.name] = {
                    key: self._clone_value(value)
                    for key, value in param_state.items()
                }

        group_state = []
        for group in self.param_groups:
            group_state.append(
                {
                    key: (
                        [param.name for param in value]
                        if key == "params"
                        else self._clone_value(value)
                    )
                    for key, value in group.items()
                }
            )

        payload = {"state": state, "param_groups": group_state}
        if isinstance(self._learning_rate, paddle.optimizer.lr.LRScheduler):
            payload["LR_Scheduler"] = self._learning_rate.state_dict()
        return payload

    @paddle.base.framework.dygraph_only
    def set_state_dict(self, state_dict):
        if "state" not in state_dict or "param_groups" not in state_dict:
            raise ValueError(
                "Unsupported optimizer state format for custom Paddle optimizer."
            )

        if (
            isinstance(self._learning_rate, paddle.optimizer.lr.LRScheduler)
            and "LR_Scheduler" in state_dict
        ):
            self._learning_rate.set_state_dict(state_dict["LR_Scheduler"])

        name_to_param = {
            param.name: param
            for group in self.param_groups
            for param in group["params"]
        }
        self.state = defaultdict(dict)
        for param_name, saved_state in state_dict["state"].items():
            if param_name not in name_to_param:
                raise ValueError(
                    f"Optimizer state contains unknown parameter: {param_name}"
                )
            param = name_to_param[param_name]
            self.state[param] = {
                key: self._clone_value(value)
                for key, value in saved_state.items()
            }

        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("Optimizer param group count mismatch during restore.")

        for group, saved_group in zip(self.param_groups, saved_groups):
            current_names = [param.name for param in group["params"]]
            if current_names != saved_group["params"]:
                raise ValueError("Optimizer param group layout mismatch during restore.")
            for key, value in saved_group.items():
                if key == "params":
                    continue
                group[key] = self._clone_value(value)

        self._sync_group_lr()

    load_state_dict = set_state_dict
