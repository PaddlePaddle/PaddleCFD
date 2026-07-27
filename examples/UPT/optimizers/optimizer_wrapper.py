# import logging
# import paddle
# from kappaschedules import object_to_schedule
# from utils.bidict import Bidict
# from utils.factory import create, create_collection
# from utils.formatting_util import float_to_scientific_notation

# from .lr_scalers import lr_scaler_from_kwargs
# from .lr_scalers.linear_lr_scaler import LinearLrScaler
# from .param_group_modifiers import param_group_modifier_from_kwargs


# class OptimizerWrapper:
#     """
#     PaddlePaddle 版 OptimizerWrapper
#     处理：
#     - 学习率随 Batch Size 自动缩放
#     - 参数分组（排除 bias/norm 的权重衰减，层级学习率缩放）
#     - 无状态学习率/权重衰减调度 (Stateless Scheduling)
#     - 梯度裁剪 (声明式绑定)
#     """

#     def __init__(
#         self,
#         model,
#         paddle_optim_ctor, # 实际上是传入的优化器构造器（如 partial(paddle.optimizer.AdamW, ...)）
#         schedule=None,
#         weight_decay_schedule=None,
#         clip_grad_value=None,
#         clip_grad_norm=None,
#         param_group_modifiers=None,
#         exclude_bias_from_wd=True,
#         exclude_norm_from_wd=True,
#         add_model_specific_param_group_modifiers=True,
#         update_counter=None,
#         lr_scale_factor=None,
#         lr_scaler=None,
#     ):
#         self.logger = logging.getLogger(type(self).__name__)
#         self.model = model
#         self.update_counter = update_counter
#         self.clip_grad_value = clip_grad_value
#         self.clip_grad_norm = clip_grad_norm
        
#         # 1. 学习率缩放
#         lr_scaler = create(lr_scaler, lr_scaler_from_kwargs) or LinearLrScaler()
#         # 兼容 Paddle 的 "lr" 键名
#         base_lr = paddle_optim_ctor.keywords.get("lr", paddle_optim_ctor.keywords.get("learning_rate"))
#         lr_scale_factor = lr_scale_factor or update_counter.effective_batch_size
#         scaled_lr = lr_scaler.scale_lr(base_lr=base_lr, lr_scale_factor=lr_scale_factor)
        
#         self.logger.info(f"base lr: {float_to_scientific_notation(base_lr, max_precision=2)}")
#         self.logger.info(f"scaled lr: {float_to_scientific_notation(scaled_lr, max_precision=2)}")
#         self.logger.info(f"lr_scaler={lr_scaler}")
#         self.logger.info(f"lr_scale_factor={lr_scale_factor}")
#         paddle_optim_ctor.keywords["lr"] = scaled_lr
        
#         # 2. 准备参数组修改器
#         param_group_modifiers = create_collection(
#             param_group_modifiers, param_group_modifier_from_kwargs
#         )
#         if add_model_specific_param_group_modifiers:
#             param_group_modifiers = (
#                 model.get_model_specific_param_group_modifiers() + param_group_modifiers
#             )

#         # 3. 扫描参数并分组
#         param_groups = []
#         self.logger.info(
#             f"group modifiers exclude_bias_from_wd={exclude_bias_from_wd} exclude_norm_from_wd={exclude_norm_from_wd} "
#             f"add_model_specific_param_group_modifiers={add_model_specific_param_group_modifiers} "
#             f"[{' '.join(str(pgm) for pgm in param_group_modifiers)}]"
#         )
#         for name, param in model.named_parameters():
#             if param.stop_gradient:
#                 continue
#             properties = {}
#             if name.endswith(".bias") and exclude_bias_from_wd:
#                 properties["weight_decay"] = 0.0
#             elif param.ndim <= 1 and exclude_norm_from_wd:
#                 properties["weight_decay"] = 0.0
            
#             for param_group_modifier in param_group_modifiers:
#                 for key, value in param_group_modifier.get_properties(model, name, param).items():
#                     if key in properties and key == "lr_scale":
#                         properties[key] *= value
#                     else:
#                         properties[key] = value
            
#             properties["name"] = name
#             properties["params"] = [param]
#             param_groups.append(properties)

#         # 4. 合并参数组以优化性能
#         merged_groups = []
#         merged_groups_properties = []
#         merged_groups_paramnames = []
#         for param_group in param_groups:
#             param_name = param_group.pop("name")
#             properties = {k: v for k, v in param_group.items() if k != "params"}
#             matching_group_idx = None
#             for i, merged_group_properties in enumerate(merged_groups_properties):
#                 if properties == merged_group_properties:
#                     matching_group_idx = i
#                     break
#             if matching_group_idx is None:
#                 merged_groups.append(param_group)
#                 merged_groups_properties.append(properties)
#                 merged_groups_paramnames.append([param_name])
#             else:
#                 merged_groups[matching_group_idx]["params"] += param_group["params"]
#                 merged_groups_paramnames[matching_group_idx].append(param_name)

#         # 5. 构造 Paddle 样式的参数组
#         final_paddle_groups = []
#         for i, group in enumerate(merged_groups):
#             # Paddle 的参数组字典，不支持 lr_scale 等自定义键，必须手动应用
#             lr_scale = group.get("lr_scale", 1.0)
#             p_group = {
#                 "params": group["params"],
#                 "learning_rate": scaled_lr * lr_scale,
#                 "weight_decay": group.get("weight_decay", paddle_optim_ctor.keywords.get("weight_decay", 0.01)),
#                 # 我们在 group 里保留自定义信息，方便 schedule 访问
#                 "lr_scale": lr_scale,
#                 "exclude_from_wd": group.get("weight_decay") == 0.0
#             }
#             final_paddle_groups.append(p_group)

#         # 6. 梯度裁剪 (Paddle 是在优化器构造时绑定的)
#         grad_clip = None
#         if self.clip_grad_norm is not None:
#             grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=self.clip_grad_norm)
#         elif self.clip_grad_value is not None:
#             grad_clip = paddle.nn.ClipGradByValue(min=-self.clip_grad_value, max=self.clip_grad_value)

#         # 7. 实例化 Paddle 优化器
#         # 提取超参数并处理参数名差异 (AdamW: eps -> epsilon, betas -> beta1/beta2)
#         betas = paddle_optim_ctor.keywords.get("betas", (0.9, 0.999))
#         eps = paddle_optim_ctor.keywords.get("eps", 1e-8)
        
#         self.paddle_optim = paddle.optimizer.AdamW(
#             learning_rate=scaled_lr,
#             parameters=final_paddle_groups,
#             beta1=betas[0],
#             beta2=betas[1],
#             epsilon=eps,
#             grad_clip=grad_clip,
#             weight_decay=paddle_optim_ctor.keywords.get("weight_decay", 0.01)
#         )
        
#         # 将 paddle_optim 指向 paddle_optim 以保持引用兼容性
#         self.paddle_optim = self.paddle_optim

#         # 8. 映射管理
#         self.param_idx_to_name = Bidict()
#         # 注意：Paddle 优化器内部存储参数的方式与 Paddle 不同，这里通过模型参数列表构建映射
#         for idx, (name, _) in enumerate(model.named_parameters()):
#             self.param_idx_to_name.set_forward(idx, name)

#         # 9. Schedule 初始化
#         self.schedule = object_to_schedule(
#             schedule,
#             batch_size=self.update_counter.effective_batch_size if self.update_counter else None,
#             updates_per_epoch=self.update_counter.updates_per_epoch if self.update_counter else None,
#             max_value=scaled_lr,
#         )
#         self.weight_decay_schedule = object_to_schedule(
#             weight_decay_schedule,
#             batch_size=self.update_counter.effective_batch_size if self.update_counter else None,
#             updates_per_epoch=self.update_counter.updates_per_epoch if self.update_counter else None,
#             max_value=paddle_optim_ctor.keywords.get("weight_decay", 0.01),
#         )

#     def _has_param_with_grad(self):
#         for param in self.model.parameters():
#             if param.grad is not None:
#                 return True
#         return False

#     def step(self, grad_scaler=None):
#         """ 执行优化步 """
#         if grad_scaler is not None:
#             # 使用混合精度训练
#             if not self._has_param_with_grad():
#                 return
#             grad_scaler.step(self.paddle_optim)
#             grad_scaler.update()
#         else:
#             # 普通训练
#             self.paddle_optim.step()

#     def schedule_step(self):
#         """ 更新学习率和权重衰减的 Schedule """
#         if self.schedule is not None:
#             base_lr = self.schedule.get_value(
#                 step=self.update_counter.cur_checkpoint.update,
#                 total_steps=self.update_counter.end_checkpoint.update,
#             )
#             # 更新每个参数组的学习率
#             for group in self.paddle_optim._param_groups:
#                 # Paddle 内部字典键名为 learning_rate
#                 lr_scale = group.get("lr_scale", 1.0)
#                 group["learning_rate"] = base_lr * lr_scale
                
#         if self.weight_decay_schedule is not None:
#             wd_val = self.weight_decay_schedule.get_value(
#                 step=self.update_counter.cur_checkpoint.update,
#                 total_steps=self.update_counter.end_checkpoint.update,
#             )
#             for group in self.paddle_optim._param_groups:
#                 if not group.get("exclude_from_wd", False):
#                     group["weight_decay"] = wd_val

#     def zero_grad(self, set_to_none=True):
#         # Paddle 对应的 API 是 clear_grad
#         self.paddle_optim.clear_grad()

#     def state_dict(self):
#         sd = self.paddle_optim.state_dict()
#         sd["param_idx_to_name"] = self.param_idx_to_name.to_forward()
#         return sd

#     def load_state_dict(self, state_dict_to_load):
#         # Paddle 的加载逻辑处理
#         if "param_idx_to_name" in state_dict_to_load:
#             state_dict_to_load.pop("param_idx_to_name")
#         self.paddle_optim.set_state_dict(state_dict_to_load)














import logging

import paddle
from kappaschedules import object_to_schedule
from utils.bidict import Bidict
from utils.factory import create, create_collection
from utils.formatting_util import float_to_scientific_notation

from .lr_scalers import lr_scaler_from_kwargs
from .lr_scalers.linear_lr_scaler import LinearLrScaler
from .param_group_modifiers import param_group_modifier_from_kwargs


class OptimizerWrapper:
    """
    wrapper for paddle optimizers that also handles
    - learning rate scaling (with batchsize)
    - creating parameter groups (e.g. excluding bias/norm from weight decay, layerwise lr scaling)
    - stateless learning rate scheduling
    - gradient clipping
    """

    def __init__(
        self,
        model,
        paddle_optim_ctor,
        schedule=None,
        weight_decay_schedule=None,
        clip_grad_value=None,
        clip_grad_norm=None,
        param_group_modifiers=None,
        exclude_bias_from_wd=True,
        exclude_norm_from_wd=True,
        add_model_specific_param_group_modifiers=True,
        update_counter=None,
        lr_scale_factor=None,
        lr_scaler=None,
    ):
        self.logger = logging.getLogger(type(self).__name__)
        self.model = model
        self.update_counter = update_counter
        self.clip_grad_value = clip_grad_value
        self.clip_grad_norm = clip_grad_norm
        
        # assert self.clip_grad_value is None or self.clip_grad_value > 0
        # assert self.clip_grad_norm is None or self.clip_grad_norm > 0
        # assert "lr" in paddle_optim_ctor.keywords
        
        lr_scaler = create(lr_scaler, lr_scaler_from_kwargs) or LinearLrScaler()
        base_lr = paddle_optim_ctor.keywords.get("lr", paddle_optim_ctor.keywords.get("learning_rate"))
        lr_scale_factor = lr_scale_factor or update_counter.effective_batch_size
        scaled_lr = lr_scaler.scale_lr(base_lr=base_lr, lr_scale_factor=lr_scale_factor)
        self.logger.info(
            f"base lr: {float_to_scientific_notation(base_lr, max_precision=2)}"
        )
        self.logger.info(
            f"scaled lr: {float_to_scientific_notation(scaled_lr, max_precision=2)}"
        )
        self.logger.info(f"lr_scaler={lr_scaler}")
        self.logger.info(f"lr_scale_factor={lr_scale_factor}")
        paddle_optim_ctor.keywords["learning_rate"] = scaled_lr
        paddle_optim_ctor.keywords.pop("lr", None)
        param_group_modifiers = create_collection(
            param_group_modifiers, param_group_modifier_from_kwargs
        )
        if add_model_specific_param_group_modifiers:
            param_group_modifiers = (
                model.get_model_specific_param_group_modifiers() + param_group_modifiers
            )
        param_groups = []
        # self.logger.info(
        #     f"group modifiers exclude_bias_from_wd={exclude_bias_from_wd} exclude_norm_from_wd={exclude_norm_from_wd} add_model_specific_param_group_modifiers={add_model_specific_param_group_modifiers} [{' '.join(str(pgm) for pgm in param_group_modifiers)}]"
        # )
        for name, param in model.named_parameters():
            if param.stop_gradient:
                continue
            properties = {}
            if name.endswith(".bias") and exclude_bias_from_wd:
                properties["weight_decay"] = 0.0
            elif param.ndim <= 1 and exclude_norm_from_wd:
                properties["weight_decay"] = 0.0
            for param_group_modifier in param_group_modifiers:
                for key, value in param_group_modifier.get_properties(
                    model, name, param
                ).items():
                    if key in properties and key == "lr_scale":
                        properties[key] *= value
                    else:
                        # assert key not in properties
                        properties[key] = value
            # assert "param" not in properties
            # assert "name" not in properties
            properties["name"] = name
            properties["params"] = [param]
            param_groups.append(properties)
        # for param_group_modifier in param_group_modifiers:
        #     assert (
        #         param_group_modifier.was_applied_successfully()
        #     ), f"{param_group_modifier} failed"
        merged_groups = []
        merged_groups_properties = []
        merged_groups_paramnames = []
        for param_group in param_groups:
            param_name = param_group.pop("name")
            properties = {k: v for k, v in param_group.items() if k != "params"}
            matching_group_idx = None
            for i, merged_group_properties in enumerate(merged_groups_properties):
                if properties == merged_group_properties:
                    matching_group_idx = i
                    break
            if matching_group_idx is None:
                merged_groups.append(param_group)
                merged_groups_properties.append(properties)
                merged_groups_paramnames.append([param_name])
            else:
                merged_groups[matching_group_idx]["params"] += param_group["params"]
                merged_groups_paramnames[matching_group_idx].append(param_name)
        for param_group in merged_groups:
            names = []
            for key, value in param_group.items():
                if key == "params":
                    continue
                if isinstance(value, float):
                    value_str = float_to_scientific_notation(
                        value, max_precision=1, remove_plus=True
                    )
                else:
                    raise NotImplementedError
                names.append(f"{key}={value_str}")
            if len(names) > 0:
                param_group["name"] = "&".join(names)
        self.logger.info(f"using {len(merged_groups)} param groups:")
        for param_group in merged_groups:
            self.logger.info(
                " ".join(
                    [
                        f"{key}={value}"
                        for key, value in param_group.items()
                        if key not in ["params", "name"]
                    ]
                    + [f"len(params)={len(param_group['params'])}"]
                )
            )
        self.param_idx_to_name = Bidict()
        idx = 0
        for group_paramnames in merged_groups_paramnames:
            for param_name in group_paramnames:
                self.param_idx_to_name.set_forward(idx, param_name)
                idx += 1
        # honor optimizer ctor + kwargs from yaml (lr, weight_decay, betas, eps, ...)
        if "betas" in paddle_optim_ctor.keywords:
            beta1, beta2 = paddle_optim_ctor.keywords.pop("betas")
            paddle_optim_ctor.keywords.setdefault("beta1", beta1)
            paddle_optim_ctor.keywords.setdefault("beta2", beta2)
        if "eps" in paddle_optim_ctor.keywords:
            paddle_optim_ctor.keywords.setdefault("epsilon", paddle_optim_ctor.keywords.pop("eps"))
        self.paddle_optim = paddle_optim_ctor(parameters=merged_groups)
        self.all_parameters = None
        if self.clip_grad_value is not None or self.clip_grad_norm is not None:
            self.all_parameters = list(model.parameters())
        for param_group in self.paddle_optim._param_groups:
            if "lr_scale" in param_group:
                assert "original_lr" not in param_group
                param_group["original_lr"] = param_group["lr"]
                param_group["lr"] *= param_group["lr_scale"]
                self.logger.info(
                    f"scaled lr of param_group '{param_group['name']}' from {float_to_scientific_notation(param_group['original_lr'], max_precision=2)} to {float_to_scientific_notation(param_group['lr'], max_precision=2)}"
                )
        self.schedule = object_to_schedule(
            schedule,
            batch_size=self.update_counter.effective_batch_size
            if self.update_counter is not None
            else None,
            updates_per_epoch=self.update_counter.updates_per_epoch
            if self.update_counter is not None
            else None,
            max_value=scaled_lr,
        )
        self.weight_decay_schedule = object_to_schedule(
            weight_decay_schedule,
            batch_size=self.update_counter.effective_batch_size
            if self.update_counter is not None
            else None,
            updates_per_epoch=self.update_counter.updates_per_epoch
            if self.update_counter is not None
            else None,
            max_value=self.paddle_optim._default_dict.get("weight_decay", 0.01),
        )
        if self.weight_decay_schedule is not None:
            for param_group in self.paddle_optim._param_groups:
                assert "exclude_from_wd" not in param_group
                param_group["exclude_from_wd"] = param_group["weight_decay"] == 0.0

    def _has_param_with_grad(self):
        for param_group in self.paddle_optim._param_groups:
            for p in param_group["params"]:
                if p.grad is not None:
                    return True
        return False

    def step(self, grad_scaler):
        if isinstance(grad_scaler, paddle.amp.GradScaler):
            if not self._has_param_with_grad():
                return
        if self.clip_grad_value is not None or self.clip_grad_norm is not None:
            grad_scaler.unscale_(self.paddle_optim)
        if self.clip_grad_value is not None:
            paddle.nn.utils.clip_grad_value_(
                parameters=self.all_parameters, clip_value=self.clip_grad_value
            )
        if self.clip_grad_norm is not None:
            paddle.nn.utils.clip_grad_norm_(
                parameters=self.all_parameters, max_norm=self.clip_grad_norm
            )
        grad_scaler.step(self.paddle_optim)
        grad_scaler.update()

    def schedule_step(self):
        if self.schedule is not None:
            lr_scale = self.schedule.get_value(
                step=self.update_counter.cur_checkpoint.update,
                total_steps=self.update_counter.end_checkpoint.update,
            )
            for param_group in self.paddle_optim._param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = param_group["lr_scale"] * lr_scale
                else:
                    param_group["lr"] = lr_scale
        if self.weight_decay_schedule is not None:
            wd_scale = self.weight_decay_schedule.get_value(
                step=self.update_counter.cur_checkpoint.update,
                total_steps=self.update_counter.end_checkpoint.update,
            )
            for param_group in self.paddle_optim._param_groups:
                if not param_group["exclude_from_wd"]:
                    param_group["weight_decay"] = wd_scale

    def zero_grad(self, set_to_none=True):
        self.paddle_optim.clear_grad(set_to_none)

    def state_dict(self):
        sd = self.paddle_optim.state_dict()
        sd["param_idx_to_name"] = self.param_idx_to_name.to_forward()
        return sd

    def load_state_dict(self, state_dict_to_load):
        if "param_idx_to_name" in state_dict_to_load:
            loaded_param_idx_to_name = Bidict(
                forward=state_dict_to_load["param_idx_to_name"]
            )
            loaded_states = state_dict_to_load["state"]
            cur_state_dict = self.paddle_optim.state_dict()
            cur_states = cur_state_dict["state"]
            cur_param_groups = cur_state_dict["param_groups"]
            for cur_param_group in cur_param_groups:
                for cur_param_idx in cur_param_group["params"]:
                    param_name = self.param_idx_to_name.get_forward(cur_param_idx)
                    loaded_param_idx = loaded_param_idx_to_name.get_backward(param_name)
                    if loaded_param_idx not in loaded_states:
                        cur_states.pop(loaded_param_idx, None)
                    else:
                        cur_states[cur_param_idx] = loaded_states[loaded_param_idx]
            state_dict_to_load = dict(state=cur_states, param_groups=cur_param_groups)
        self.paddle_optim.load_state_dict(state_dict_to_load)
