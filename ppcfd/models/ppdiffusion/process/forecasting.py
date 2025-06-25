import math
from typing import Any
from typing import Dict

import paddle
from omegaconf import DictConfig

from ppcfd.models.ppdiffusion.process.base_process import BaseProcess


class Forecasting(BaseProcess):
    def __init__(self, cfg: DictConfig):
        super().__init__(
            cfg=cfg,
            num_predictions=cfg.FORECASTING.num_predictions,
            inputs_noise=cfg.FORECASTING.prediction_inputs_noise,
        )

        self.CHANNEL_DIM = -3
        self.window = cfg.FORECASTING.window
        self.horizon = cfg.FORECASTING.horizon
        self.stack_window_to_channel_dim = cfg.FORECASTING.stack_window_to_channel_dim
        self.num_timesteps = cfg.FORECASTING.num_timesteps
        self.pred_timesteps = cfg.FORECASTING.pred_timesteps
        self.use_time_as_extra_input = False

        self.forward_cond = cfg.FORECASTING.forward_cond
        fcond_options = ["data", "none", "data+noise"]
        assert (
            self.forward_cond in fcond_options
        ), f"Error: forward_cond should be one of {fcond_options} but got {self.forward_cond}."

    def set_model(self, model):
        self.model = model

    def model_cfg_transform(self, cfg_model):
        # TODO: maybe params change
        # ratio = (self.window + 1) if self.stack_window_to_channel_dim else 2
        cfg_new = dict(cfg_model)
        cfg_new["num_input_channels"] = cfg_model["input_channels"]
        cfg_new["num_output_channels"] = cfg_model["output_channels"]
        return cfg_new

    def transform_inputs(
        self, inputs: paddle.Tensor, time: paddle.Tensor = None, ensemble: bool = True, **kwargs
    ) -> paddle.Tensor:
        if self.stack_window_to_channel_dim and len(inputs.shape) == 5:
            # "b window c lat lon -> b (window c) lat lon"
            inputs = inputs.reshape([0, -1] + list(inputs.shape[3:]))
        if ensemble:
            inputs = self.get_ensemble_inputs(inputs, **kwargs)
        return inputs

    def get_extra_model_kwargs(
        self,
        data_dict: Dict[str, paddle.Tensor],
        time: paddle.Tensor,
        ensemble: bool,
        is_ar_mode: bool = False,
    ) -> Dict[str, Any]:
        dynamics_shape = data_dict["dynamics"].shape  # b, dyn_len, c, h, w = dynamics.shape
        extra_kwargs = {}
        ensemble_k = ensemble and not is_ar_mode
        if self.use_time_as_extra_input:
            data_dict["time"] = time
        for k, v in data_dict.items():
            if k == "dynamics":
                continue
            elif k == "metadata":
                extra_kwargs[k] = self.get_ensemble_inputs(v, add_noise=False) if ensemble_k else v
                continue

            v_shape_no_channel = v.shape[1 : self.CHANNEL_DIM] + v.shape[self.CHANNEL_DIM + 1 :]
            time_varying_feature = dynamics_shape[1 : self.CHANNEL_DIM] + dynamics_shape[self.CHANNEL_DIM + 1 :]

            if v_shape_no_channel == time_varying_feature:
                # if same shape as dynamics (except for batch size/#channels), then assume it is a time-varying feature
                extra_kwargs[k] = v[:, : self.window, ...]
                extra_kwargs[k] = self.transform_inputs(extra_kwargs[k], time=time, ensemble=ensemble, add_noise=False)
            else:
                # Static features
                extra_kwargs[k] = self.get_ensemble_inputs(v, add_noise=False) if ensemble else v
        return extra_kwargs

    def data_transform(self, data_dict, time=None, ensemble=False, is_ar_mode=False):
        extra_kwargs = self.get_extra_model_kwargs(
            data_dict,
            time=time,
            ensemble=ensemble,
            is_ar_mode=is_ar_mode,
        )
        if not is_ar_mode:
            dynamics = data_dict["dynamics"]  # dynamics is a (b, t, c, h, w) tensor
            inputs = dynamics[:, : self.window, ...]
            inputs = self.transform_inputs(inputs, ensemble=ensemble)
            return inputs, extra_kwargs
        else:
            return None, extra_kwargs

    def add_noise(self, condition):
        if self.forward_cond == "data":
            return condition
        if self.forward_cond == "none":
            return None
        if "data+noise" in self.forward_cond:
            # simply use factor t/T to scale the condition and factor (1-t/T) to scale the noise
            # this is the same as using a linear combination of the condition and noise
            time_factor = time.astype("float32") / (self.num_timesteps - 1)  # shape: (b,)
            time_factor = time_factor.reshape([condition.shape[0]] + [1] * (condition.ndim - 1))  # shape: (b, 1, 1, 1)
            # add noise to the data in a linear combination, s.t. the noise is more important at the beginning (t=0)
            # and less important at the end (t=T)
            noise = paddle.randn_like(condition) * (1 - time_factor)
            return time_factor * condition + noise

    def boundary_conditions(
        self,
        preds: paddle.Tensor,
        targets: paddle.Tensor,
        metadata,
        time: float = None,
    ) -> paddle.Tensor:
        batch_size = targets.shape[0]
        for b_i in range(batch_size):
            t_i = time if isinstance(time, float) else time[b_i].item()
            in_vel = float(metadata["in_velocity"][b_i].item())
            fixed_mask_sol_press = metadata["fixed_mask"][b_i, ...]
            assert (
                fixed_mask_sol_press.shape == preds.shape[-3:]
            ), f"fixed_mask_sol_press={fixed_mask_sol_press.shape}, predictions={preds.shape}"
            vertex_y = metadata["vertices"][b_i, 1, 0, :]

            lb_idx = paddle.zeros((3, 221, 42), dtype=paddle.bool)  # left boundary
            lb_idx[0, 0, :] = True  # only for first p
            lb = in_vel * 4 * vertex_y * (0.41 - vertex_y) / (0.41**2) * (1 - math.exp(-5 * t_i))
            lb = lb.unsqueeze(0)

            # the predictions should be of shape (*, 3, 221, 42)
            preds1 = paddle.where(
                fixed_mask_sol_press.broadcast_to(preds[b_i].shape), paddle.zeros_like(preds[b_i]), preds[b_i]
            )
            preds[b_i] = preds1.reshape(preds[b_i].shape)

            preds2 = paddle.where(lb_idx.broadcast_to(preds[b_i].shape), lb.broadcast_to(preds[b_i].shape), preds[b_i])
            preds[b_i] = preds2.reshape(preds[b_i].shape)
        return preds

    def get_bc_kwargs(self, data_dict):
        metadata = data_dict["metadata"]
        t0 = metadata["t"][:, 0]
        dt = metadata["time_step_size"]
        return dict(t0=t0, dt=dt)
