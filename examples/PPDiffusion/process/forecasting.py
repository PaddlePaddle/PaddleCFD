from typing import Any, Dict, Sequence

import numpy as np
import paddle
from omegaconf import DictConfig


class Forecasting:
    """
    DYffusion model with a pretrained interpolator

    Args:
        interpolator: the interpolator model
        lambda_rec_base: the weight of the reconstruction loss
        lambda_rec_fb: the weight of the reconstruction loss (using the predicted xt_last as feedback)
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.CHANNEL_DIM = -3
        self.window = cfg.FORECASTING.window
        self.horizon = cfg.FORECASTING.horizon
        self.stack_window_to_channel_dim = cfg.FORECASTING.stack_window_to_channel_dim
        self.num_timesteps = cfg.FORECASTING.num_timesteps
        self.inputs_noise = cfg.FORECASTING.prediction_inputs_noise
        self.pred_timesteps = cfg.FORECASTING.pred_timesteps
        self.num_predictions = cfg.FORECASTING.num_predictions
        self.USE_TIME_AS_EXTRA_INPUT = False

        self.forward_cond = cfg.FORECASTING.forward_cond
        fcond_options = ["data", "none", "data+noise"]
        assert (
            self.forward_cond in fcond_options
        ), f"Error: forward_cond should be one of {fcond_options} but got {self.forward_cond}."

    def model_cfg_transform(self, cfg_model):
        # TODO: maybe params change
        # ratio = (self.window + 1) if self.stack_window_to_channel_dim else 2
        cfg_new = dict(cfg_model)
        cfg_new["num_input_channels"] = cfg_model["input_channels"]
        cfg_new["num_output_channels"] = cfg_model["output_channels"]
        return cfg_new

    def get_ensemble_inputs(self, inputs_raw, add_noise=True, flatten_into_batch_dim=True):
        """Get the inputs for the ensemble predictions"""
        if inputs_raw is None:
            return None
        if self.num_timesteps <= 1:
            return inputs_raw

        # create a batch of inputs for the ensemble predictions
        if isinstance(inputs_raw, dict):
            return {k: self.get_ensemble_inputs(v, add_noise, flatten_into_batch_dim) for k, v in inputs_raw.items()}

        if isinstance(inputs_raw, Sequence):
            inputs = np.array([inputs_raw] * self.num_timesteps)
        elif add_noise:
            noise = self.inputs_noise * paddle.randn(shape=inputs_raw.shape, dtype=inputs_raw.dtype)
            inputs = paddle.stack([inputs_raw + noise for _ in range(self.num_timesteps)], axis=0)
        else:
            inputs = paddle.stack([inputs_raw for _ in range(self.num_timesteps)], axis=0)

        if flatten_into_batch_dim:
            # flatten num_timesteps and batch dimensions
            inputs = inputs.reshape([-1] + list(inputs.shape[2:]))
        return inputs

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
        is_autoregressive: bool = False,
    ) -> Dict[str, Any]:
        dynamics_shape = data_dict["dynamics"].shape  # b, dyn_len, c, h, w = dynamics.shape
        extra_kwargs = {}
        ensemble_k = ensemble and not is_autoregressive
        if self.USE_TIME_AS_EXTRA_INPUT:
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

    def get_evaluation_inputs(self, data_dict, ensemble=False):
        dynamics = data_dict["dynamics"]  # dynamics is a (b, t, c, h, w) tensor
        inputs = dynamics[:, : self.window, ...]
        inputs = self.transform_inputs(inputs, ensemble=False)
        x_last = data_dict["dynamics"][:, -1, ...]
        return x_last

    def get_extra_kwargs(self, data_dict, time=None, ensemble=False):
        extra_kwargs = self.get_extra_model_kwargs(
            data_dict,
            time=time,
            ensemble=ensemble,
        )
        b = data_dict["dynamics"].shape[0]
        kwargs = {}
        kwargs["static_cond"] = extra_kwargs["condition"]
        kwargs["condition"] = inputs
        kwargs["time"] = (
            extra_kwargs["time"]
            if "time" in extra_kwargs
            else paddle.randint(low=0, high=self.num_timesteps, shape=(b,), dtype="int64")
        )
        return kwargs

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

    def set_model(self, model):
        self.model = model

    def boundary_conditions(
        self,
        preds: paddle.Tensor,
        targets: paddle.Tensor,
        metadata,
        time: float = None,
    ) -> paddle.Tensor:
        print("### datamodule.boundary_conditions")
        batch_size = targets.shape[0]
        for b_i in range(batch_size):
            t_i = time if isinstance(time, float) else time[b_i].item()
            print("type(t_i)", type(t_i))
            in_velocity = float(metadata["in_velocity"][b_i].item())
            fixed_mask_solutions_pressures = metadata["fixed_mask"][b_i, ...]
            assert (
                fixed_mask_solutions_pressures.shape == preds.shape[-3:]
            ), f"fixed_mask_solutions_pressures={fixed_mask_solutions_pressures.shape}, predictions={preds.shape}"
            vertex_y = metadata["vertices"][b_i, 1, 0, :]

            left_boundary_indexing = paddle.zeros((3, 221, 42), dtype=paddle.bool)
            left_boundary_indexing[0, 0, :] = True  # only for first p
            left_boundary = in_velocity * 4 * vertex_y * (0.41 - vertex_y) / (0.41**2) * (1 - math.exp(-5 * t_i))
            left_boundary = left_boundary.unsqueeze(0)

            # the predictions should be of shape (*, 3, 221, 42)
            preds[b_i, ..., fixed_mask_solutions_pressures] = 0
            preds[b_i, ..., left_boundary_indexing] = left_boundary
        return preds

    def get_bc_kwargs(self, data_dict):
        metadata = data_dict["metadata"]
        t0 = metadata["t"][:, 0]
        dt = metadata["time_step_size"]
        return dict(t0=t0, dt=dt)
