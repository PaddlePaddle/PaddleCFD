import math
from typing import Optional

import numpy as np
import paddle
from omegaconf import DictConfig
from tqdm import tqdm

from ppcfd.models.ppdiffusion.process.forecasting import Forecasting
from ppcfd.models.ppdiffusion.process.interpolation import Interpolation
from ppcfd.models.ppdiffusion.process.sampling import Sampling


class DYffusion:
    """
    DYffusion model with a pretrained interpolator

    Args:
        interpolator: the interpolator model
        lambda_rec_base: the weight of the reconstruction loss
        lambda_rec_fb: the weight of the reconstruction loss (using the predicted xt_last as feedback)
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.interp_obj = Interpolation(cfg)
        self.forecast_obj = Forecasting(cfg)
        self.sampling_obj = Sampling(cfg, self._interpolate)

        self.num_timesteps = cfg.FORECASTING.num_timesteps
        self.lambda_rec_base = cfg.FORECASTING.lambda_rec_base
        self.lambda_rec_fb = cfg.FORECASTING.lambda_rec_fb

    def init_models(self, interp_model, forecast_model):
        self.interp_obj.set_model(interp_model)
        self.interp_obj.model.freeze()
        self.forecast_obj.set_model(forecast_model)
        do_enable = self.forecast_obj.model.training or self.cfg.EVAL.enable_infer_dropout
        ipol_handles = [self.interp_obj.model]
        self.sampling_obj.update_handles(ipol_handles, do_enable)

    def _interpolate(
        self,
        init_cond: paddle.Tensor,
        x_last: paddle.Tensor,
        time: paddle.Tensor,
        static_cond: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        # interpolator networks uses time in [1, horizon-1]
        assert (0 < time).all() and (
            time < self.interp_obj.horizon
        ).all(), f"interpolate time must be in (0, {self.interp_obj.horizon}), got {time}"

        # select condition data to be consistent with the interpolator training data
        interp_inputs = paddle.concat([init_cond, x_last], axis=1)
        interp_preds = self.interp_obj.model(interp_inputs, condition=static_cond, time=time, **kwargs)
        # interp_preds = self.interp_obj.reshape_preds(interp_preds)    # do not reshape for training
        return interp_preds

    def encode_time(self, time):
        time_encoding = self.sampling_obj.time_encoding
        assert time_encoding in [
            "discrete",
            "normalized",
            "dynamics",
        ], f"Invalid time_encoding: {time_encoding}."
        if time_encoding == "discrete":
            return time.astype("float32")
        if time_encoding == "normalized":
            return time.astype("float32") / self.num_timesteps
        if time_encoding == "dynamics":
            return self.sampling_obj.diffu_to_interp_step(time)

    def predict_x_last(
        self,
        x_t: paddle.Tensor,
        condition: paddle.Tensor,
        time: paddle.Tensor,
        is_sampling: bool = False,
        static_cond: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        # predict_x_last = using model in forward mode
        forward_cond = self.forecast_obj.add_noise(condition)
        if static_cond is not None:
            forward_cond = static_cond if forward_cond is None else paddle.concat([forward_cond, static_cond], axis=1)
        time = self.encode_time(time)
        x_last_pred = self.forecast_obj.model(x_t, time=time, condition=forward_cond)
        return x_last_pred

    def forward(self, data_dict):
        """forward of training.

        Args:
            data_dict (Dict): data dict of dynamics, conditions and metadata and others(if exist).
                The "condition" in data_dict actually refers to the static condition,
                which represents information such as the obstacle mask.
        """
        dynamics = data_dict["dynamics"]
        inputs, extra_kwargs = self.forecast_obj.data_transform(data_dict, ensemble=False)
        condition = inputs
        time = (
            extra_kwargs["time"]
            if "time" in extra_kwargs
            else paddle.randint(low=0, high=self.num_timesteps, shape=(dynamics.shape[0],), dtype="int64")
        )
        static_cond = extra_kwargs["condition"] if "condition" in extra_kwargs else None
        x_last = dynamics[:, -1, ...]

        def _gen_inputs_kst(time_mask: paddle.Tensor, time_offset: int = 0):
            if not time_mask.any():
                return None
            subset = {
                "x_end": condition[time_mask],
                "x0": x_last[time_mask],
                "time": time[time_mask] + time_offset,
                "static_cond": static_cond[time_mask] if static_cond is not None else None,
            }
            return self.sampling_obj.q_sample(**subset, num_predictions=1).astype(condition.dtype)

        # Create the inputs for the forecasting model
        #   1. For t=0, simply use the initial conditions
        x_t = condition.clone()
        #   2. For t>0, we need to interpolate the data using the interpolator
        time_mask = time > 0
        if time_mask.any():
            x_t[time_mask] = _gen_inputs_kst(time_mask=time_mask)

        # Train the forward predictions (i.e. predict xt_last from xt_t)
        xt_last_pred = self.predict_x_last(x_t, condition=condition, time=time, static_cond=static_cond)
        preds = [xt_last_pred]
        targets = [x_last]

        # Train the forward predictions II by emulating one more step of the diffusion process
        time_mask_sec = time <= self.num_timesteps - 2
        if self.lambda_rec_fb > 0 and time_mask_sec.any():
            x_interp_sec = _gen_inputs_kst(time_mask=time_mask_sec, time_offset=1)
            xt_last_pred_sec = self.predict_x_last(
                x_interp_sec,
                condition=condition[time_mask_sec],
                time=time[time_mask_sec] + 1,
                static_cond=static_cond[time_mask_sec],
            )
            preds.append(xt_last_pred_sec)
            targets.append(x_last[time_mask_sec])
        return preds, targets

    def get_loss(self, preds, targets, loss_fn):
        loss_1 = loss_fn(preds[0], targets[0])
        loss_2 = loss_fn(preds[1], targets[1]) if len(preds) == 2 else 0.0
        return loss_1 * self.lambda_rec_base + loss_2 * self.lambda_rec_fb

    def predict(self, inputs, reshape_ensemble_dim=True, **kwargs):
        kwargs["static_cond"] = kwargs.pop("condition") if "condition" in kwargs else None
        kwargs["sampling_fn"] = self.predict_x_last
        kwargs["num_input_channels"] = self.forecast_obj.model.num_input_channels
        if "num_predictions" not in kwargs:
            kwargs["num_predictions"] = self.forecast_obj.num_predictions
        preds_dict = self.sampling_obj.sample(inputs, **kwargs)
        return_dict = {k: self.forecast_obj.reshape_preds(v) for k, v in preds_dict.items()}
        return return_dict

    @paddle.no_grad()
    def eval(
        self,
        data_dict,
        pred_horizon=64,
        metric_fn=None,
        enable_ar: bool = False,
        ar_steps: int = 0,
    ):
        dynamics = data_dict["dynamics"].clone()
        return_dict = dict()
        with_bc = hasattr(self.forecast_obj, "boundary_conditions")
        if with_bc:
            kwargs = self.forecast_obj.get_bc_kwargs(data_dict)
            total_t = kwargs["t0"]
            dt = kwargs["dt"]
        else:
            total_t = 0.0
            dt = 1.0

        forecast_horizon = self.forecast_obj.horizon
        self.pred_timesteps = list(np.arange(1, forecast_horizon + 1))
        predicted_range_last = [0.0] + self.pred_timesteps[:-1]
        ar_window_steps_t = self.pred_timesteps[-self.forecast_obj.window :]  # autoregressive window steps,

        ar_inputs = None
        ar_loops = 1
        if enable_ar:
            if dynamics.shape[1] < pred_horizon:
                raise ValueError(f"Prediction horizon {pred_horizon} exceeds dynamics shape {dynamics.shape}")
            ar_loops = (
                max(1, math.ceil(pred_horizon / forecast_horizon))
                if ar_steps == 0 and pred_horizon is not None
                else ar_steps + 1
            )

        # enable autoregressive
        for ar_step in tqdm(
            range(ar_loops),
            desc="Autoregressive Step",
            position=0,
            leave=True,
            disable=ar_loops <= 1,
        ):
            is_ar_mode = ar_inputs is not None
            inputs, extra_kwargs = self.forecast_obj.data_transform(
                data_dict,
                ensemble=True,
                is_ar_mode=is_ar_mode,
            )
            if is_ar_mode:
                inputs = ar_inputs
                extra_kwargs["num_predictions"] = 1
            sampling_data_dict = self.predict(inputs, **extra_kwargs)

            ar_window_steps = []
            results = {}
            for t_step_last, t_step in zip(predicted_range_last, self.pred_timesteps):
                total_horizon = ar_step * forecast_horizon + t_step
                if total_horizon > pred_horizon:
                    # May happen if we have a prediction horizon that is not a multiple of the true horizon
                    break
                total_t += dt * (t_step - t_step_last)  # update time, by default this is == dt
                if with_bc and float(total_horizon).is_integer():
                    target_time = self.forecast_obj.window + int(total_horizon) - 1
                    targets = dynamics[:, target_time, ...]
                    preds = self.forecast_obj.boundary_conditions(
                        preds=sampling_data_dict[f"t{t_step}_preds"],
                        targets=targets,
                        metadata=data_dict.get("metadata", None),
                        time=total_t,
                    )
                else:
                    targets = None
                    preds = sampling_data_dict[f"t{t_step}_preds"]

                results = {f"t{total_horizon}_preds": preds, f"t{total_horizon}_targets": targets}
                return_dict.update(results)

                if t_step in ar_window_steps_t:
                    # if predicted_range == self.horizon_range and window == 1, then this is just the last step :)
                    # Need to keep the last window steps that are INTEGER steps!
                    # [20, 4, 3, 221, 42] --> [80, 1, 3, 221, 42]
                    ar_window_steps += [preds.reshape([-1, 1, *preds.shape[-3:]])]  # keep t,c,h,w

                if self.forecast_obj.num_predictions > 1:
                    preds = paddle.mean(preds, axis=0)
                    assert preds.shape == targets.shape, (
                        f"After averaging over ensemble dim: "
                        f"preds.shape={preds.shape} != targets.shape={targets.shape}"
                    )
                # metric = metric_fn(preds, targets)

            if ar_step < ar_loops - 1:  # if not last step, then update dynamics
                ar_inputs = paddle.concat(ar_window_steps, axis=1)  # shape (b, window, c, h, w)
                ar_inputs = self.forecast_obj.transform_inputs(ar_inputs, ensemble=False)
                data_dict["dynamics"] *= 1e6  # become completely dummy after first multistep prediction

        return return_dict
