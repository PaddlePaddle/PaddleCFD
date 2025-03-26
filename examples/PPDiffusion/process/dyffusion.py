from typing import Optional

import paddle
from omegaconf import DictConfig

from .forecasting import Forecasting
from .interpolation import Interpolation
from .sampling import Sampling


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

    def data_transform(self, data_dict):
        x_last = self.forecast_obj.get_evaluation_inputs(data_dict)
        kwargs = self.forecast_obj.get_extra_kwargs(data_dict)
        return x_last, kwargs

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

    def gen_inputs_kst(
        self,
        xt_last: paddle.Tensor,
        time_selected: paddle.Tensor,
        condition: paddle.Tensor,
        time: paddle.Tensor,
        static_cond: paddle.Tensor = None,
        kst: int = 0,
    ):
        """Generate the inputs for the forecasting model.

        Args:
            xt_last (paddle.Tensor): the start/target data  (time = horizon)
            condition (paddle.Tensor): the initial condition data  (time = 0)
            time (paddle.Tensor): the time step of the diffusion process
            static_cond (paddle.Tensor, optional): the static condition data (if any). Defaults to None.
        """
        if paddle.any(time_selected):
            cond_sub = condition[time_selected]
            xt_last_sub = xt_last[time_selected]
            time_sub = time[time_selected] + kst
            static_cond_sub = None if static_cond is None else static_cond[time_selected]
            return self.sampling_obj.q_sample(
                x_end=cond_sub,
                x0=xt_last_sub,
                time=time_sub,
                static_cond=static_cond_sub,
                num_predictions=1,
            ).astype(condition.dtype)

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

        time = self.sampling_obj.encode_time(time)
        x_last_pred = self.forecast_obj.model(x_t, time=time, condition=forward_cond)
        return x_last_pred

    def forward(self, data_dict):
        x_last, kwargs = self.data_transform(data_dict)
        condition, time, static_cond = kwargs["condition"], kwargs["time"], kwargs["static_cond"]

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
        x_t[time > 0] = _gen_inputs_kst(time_mask=time > 0)
        xt_last_pred = self.predict_x_last(x_t, condition=condition, time=time, static_cond=static_cond)
        preds = [xt_last_pred]
        targets = [x_last]

        # Train the forward predictions (i.e. predict xt_last from xt_t)
        if self.lambda_rec_fb > 0:
            time_mask_sec = time <= self.num_timesteps - 2
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

    @paddle.no_grad()
    def eval(
        self,
        data_dict,
        pred_horizon=64,
        metric_fn=None,
    ):
        return_dict = dict()
        inputs = self.forecast_obj.get_evaluation_inputs(data_dict, ensemble=ensemble)
        extra_kwargs = self.forecast_obj.get_extra_kwargs(data_dict, ensemble=ensemble)
        if with_bc:
            kwargs = self.forecast_obj.get_bc_kwargs(data_dict)
            total_t = kwargs["t0"]
            dt = kwargs["dt"]
        else:
            total_t = 0.0
            dt = 1.0

        # TODO: predict
        sampling_data_dict = self.predict(inputs, **extra_kwargs, **kwargs)

        # TODO: enable autoregressive
        # n_outer_loops = 1  # Simple evaluation without autoregressive steps
        self.pred_timesteps = list(np.arange(1, self.horizon + 1))
        predicted_range_last = [0.0] + self.pred_timesteps[:-1]

        results = {}
        for t_step_last, t_step in zip(predicted_range_last, self.pred_timesteps):
            if t_step > pred_horizon:
                # May happen if we have a prediction horizon that is not a multiple of the true horizon
                break
            total_t += dt * (t_step - t_step_last)  # update time, by default this is == dt
            target_time = self.forecast_obj.window + int(t_step) - 1
            targets = dynamics[:, target_time, ...]
            if with_bc:
                preds = self.forecast_obj.boundary_conditions(
                    preds=sampling_data_dict[f"t{t_step}_preds"],
                    targets=targets,
                    metadata=data_dict.get("metadata", None),
                    time=total_t,
                )
            else:
                preds = sampling_data_dict[f"t{t_step}_preds"]

            results = {f"t{t_step}_preds": preds, f"t{t_step}_targets": targets}
            return_dict.update(results)
            if self.forecast_obj.num_predictions > 1:
                preds = paddle.mean(preds, axis=0)

    def predict(self, inputs, reshape_ensemble_dim=True, **kwargs):
        kwargs["static_condition"] = condition
        inital_condition = inputs
        results = self.sampling_obj.sample(inital_condition, **kwargs)
        results = {"preds": results}

        results = self.reshape_predictions(results, reshape_ensemble_dim)
        results = self.unpack_predictions(results)
        return results
