from typing import Dict
from typing import List

import numpy as np
import paddle
from einops import rearrange
from omegaconf import DictConfig

from ppcfd.models.ppdiffusion.process.base_process import BaseProcess


class Interpolation(BaseProcess):
    def __init__(self, cfg: DictConfig):
        super().__init__(
            cfg=cfg,
            num_predictions=cfg.INTERPOLATION.num_predictions,
            inputs_noise=cfg.INTERPOLATION.prediction_inputs_noise,
        )
        self.window = cfg.INTERPOLATION.window
        self.horizon = cfg.INTERPOLATION.horizon
        self.stack_window_to_channel_dim = cfg.INTERPOLATION.stack_window_to_channel_dim

    def set_model(self, model):
        self.model = model

    def model_cfg_transform(self, cfg_model):
        ratio = (self.window + 1) if self.stack_window_to_channel_dim else 2
        cfg_new = dict(cfg_model)
        cfg_new["num_input_channels"] = cfg_model["input_channels"] * ratio
        cfg_new["num_output_channels"] = cfg_model["output_channels"]
        return cfg_new

    def get_inputs_from_dynamics(self, dynamics):
        """
        Get the inputs from the dynamics tensor.
        Since we are doing interpolation, this consists of the first window frames plus the last frame.
        """
        assert dynamics.shape[1] == self.window + self.horizon, "dynamics must have shape (b, t, c, h, w)"
        past_steps = dynamics[:, : self.window, ...]  # (b, window, c, lat, lon) at time 0
        last_step = dynamics[:, -1, ...]  # (b, c, lat, lon) at time t=window+horizon
        if self.stack_window_to_channel_dim:
            past_steps = rearrange(past_steps, "b window c lat lon -> b (window c) lat lon")
        else:
            last_step = last_step.unsqueeze(1)  # (b, 1, c, lat, lon)
        inputs = paddle.concat([past_steps, last_step], axis=1)  # (b, window*c + c, lat, lon)
        return inputs

    def data_transform(self, data_dict):
        dynamics = data_dict["dynamics"]  # dynamics is a (b, t, c, h, w) tensor
        inputs = self.get_inputs_from_dynamics(dynamics)  # (b, c, h, w) at time 0
        b = dynamics.shape[0]

        possible_times = paddle.to_tensor(np.arange(1, self.horizon), dtype=paddle.int64)  # (h,)
        # take random choice of time
        t = possible_times[paddle.randint(0, len(possible_times), shape=[b], dtype=paddle.int64)]  # (b,)
        # t = paddle.randint(start_t, max_t, (b,), dtype=paddle.int64)  # (b,)
        targets = dynamics[paddle.arange(b), self.window + t - 1, ...]  # (b, c, h, w)
        data_dict.update({"inputs": inputs, "targets": targets, "time": t})
        return data_dict

    def get_evaluation_inputs(self, dynamics, **kwargs):
        inputs = self.get_inputs_from_dynamics(dynamics)
        inputs = self.get_ensemble_inputs(inputs)
        return inputs

    def get_extra_model_kwargs(self, data_dict):
        extra_kwargs = {}
        for k, v in data_dict.items():
            if k != "dynamics":
                extra_kwargs[k] = self.get_ensemble_inputs(v, add_noise=False)
        return extra_kwargs

    def concat_results(self, outputs: List[Dict]):
        results = {}
        for key in outputs[0].keys():
            data_list = [out[key] for out in outputs]
            if "target" not in key:
                assert data_list[0].shape[0] == self.num_predictions, "Shape of preds is error"
                batch_axis = 1
            else:
                batch_axis = 0

            results[key] = paddle.concat(data_list, axis=batch_axis)
        return results

    def forward(self, data_dict):
        data_dict = self.data_transform(data_dict)
        targets = data_dict.pop("targets")
        preds = self.model(**data_dict)
        return preds, targets

    def get_loss(self, preds, targets, loss_fn):
        # # TODO: predictions_mask
        # if predictions_mask is not None:
        #     preds = preds[predictions_mask]
        loss = loss_fn(preds, targets)
        return loss

    @paddle.no_grad()
    def eval(self, data_dict, metric_fn=None):
        return_dict = dict()
        dynamics = data_dict["dynamics"]
        inputs = self.get_evaluation_inputs(dynamics)
        extra_kwargs = self.get_extra_model_kwargs(data_dict)
        # t_step_metrics = defaultdict(list)
        for t_step in np.arange(1, self.horizon):
            targets = dynamics[:, self.window + t_step - 1, ...]
            time = paddle.full(shape=inputs.shape[0:1], fill_value=t_step, dtype="int64")
            preds = self.model(inputs, time=time, **extra_kwargs)
            preds = self.reshape_preds(preds)
            results = {f"t{t_step}_preds": preds, f"t{t_step}_targets": targets}
            return_dict.update(results)
        return return_dict
