from typing import Sequence

import numpy as np
import paddle
from omegaconf import DictConfig


class BaseProcess:
    def __init__(self, cfg: DictConfig, num_predictions, inputs_noise):
        self.cfg = cfg
        self.num_predictions = num_predictions
        self.inputs_noise = inputs_noise

    def get_ensemble_inputs(self, inputs_raw, add_noise=True, flatten_into_batch_dim=True):
        """Get the inputs for the ensemble predictions"""
        if inputs_raw is None:
            return None
        if self.num_predictions <= 1:
            return inputs_raw

        # create a batch of inputs for the ensemble predictions
        if isinstance(inputs_raw, dict):
            return {k: self.get_ensemble_inputs(v, add_noise, flatten_into_batch_dim) for k, v in inputs_raw.items()}

        if isinstance(inputs_raw, Sequence):
            inputs = np.array([inputs_raw] * self.num_predictions)
        elif add_noise:
            noise = self.inputs_noise * paddle.randn(shape=inputs_raw.shape, dtype=inputs_raw.dtype)
            inputs = paddle.stack([inputs_raw + noise for _ in range(self.num_predictions)], axis=0)
        else:
            inputs = paddle.stack([inputs_raw for _ in range(self.num_predictions)], axis=0)

        if flatten_into_batch_dim:
            # flatten num_predictions and batch dimensions "N B ... -> (N B) ..."
            inputs = inputs.reshape([-1] + list(inputs.shape[2:]))
        return inputs

    def reshape_preds(self, preds: paddle.Tensor):
        N, C, W, H = preds.shape
        assert (
            N % self.num_predictions == 0
        ), f"Number of samples {N} must be divisible by ensemble size {self.num_predictions}"
        preds = preds.reshape([self.num_predictions, -1, C, W, H])
        return preds
