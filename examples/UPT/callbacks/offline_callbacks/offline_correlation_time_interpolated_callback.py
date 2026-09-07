from functools import partial

import einops
import paddle
from callbacks.base.periodic_callback import PeriodicCallback
from kappadata.wrappers import ModeWrapper


class OfflineCorrelationTimeInterpolatedCallback(PeriodicCallback):
    def __init__(
        self, dataset_key, num_rollout_timesteps=None, rollout_kwargs=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.__config_id = None
        self.dataset = None
        self.num_rollout_timesteps = num_rollout_timesteps
        self.rollout_kwargs = rollout_kwargs or {}

    def _register_sampler_configs(self, trainer):
        self.__config_id = self._register_sampler_config_from_key(
            key=self.dataset_key, mode=trainer.dataset_mode
        )

    def _before_training(self, trainer, **kwargs):
        self.dataset, _ = self.data_container.get_dataset(
            key=self.dataset_key, mode=trainer.dataset_mode
        )
        if self.num_rollout_timesteps is None:
            self.num_rollout_timesteps = self.dataset.getdim_timestep()
        else:
            assert 0 < self.num_rollout_timesteps <= self.dataset.getdim_timestep()

    def _forward(self, batch, model, trainer, trainer_model):
        data = trainer_model.prepare(batch)
        batch, ctx = batch
        target = data.pop("target")
        x = data.pop("x")
        assert x.ndim == 4, "expected data to be of shape (bs, dim, height, width)"
        assert (
            target.ndim == 3
        ), "expected data to be of shape (bs * num_points, input_dim, max_timesteps)"
        if target.size(2) != self.num_rollout_timesteps:
            target = target[:, :, : self.num_rollout_timesteps]
        _, model_input_dim = model.input_shape
        _, _, _, x_dim = x.shape
        assert model_input_dim % x_dim == 0
        num_input_timesteps = model_input_dim // x_dim
        x = einops.repeat(
            x,
            "batch_size height width dim -> batch_size height width (num_input_timesteps dim)",
            num_input_timesteps=num_input_timesteps,
        )
        data.pop("timestep", None)
        with trainer.autocast_context:
            x_hat = model.rollout(
                x=x,
                num_rollout_timesteps=self.num_rollout_timesteps,
                **data,
                **self.rollout_kwargs,
            )
        start = 0
        mean_corrs_per_timestep = []
        num_query_pos = data["query_pos"].size(1)
        for _ in range(len(x)):
            end = start + num_query_pos
            cur_preds = x_hat[start:end]
            cur_target = target[start:end]
            cur_preds_mean = paddle.mean(cur_preds, dim=2, keepdim=True)
            cur_target_mean = paddle.mean(cur_target, dim=2, keepdim=True)
            cur_preds_std = paddle.std(x=cur_preds, axis=2, unbiased=False)
            cur_target_std = paddle.std(x=cur_target, axis=2, unbiased=False)
            mean_corr_per_timestep = (
                paddle.mean(
                    (cur_preds - cur_preds_mean) * (cur_target - cur_target_mean), dim=2
                )
                / (cur_preds_std * cur_target_std).clamp(min=1e-12)
            ).mean(dim=0)
            mean_corrs_per_timestep.append(mean_corr_per_timestep)
            start = end
        mean_corrs_per_timestep = paddle.stack(mean_corrs_per_timestep)
        return mean_corrs_per_timestep

    def _periodic_callback(
        self, model, trainer, trainer_model, batch_size, data_iter, **_
    ):
        mean_corrs_per_timestep = self.iterate_over_dataset(
            forward_fn=partial(
                self._forward, model=model, trainer=trainer, trainer_model=trainer_model
            ),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )
        self.writer.add_scalar(
            key=f"correlation/{self.dataset_key}",
            value=mean_corrs_per_timestep.mean(),
            logger=self.logger,
            format_str=".4f",
        )
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            min_values, min_indices = (mean_corrs_per_timestep >= thresh).min(axis=1), (
                mean_corrs_per_timestep >= thresh
            ).argmin(axis=1)
            min_indices[min_values] = self.num_rollout_timesteps
            mean_corr_time = min_indices.float().mean()
            self.writer.add_scalar(
                key=f"correlation_time/thresh{str(thresh).replace('.', '')}/{self.dataset_key}",
                value=mean_corr_time,
                logger=self.logger,
                format_str=".4f",
            )
