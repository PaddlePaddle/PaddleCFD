from functools import partial

import einops
import paddle
from callbacks.base.periodic_callback import PeriodicCallback
from kappadata.wrappers import ModeWrapper
from paddle_geometric.utils import scatter
from utils.formatting_util import dict_to_string


class OfflineCfdRolloutMeshLossCallback(PeriodicCallback):
    def __init__(
        self, dataset_key, num_rollout_timesteps=None, rollout_kwargs=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.num_rollout_timesteps = num_rollout_timesteps
        self.rollout_kwargs = rollout_kwargs or {}
        self.__config_id = None

    def _before_training(self, trainer, **kwargs):
        dataset, _ = self.data_container.get_dataset(
            key=self.dataset_key, mode=trainer.dataset_mode
        )
        if self.num_rollout_timesteps is None:
            self.num_rollout_timesteps = dataset.getdim_timestep()
        else:
            assert 0 < self.num_rollout_timesteps <= dataset.getdim_timestep()

    def _register_sampler_configs(self, trainer):
        self.__config_id = self._register_sampler_config_from_key(
            key=self.dataset_key, mode=trainer.dataset_mode
        )

    def _forward(self, batch, model, trainer, trainer_model):
        data = trainer_model.prepare(batch)
        batch, ctx = batch
        batch_idx = ctx["batch_idx"].to(model.device, non_blocking=True)
        target = data.pop("target")
        x = data.pop("x")
        assert x.ndim == 2, "expected data to be of shape (bs * num_points, input_dim)"
        assert (
            target.ndim == 3
        ), "expected data to be of shape (bs * num_points, input_dim, max_timesteps)"
        if target.size(2) != self.num_rollout_timesteps:
            target = target[:, :, : self.num_rollout_timesteps]
        _, model_input_dim = model.input_shape
        _, x_dim = x.shape
        assert model_input_dim % x_dim == 0
        num_input_timesteps = model_input_dim // x_dim
        x = einops.repeat(
            x,
            "batch_num_points num_channels -> batch_num_points (num_input_timesteps num_channels)",
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
        normalized_delta = (x_hat - target).abs()
        normalized_delta = normalized_delta.mean(dim=[1])
        assert self.num_rollout_timesteps > 50
        normalized_delta10 = scatter(
            src=normalized_delta[:, :10].mean(dim=1), index=batch_idx, reduce="mean"
        )
        normalized_delta20 = scatter(
            src=normalized_delta[:, :20].mean(dim=1), index=batch_idx, reduce="mean"
        )
        normalized_delta50 = scatter(
            src=normalized_delta[:, :50].mean(dim=1), index=batch_idx, reduce="mean"
        )
        normalized_delta = scatter(
            src=normalized_delta.mean(dim=1), index=batch_idx, reduce="mean"
        )
        return dict(
            normalized_delta10=normalized_delta10,
            normalized_delta20=normalized_delta20,
            normalized_delta50=normalized_delta50,
            normalized_delta99=normalized_delta,
        )

    def _periodic_callback(
        self, model, trainer, trainer_model, batch_size, data_iter, **_
    ):
        results = self.iterate_over_dataset(
            forward_fn=partial(
                self._forward, model=model, trainer=trainer, trainer_model=trainer_model
            ),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )
        for num_rollout_timesteps in [10, 20, 50, 99]:
            metric_identifier = f"{self.dataset_key}/0to{num_rollout_timesteps}"
            if len(self.rollout_kwargs) > 0:
                metric_identifier = (
                    f"{metric_identifier}/{dict_to_string(self.rollout_kwargs)}"
                )
            self.writer.add_scalar(
                key=f"delta/{metric_identifier}/overall/normalized",
                value=results[f"normalized_delta{num_rollout_timesteps}"].mean(),
                logger=self.logger,
                format_str=".10f",
            )
