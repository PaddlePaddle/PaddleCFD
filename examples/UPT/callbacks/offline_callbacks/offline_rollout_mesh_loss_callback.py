from functools import partial

import einops
import paddle
from callbacks.base.periodic_callback import PeriodicCallback
from kappadata.wrappers import ModeWrapper
from paddle_geometric.utils import scatter
from utils.formatting_util import dict_to_string


class OfflineRolloutMeshLossCallback(PeriodicCallback):
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
        data = trainer_model.prepare(batch, mode="rollout")
        batch, ctx = batch
        batch_idx = ctx["batch_idx"].to(model.device, non_blocking=True)
        assert "target" not in data
        x = data.pop("x")
        assert (
            x.ndim == 3
        ), "expected data to be of shape (bs * num_points, num_total_timesteps + 1, input_dim)"
        if x.size(1) != self.num_rollout_timesteps + 1:
            x = x[:, : self.num_rollout_timesteps + 1]
        model_input_dim, _ = model.input_shape
        _, _, x_input_dim = x.shape
        assert model_input_dim % x_input_dim == 0
        num_input_timesteps = model_input_dim // x_input_dim
        x0 = einops.repeat(
            x[:, (0)],
            "batch_num_points num_channels ... -> batch_num_points (num_input_timesteps num_channels) ...",
            num_input_timesteps=num_input_timesteps,
        )
        data.pop("timestep", None)
        with trainer.autocast_context:
            predictions = model.rollout(
                x0=x0,
                num_rollout_timesteps=self.num_rollout_timesteps,
                **data,
                **self.rollout_kwargs,
            )
        ground_truth = x[:, 1 : 1 + self.num_rollout_timesteps]
        normalized_delta = (predictions - ground_truth).abs()
        normalized_delta = normalized_delta.mean(dim=2)
        normalized_delta = normalized_delta.mean(dim=1)
        normalized_delta = scatter(src=normalized_delta, index=batch_idx, reduce="mean")
        return dict(overall_normalized_delta=normalized_delta)

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
        metric_identifier = f"{self.dataset_key}/0to{self.num_rollout_timesteps}"
        if len(self.rollout_kwargs) > 0:
            metric_identifier = (
                f"{metric_identifier}/{dict_to_string(self.rollout_kwargs)}"
            )
        self.writer.add_scalar(
            key=f"delta/{metric_identifier}/overall/normalized",
            value=results["overall_normalized_delta"].mean(),
            logger=self.logger,
            format_str=".10f",
        )
