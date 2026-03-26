from functools import partial

import einops
import kappaprofiler as kp
import paddle
from callbacks.base.periodic_callback import PeriodicCallback
from kappadata.wrappers import ModeWrapper


class OfflineRolloutSpeedCallback(PeriodicCallback):
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
        assert x.ndim == 2, "expected data to be of shape (bs * num_points, input_dim)"
        assert (
            target.ndim == 3
        ), "expected data to be of shape (bs * num_points, input_dim, max_timesteps)"
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
        return x_hat.mean().item()

    def _periodic_callback(
        self, model, trainer, trainer_model, batch_size, data_iter, **_
    ):
        with kp.Stopwatch() as sw:
            self.iterate_over_dataset(
                forward_fn=partial(
                    self._forward,
                    model=model,
                    trainer=trainer,
                    trainer_model=trainer_model,
                ),
                config_id=self.__config_id,
                batch_size=batch_size,
                data_iter=data_iter,
            )
        self.logger.info(f"rollout took: {sw.elapsed_seconds:.3f} seconds")
