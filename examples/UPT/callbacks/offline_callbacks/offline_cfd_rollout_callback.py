from functools import partial

import einops
import paddle
from callbacks.base.periodic_callback import PeriodicCallback
from kappadata.wrappers import ModeWrapper


class OfflineCfdRolloutCallback(PeriodicCallback):
    def __init__(
        self, dataset_key, num_rollout_timesteps=None, rollout_kwargs=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.__config_id = None
        self.dataset = None
        self.num_rollout_timesteps = num_rollout_timesteps
        self.rollout_kwargs = rollout_kwargs or {}
        self.out = self.path_provider.stage_output_path / "rollout"
        self.counter = 0
        self.dataset_mode = None

    def _register_sampler_configs(self, trainer):
        self.dataset_mode = f"{trainer.dataset_mode} target_t0"
        self.__config_id = self._register_sampler_config_from_key(
            key=self.dataset_key, mode=self.dataset_mode
        )

    def _before_training(self, trainer, **kwargs):
        self.out.mkdir(exist_ok=True)
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
        target_t0 = ModeWrapper.get_item(
            mode=self.dataset_mode, item="target_t0", batch=batch
        )
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
        start = 0
        if "batch_idx" in data:
            batch_idx = data["batch_idx"]
        else:
            batch_idx = ctx["batch_idx"].to(model.device)
        batch_size = batch_idx.unique().size
        for i in range(batch_size):
            end = start + (batch_idx == i).sum()
            cur_preds = x_hat[start:end]
            cur_target = target[start:end]
            cur_mesh_pos = data["mesh_pos"][start:end]
            cur_target_t0 = target_t0[(i), start:end]
            paddle.save(
                obj=cur_preds.half().clone(),
                path=self.out / f"{self.counter:04d}_rollout.th",
            )
            paddle.save(
                obj=cur_target.half().clone(),
                path=self.out / f"{self.counter:04d}_target.th",
            )
            paddle.save(
                obj=cur_mesh_pos.half().clone(),
                path=self.out / f"{self.counter:04d}_meshpos.th",
            )
            paddle.save(obj=cur_target_t0, path=self.out / f"{self.counter:04d}_t0.th")
            self.counter += 1
            start = end

    def _periodic_callback(
        self, model, trainer, trainer_model, batch_size, data_iter, **_
    ):
        self.counter = 0
        self.iterate_over_dataset(
            forward_fn=partial(
                self._forward, model=model, trainer=trainer, trainer_model=trainer_model
            ),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )
