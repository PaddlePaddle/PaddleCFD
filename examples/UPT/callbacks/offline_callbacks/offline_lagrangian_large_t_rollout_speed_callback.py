from functools import partial

import einops
import kappaprofiler as kp
import paddle
from callbacks.base.periodic_callback import PeriodicCallback
from kappadata.wrappers import ModeWrapper
from paddle_geometric.utils import scatter
from utils.formatting_util import dict_to_string


class OfflineLagrangianLargeTRolloutSpeedCallback(PeriodicCallback):
    def __init__(
        self, dataset_key, num_rollout_timesteps=None, rollout_kwargs=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.num_rollout_timesteps = num_rollout_timesteps
        self.rollout_kwargs = rollout_kwargs or {}
        self.out = self.path_provider.stage_output_path / "rollout"
        self.__config_id = None
        bounds = paddle.tensor(self.data_container.get_dataset().metadata["bounds"])
        self.box = bounds[:, (1)] - bounds[:, (0)]

    def _before_training(self, trainer, **kwargs):
        self.out.mkdir(exist_ok=True)
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
        batch, ctx = batch
        x = ModeWrapper.get_item(mode=trainer.dataset_mode, item="x", batch=batch)
        x = x.to(model.device)
        all_pos = ModeWrapper.get_item(
            mode=trainer.dataset_mode, item="all_pos", batch=batch
        )
        all_pos = all_pos.to(model.device)
        all_vel = ModeWrapper.get_item(
            mode=trainer.dataset_mode, item="all_vel", batch=batch
        )
        all_vel = all_vel.to(model.device)
        if (
            "const_timestep" in trainer.forward_kwargs
            and trainer.forward_kwargs["const_timestep"]
        ):
            timestep = None
        else:
            timestep = ModeWrapper.get_item(
                mode=trainer.dataset_mode, item="timestep", batch=batch
            )
            timestep = timestep.to(model.device)
        edge_index = ModeWrapper.get_item(
            mode=trainer.dataset_mode, item="edge_index", batch=batch
        )
        edge_index = edge_index.to(model.device)
        batch_idx = ctx["batch_idx"].to(model.device)
        x = einops.rearrange(
            x, "a num_input_timesteps dim -> a (num_input_timesteps dim)"
        )
        unbatch_idx = ctx["unbatch_idx"].to(model.device)
        unbatch_select = ctx["unbatch_select"].to(model.device)
        with trainer.autocast_context:
            vel_pred = model.rollout_large_t_timing(
                x=x,
                all_pos=all_pos,
                timestep=timestep,
                edge_index=edge_index,
                batch_idx=batch_idx,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select,
            )
        return vel_pred.mean().item()

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
