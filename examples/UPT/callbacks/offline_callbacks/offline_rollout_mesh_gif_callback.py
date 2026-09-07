import io
import os
from functools import partial

import einops
import matplotlib.pyplot as plt
import paddle
import scipy
from callbacks.base.periodic_callback import PeriodicCallback
from kappadata.wrappers import ModeWrapper
from kappautils.images.png import png_writer_viridis
from kappautils.images.points_to_image import coords_to_image
from PIL import Image
from utils.formatting_util import dict_to_string
from utils.param_checking import to_2tuple


class OfflineRolloutMeshGifCallback(PeriodicCallback):
    def __init__(
        self,
        dataset_key,
        resolution,
        num_rollout_timesteps=None,
        rollout_kwargs=None,
        **kwargs,
    ):
        super().__init__(batch_size=1, **kwargs)
        self.dataset_key = dataset_key
        self.resolution = resolution
        self.num_rollout_timesteps = num_rollout_timesteps
        self.rollout_kwargs = rollout_kwargs or {}
        self.__config_id = None
        self.dataset_mode = None
        self.dataset = None
        self.out = None

    def _register_sampler_configs(self, trainer):
        self.dataset_mode = ModeWrapper.add_item(
            mode=trainer.dataset_mode, item="index"
        )
        self.dataset, _ = self.data_container.get_dataset(
            key=self.dataset_key, mode=self.dataset_mode
        )
        self.__config_id = self._register_sampler_config_from_key(
            key=self.dataset_key, mode=self.dataset_mode
        )

    def _before_training(self, trainer, **kwargs):
        self.out = self.path_provider.stage_output_path / "rollout"
        self.out.mkdir(exist_ok=True)
        if self.num_rollout_timesteps is None:
            self.num_rollout_timesteps = self.dataset.getdim_timestep()
        else:
            assert 0 < self.num_rollout_timesteps <= self.dataset.getdim_timestep()

    def _tensor_to_pil_paddle(self, data, progress, pos):
        data = paddle.stack(
            [
                coords_to_image(coords=pos, resolution=self.resolution, weights=data[i])
                for i in range(3)
            ]
        )
        data_min = data.flatten(start_dim=1).min(axis=1)
        data -= data_min.view(-1, 1, 1)
        data_max = data.flatten(start_dim=1).max(axis=1)
        data /= data_max.view(-1, 1, 1)
        data = einops.rearrange(data, "three height width -> (three height) width")
        progress_tensor = paddle.zeros(
            size=(1, data.size(1)), dtype=data.dtype, device=data.device
        )
        progress_tensor[:, : round(progress * data.size(1))] = 1
        data = paddle.concat([progress_tensor, data])
        with io.BytesIO() as buffer:
            png_writer_viridis(data.unsqueeze(0), buffer, save_format="png")
            buffer.seek(0)
            img = Image.open(buffer)
            pil = img.convert("RGB")
        return pil

    def _forward(self, batch, model, trainer, trainer_model):
        data = trainer_model.prepare(
            batch, dataset_mode=self.dataset_mode, mode="rollout"
        )
        batch, ctx = batch
        idx = ModeWrapper.get_item(mode=self.dataset_mode, item="index", batch=batch)
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
        trajectories = paddle.concat([ground_truth, predictions])
        del ground_truth
        del predictions
        trajectories = self.dataset.denormalize(trajectories, inplace=True, dim=2)
        denormed_ground_truth, denormed_predictions = trajectories.chunk(2)
        denormalized_deltas = (denormed_ground_truth - denormed_predictions).abs()
        trajectories = paddle.concat([trajectories, denormalized_deltas])
        if "mesh_pos" in data:
            pos = data["mesh_pos"]
        elif "pos" in data:
            pos = data["pos"]
        else:
            raise NotImplementedError
        for i, trajectory in enumerate([trajectories]):
            prefix = f"{self.dataset_key}_{self.update_counter.cur_checkpoint}"
            if len(self.rollout_kwargs) > 0:
                prefix = f"{prefix}_{dict_to_string(self.rollout_kwargs, item_seperator='-')}"
            prefix = f"{prefix}_idx{idx[i]:04d}"
            velocity = trajectory[:, :, 1:]
            velocity_magnitude = paddle.sqrt(paddle.sum(velocity**2, dim=2))
            self.logger.info(f"generating vmag images")
            velocity_magnitude = einops.rearrange(
                velocity_magnitude,
                "(three num_points) num_rollout_timesteps -> num_rollout_timesteps three num_points",
                three=3,
            )
            imgs = [
                self._tensor_to_pil_paddle(
                    velocity_magnitude[j],
                    progress=j / max(1, len(velocity_magnitude) - 1),
                    pos=pos,
                )
                for j in range(len(velocity_magnitude))
            ]
            uri = self.out / f"vmag_{prefix}.gif"
            self.logger.info(f"generating vmag gif '{uri.as_posix()}'")
            imgs[0].save(
                fp=uri,
                format="GIF",
                append_images=imgs[1:],
                save_all=True,
                duration=100,
                loop=0,
            )

    def _periodic_callback(
        self, model, trainer, trainer_model, batch_size, data_iter, **_
    ):
        self.iterate_over_dataset(
            forward_fn=partial(
                self._forward, model=model, trainer=trainer, trainer_model=trainer_model
            ),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )
