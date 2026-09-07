import io
from functools import partial

import einops
import paddle
from callbacks.base.periodic_callback import PeriodicCallback
from kappadata.wrappers import ModeWrapper
from kappautils.images.png import png_writer_viridis
from kappautils.images.points_to_image import coords_to_image
from PIL import Image
from utils.formatting_util import dict_to_string


class OfflineCfdRolloutMeshGifCallback(PeriodicCallback):
    def __init__(
        self,
        dataset_key,
        resolution,
        num_rollout_timesteps=None,
        rollout_kwargs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.resolution = resolution
        self.num_rollout_timesteps = num_rollout_timesteps
        self.rollout_kwargs = rollout_kwargs or {}
        self.out = self.path_provider.stage_output_path / "rollout"
        self.dataset = None
        self.dataset_mode = None
        self.__config_id = None

    def _register_sampler_configs(self, trainer):
        self.dataset_mode = ModeWrapper.add_item(
            mode=trainer.dataset_mode, item="index"
        )
        self.__config_id = self._register_sampler_config_from_key(
            key=self.dataset_key, mode=self.dataset_mode
        )

    def _before_training(self, trainer, **kwargs):
        self.out.mkdir(exist_ok=True)
        self.dataset, _ = self.data_container.get_dataset(
            key=self.dataset_key, mode=self.dataset_mode
        )
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
        data = trainer_model.prepare(batch, dataset_mode=self.dataset_mode)
        batch, _ = batch
        index = ModeWrapper.get_item(mode=self.dataset_mode, item="index", batch=batch)
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
        x_hat = self.dataset.denormalize(x_hat, inplace=True)
        target = self.dataset.denormalize(target, inplace=True)
        x_hat = x_hat[:, 1:].norm(dim=1)
        target = target[:, 1:].norm(dim=1)
        delta = (x_hat - target).abs()
        for i in range(len(index)):
            prefix = f"{self.dataset_key}_{self.update_counter.cur_checkpoint}"
            if len(self.rollout_kwargs) > 0:
                prefix = f"{prefix}_{dict_to_string(self.rollout_kwargs, item_seperator='-')}"
            prefix = f"{prefix}_idx{index[i]:04d}"
            imgs = []
            for t in range(self.num_rollout_timesteps):
                img = self._tensor_to_pil_paddle(
                    data=paddle.stack([target[:, (t)], x_hat[:, (t)], delta[:, (t)]]),
                    progress=t / max(1, self.num_rollout_timesteps - 1),
                    pos=data["mesh_pos"],
                )
                imgs.append(img)
            uri = self.out / f"vmag_{prefix}.gif"
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
        self.logger.info(f"out: {self.out.as_posix()}")
        self.iterate_over_dataset(
            forward_fn=partial(
                self._forward, model=model, trainer=trainer, trainer_model=trainer_model
            ),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )
