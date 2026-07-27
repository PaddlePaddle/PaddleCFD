import io
import os
from functools import partial
from datasets.collators.cfd_simformer_collator import CfdSimformerCollator
import einops
import matplotlib.pyplot as plt
import paddle
import numpy as np
import scipy
from callbacks.base.periodic_callback import PeriodicCallback
from kappadata.wrappers import ModeWrapper
from kappautils.images.png import png_writer_viridis
from kappautils.images.points_to_image import coords_to_image
from PIL import Image
from utils.formatting_util import dict_to_string
from utils.param_checking import to_2tuple
from paddle.vision.transforms import to_tensor


def paddle_to_pil(tensor):
    # 1. 将 Tensor 移动到 CPU 并转为 NumPy
  
    img_array = tensor.numpy()
    
    if len(img_array.shape) == 3:
        img_array = img_array.transpose((1, 2, 0))
    
    # 3. 如果数据是 0.0 - 1.0 之间，需转回 0-255 整数
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
        
    return Image.fromarray(img_array)

def paddle_default_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class OfflineRolloutMeshCallback(PeriodicCallback):
    def __init__(
        self,
        dataset_key,
        num_rollout_timesteps=None,
        use_teacher_forcing=False,
        rollout_kwargs=None,
        resolution=None,
        save_gif=False,
        save_pngs=False,
        save_plots=False,
        visualize_pressure=False,
        visualize_velocities=False,
        visualize_velocity_magnitude=True,
        duration_per_frame=100,
        visualization_backend="paddle",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.__config_id = None
        self.out = None
        self.dataset = None
        self.num_rollout_timesteps = num_rollout_timesteps
        self.use_teacher_forcing = use_teacher_forcing
        self.rollout_kwargs = rollout_kwargs or {}
        self.resolution = resolution
        self.save_gif = save_gif
        self.save_pngs = save_pngs
        self.save_plots = save_plots
        self.visualize_pressure = visualize_pressure
        self.visualize_velocities = visualize_velocities
        self.visualize_velocity_magnitude = visualize_velocity_magnitude
        self.duration_per_frame = duration_per_frame
        self.visualization_backend = visualization_backend

    def _before_training(self, **kwargs):
        if os.name == "nt" and "KMP_DUPLICATE_LIB_OK" not in os.environ:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.out = self.path_provider.stage_output_path / "rollout"
        self.out.mkdir(exist_ok=True)
        (self.out / "gifs").mkdir(exist_ok=True)
        if self.save_pngs:
            (self.out / "pngs").mkdir(exist_ok=True)
        if self.save_plots:
            (self.out / "plots").mkdir(exist_ok=True)
        self.dataset, collator = self.data_container.get_dataset(
            key=self.dataset_key, mode=self.dataset_mode
        )
        assert isinstance(
            collator.collator, CfdSimformerCollator,
        )
        if self.num_rollout_timesteps is None:
            self.num_rollout_timesteps = self.dataset.getdim_timestep()
        else:
            assert 0 < self.num_rollout_timesteps <= self.dataset.getdim_timestep()

    @property
    def dataset_mode(self):
        return "index pos edge_index x geometry2d velocity"

    def _register_sampler_configs(self, trainer):
        self.__config_id = self._register_sampler_config_from_key(
            key=self.dataset_key, mode=self.dataset_mode
        )

    @staticmethod
    def _tensor_to_pil_matplotlib(data, progress, pos):
        y, x = pos.cpu().unbind(1)
        data = data.cpu()
        with io.BytesIO() as buffer:
            images = []
            for i in range(3):
                buffer.seek(0)
                plt.scatter(x, y, s=0.01, c=data[i].cpu(), cmap="viridis")
                plt.axis("off")
                plt.xlim(0, 300)
                plt.ylim(0, 200)
                plt.savefig(buffer, bbox_inches="tight", format="jpg")
                images.append(to_tensor(Image.open(buffer)))
        data = paddle.concat(images, dim=1).to(pos.device)
        progress_tensor = paddle.zeros(
            size=(data.size(0), 1, data.size(2)), dtype=data.dtype, device=data.device
        )
        progress_tensor[:, :, : round(progress * data.size(2))] = 1
        data = paddle.concat([progress_tensor, data], dim=1)
        return paddle_to_pil(data)
    
    def _tensor_to_pil_paddle(self, data, progress, pos):
        data = paddle.stack(
            [
                coords_to_image(coords=pos, resolution=self.resolution, weights=data[i])
                for i in range(3)
            ]
        )
        data_min = (
            data.flatten(start_dim=1).min(axis=1),
            data.flatten(start_dim=1).argmin(axis=1),
        ).values
        data -= data_min.view(-1, 1, 1)
        data_max = (
            data.flatten(start_dim=1).max(axis=1),
            data.flatten(start_dim=1).argmax(axis=1),
        ).values
        data /= data_max.view(-1, 1, 1)
        data = einops.rearrange(data, "three height width -> (three height) width")
        progress_tensor = paddle.zeros(
            size=(1, data.size(1)), dtype=data.dtype, device=data.device
        )
        progress_tensor[:, : round(progress * data.size(1))] = 1
        data = paddle.concat([progress_tensor, data])
        temp_out = (
            self.path_provider.get_temp_path() / f"{self.path_provider.stage_id}.png"
        )
        png_writer_viridis(data.unsqueeze(0), temp_out)
        pil = paddle_default_loader(temp_out)
        return pil

    def tensor_to_pil(self, data, progress, pos):
        if self.visualization_backend == "matplotlib":
            return self._tensor_to_pil_matplotlib(data=data, progress=progress, pos=pos)
        if self.visualization_backend == "paddle":
            return self._tensor_to_pil_paddle(data=data, progress=progress, pos=pos)
        raise NotImplementedError

    def visualize(self, idx, trajectories, deltas, pos):
        if (
            sum(
                [
                    self.visualize_pressure,
                    self.visualize_velocities,
                    self.visualize_velocity_magnitude,
                ]
            )
            == 0
        ):
            return
        if sum([self.save_gif, self.save_pngs]) == 0:
            return
        assert idx.size == 1, "only batchsize=1 is supported for now"
        trajectories = paddle.concat([trajectories, deltas])
        for i, trajectory in enumerate([trajectories]):
            prefix = f"{self.dataset_key}_{self.update_counter.cur_checkpoint}_{self.visualization_backend}"
            if len(self.rollout_kwargs) > 0:
                prefix = f"{prefix}_{dict_to_string(self.rollout_kwargs, item_seperator='-')}"
            if self.use_teacher_forcing:
                prefix = f"{prefix}_tforced"
            prefix = f"{prefix}_idx{idx[i]:04d}"
            data = {}
            if self.visualize_pressure:
                data["pressure"] = trajectory[:, :, (0)]
            if self.visualize_velocities:
                data["v0"] = trajectory[:, :, (1)]
                data["v1"] = trajectory[:, :, (2)]
            if self.visualize_velocity_magnitude:
                velocity = trajectory[:, :, 1:]
                velocity_magnitude = paddle.sqrt(paddle.sum(velocity**2, dim=2))
                data["vmag"] = velocity_magnitude
            for name, item in data.items():
                self.logger.info(f"generating {name} images")
                item = einops.rearrange(
                    item,
                    "(three num_points) num_rollout_timesteps -> num_rollout_timesteps three num_points",
                    three=3,
                )
                imgs = [
                    self.tensor_to_pil(
                        item[j], progress=j / max(1, len(item) - 1), pos=pos
                    )
                    for j in range(len(item))
                ]
                if self.save_gif:
                    uri = self.out / "gifs" / f"{name}_{prefix}.gif"
                    self.logger.info(f"generating {name} gif '{uri.as_posix()}'")
                    imgs[0].save(
                        fp=uri,
                        format="GIF",
                        append_images=imgs[1:],
                        save_all=True,
                        duration=self.duration_per_frame,
                        loop=0,
                    )
                if self.save_pngs:
                    self.logger.info(f"storing individual {name} pngs")
                    for j, img in enumerate(imgs):
                        img.save(self.out / "pngs" / f"{name}_{prefix}_ts{j:04d}.png")

    def _forward(self, batch, model, trainer):
        batch, ctx = batch
        idx = ModeWrapper.get_item(mode=self.dataset_mode, item="index", batch=batch)
        x = ModeWrapper.get_item(mode=self.dataset_mode, item="x", batch=batch)
        geometry2d = ModeWrapper.get_item(
            mode=self.dataset_mode, item="geometry2d", batch=batch
        )
        geometry2d = geometry2d.to(model.device, non_blocking=True)
        velocity = ModeWrapper.get_item(
            mode=self.dataset_mode, item="velocity", batch=batch
        )
        velocity = velocity.to(model.device, non_blocking=True)
        pos = ModeWrapper.get_item(mode=self.dataset_mode, item="pos", batch=batch)
        pos = pos.to(model.device, non_blocking=True)
        padded_pos = ctx["padded_pos"].to(model.device, non_blocking=True)
        batch_idx = ctx["batch_idx"].to(model.device, non_blocking=True)
        unbatch_idx = ctx["unbatch_idx"].to(model.device, non_blocking=True)
        unbatch_select = ctx["unbatch_select"].to(model.device, non_blocking=True)
        edge_index = ModeWrapper.get_item(
            mode=self.dataset_mode, item="edge_index", batch=batch
        )
        edge_index = edge_index.to(model.device, non_blocking=True)
        assert (
            x.ndim == 3
        ), "expected data to be of shape (bs * num_points, num_total_timesteps + 1, num_channels)"
        if x.size(1) != self.num_rollout_timesteps + 1:
            x = x[:, : self.num_rollout_timesteps + 1]
        x = x.to(model.device, non_blocking=True)
        with trainer.autocast_context:
            if self.use_teacher_forcing:
                assert self.num_rollout_timesteps + 1 == x.size(1)
                predictions = model.rollout_teacher_forced(
                    x=x,
                    geometry2d=geometry2d,
                    velocity=velocity,
                    pos=pos,
                    padded_pos=padded_pos,
                    batch_idx=batch_idx,
                    unbatch_idx=unbatch_idx,
                    unbatch_select=unbatch_select,
                    edge_index=edge_index,
                    num_rollout_timesteps=self.num_rollout_timesteps,
                    **self.rollout_kwargs,
                )
            else:
                predictions = model.rollout(
                    x0=x[:, (0)],
                    geometry2d=geometry2d,
                    velocity=velocity,
                    pos=pos,
                    padded_pos=padded_pos,
                    batch_idx=batch_idx,
                    unbatch_idx=unbatch_idx,
                    unbatch_select=unbatch_select,
                    edge_index=edge_index,
                    num_rollout_timesteps=self.num_rollout_timesteps,
                    **self.rollout_kwargs,
                )
        ground_truth = x[:, 1 : 1 + self.num_rollout_timesteps]
        trajectories = paddle.concat([ground_truth, predictions])
        del ground_truth
        del predictions
        trajectories = einops.rearrange(
            trajectories,
            "num_points num_timesteps num_channels -> num_timesteps num_channels num_points",
        )
        trajectories = self.dataset.denormalize(trajectories, inplace=True)
        trajectories = einops.rearrange(
            trajectories,
            "num_timesteps num_channels num_points -> num_points num_timesteps num_channels",
        )
        denormed_ground_truth, denormed_predictions = trajectories.chunk(2)
        denormalized_deltas = (denormed_ground_truth - denormed_predictions).abs()
        self.visualize(
            idx=idx, trajectories=trajectories, deltas=denormalized_deltas, pos=pos
        )
        results = dict(
            overall_denormalized_delta=denormalized_deltas.flatten(start_dim=1).mean(
                dim=-1
            )
        )
        if self.save_plots:
            raise NotImplementedError
        return results

    def _periodic_callback(self, model, trainer, batch_size, data_iter, **_):
        results = self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer),
            config_id=self.__config_id,
            batch_size=batch_size,
            data_iter=data_iter,
        )
        metric_identifier = f"{self.dataset_key}/0to{self.num_rollout_timesteps}"
        if len(self.rollout_kwargs) > 0:
            metric_identifier = (
                f"{metric_identifier}/{dict_to_string(self.rollout_kwargs)}"
            )
        if self.use_teacher_forcing:
            metric_identifier = f"{metric_identifier}/tforced"
        self.writer.add_scalar(
            key=f"delta/{metric_identifier}/overall/denormalized",
            value=results["overall_denormalized_delta"].mean(),
            logger=self.logger,
            format_str=".10f",
        )
        if self.save_plots:
            raise NotImplementedError
