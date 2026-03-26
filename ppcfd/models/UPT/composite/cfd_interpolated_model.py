import einops
import paddle
import scipy
from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create


class CfdInterpolatedModel(CompositeModelBase):
    def __init__(self, latent, decoder, conditioner=None, **kwargs):
        super().__init__(**kwargs)
        common_kwargs = dict(
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )
        self.static_ctx[
            "grid_resolution"
        ] = self.data_container.get_dataset().grid_resolution
        self.static_ctx["ndim"] = len(self.data_container.get_dataset().grid_resolution)
        self.conditioner = create(conditioner, model_from_kwargs, **common_kwargs)
        self.latent = create(
            latent, model_from_kwargs, input_shape=self.input_shape, **common_kwargs
        )
        self.decoder = create(
            decoder,
            model_from_kwargs,
            **common_kwargs,
            input_shape=self.latent.output_shape,
            output_shape=self.output_shape
        )

    @property
    def submodels(self):
        return dict(
            **dict(conditioner=self.conditioner)
            if self.conditioner is not None
            else {},
            latent=self.latent,
            decoder=self.decoder
        )

    def forward(self, x, query_pos, timestep=None, velocity=None):
        outputs = {}
        if self.conditioner is not None:
            condition = self.conditioner(timestep=timestep, velocity=velocity)
        else:
            condition = None
        dynamics = self.latent(x, condition=condition)
        x_hat = self.decoder(dynamics, query_pos=query_pos)
        outputs["x_hat"] = x_hat
        return outputs

    @paddle.no_grad()
    def rollout(self, x, query_pos, velocity=None, num_rollout_timesteps=None):
        max_timesteps = self.data_container.get_dataset().getdim_timestep()
        num_rollout_timesteps = num_rollout_timesteps or max_timesteps
        assert 0 < num_rollout_timesteps <= max_timesteps
        dataset = self.data_container.get_dataset()
        x_linspace = paddle.linspace(0, dataset.max_x_pos, dataset.grid_resolution[1])
        y_linspace = paddle.linspace(0, dataset.max_y_pos, dataset.grid_resolution[0])
        grid_pos = paddle.meshgrid(x_linspace, y_linspace, indexing="xy")
        x_hats = []
        timestep = paddle.zeros(1, device=x.device, dtype=paddle.long)
        for _ in range(num_rollout_timesteps):
            outputs = self(x, query_pos=query_pos, timestep=timestep, velocity=velocity)
            x_hat = outputs["x_hat"]
            x_hats.append(x_hat)
            x_hat = einops.rearrange(
                x_hat,
                "(batch_size num_query_pos) dim -> batch_size num_query_pos dim",
                batch_size=len(query_pos),
            )
            interpolated = []
            for i in range(len(x_hat)):
                grid = paddle.from_numpy(
                    scipy.interpolate.griddata(
                        query_pos[i].cpu().unbind(1),
                        x_hat[i].cpu(),
                        grid_pos,
                        method="linear",
                        fill_value=0.0,
                    )
                ).float()
                interpolated.append(grid)
            x_hat = paddle.stack(interpolated).to(x.device)
            x = paddle.concat([x[(...), x_hat.size(-1) :], x_hat], dim=-1)
            timestep.add_(1)
        return paddle.stack(x_hats, dim=2)
