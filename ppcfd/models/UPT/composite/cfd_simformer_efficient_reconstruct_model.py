import einops
import paddle
from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create


class CfdSimformerEfficientReconstructModel(CompositeModelBase):
    def __init__(
        self,
        encoder,
        latent,
        decoder,
        conditioner=None,
        geometry_encoder=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        common_kwargs = dict(
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )
        self.conditioner = create(
            conditioner,
            model_from_kwargs,
            **common_kwargs,
            input_shape=self.input_shape
        )
        self.geometry_encoder = create(
            geometry_encoder, model_from_kwargs, **common_kwargs
        )
        if self.geometry_encoder is not None:
            assert (
                self.geometry_encoder.output_shape is not None
                and len(self.geometry_encoder.output_shape) == 2
            )
            self.static_ctx["num_static_tokens"] = self.geometry_encoder.output_shape[0]
        else:
            self.static_ctx["num_static_tokens"] = 0
        if self.conditioner is not None:
            self.static_ctx["dim"] = self.conditioner.dim
        elif self.geometry_encoder is not None:
            self.static_ctx["dim"] = self.geometry_encoder.output_shape[1]
        else:
            self.static_ctx["dim"] = latent["kwargs"]["dim"]
        self.encoder = create(
            encoder, model_from_kwargs, input_shape=self.input_shape, **common_kwargs
        )
        assert self.encoder.output_shape is not None
        self.latent = create(
            latent,
            model_from_kwargs,
            input_shape=self.encoder.output_shape,
            **common_kwargs
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
            **dict(geometry_encoder=self.geometry_encoder)
            if self.geometry_encoder is not None
            else {},
            encoder=self.encoder,
            latent=self.latent,
            decoder=self.decoder
        )

    def forward(
        self,
        x,
        geometry2d,
        timestep,
        velocity,
        mesh_pos,
        query_pos,
        mesh_edges,
        batch_idx,
        unbatch_idx,
        unbatch_select,
        reconstruction_input,
        reconstruction_pos,
        detach_reconstructions=True,
        reconstruct_prev_x=False,
        reconstruct_dynamics=False,
    ):
        outputs = {}
        if self.conditioner is not None:
            condition = self.conditioner(timestep=timestep, velocity=velocity)
        else:
            condition = None
        if self.geometry_encoder is not None:
            static_tokens = self.geometry_encoder(geometry2d)
            outputs["static_tokens"] = static_tokens
        else:
            static_tokens = None
        prev_dynamics = self.encoder(
            x,
            mesh_pos=mesh_pos,
            mesh_edges=mesh_edges,
            batch_idx=batch_idx,
            condition=condition,
            static_tokens=static_tokens,
        )
        outputs["prev_dynamics"] = prev_dynamics
        dynamics = self.latent(
            prev_dynamics, condition=condition, static_tokens=static_tokens
        )
        outputs["dynamics"] = dynamics
        x_hat = self.decoder(
            dynamics,
            query_pos=query_pos,
            unbatch_idx=unbatch_idx,
            unbatch_select=unbatch_select,
            condition=condition,
            static_tokens=static_tokens,
        )
        outputs["x_hat"] = x_hat
        if reconstruct_dynamics:
            next_timestep = paddle.clip(
                x=timestep + 1, max=self.conditioner.num_total_timesteps - 1
            )
            next_condition = self.conditioner(timestep=next_timestep, velocity=velocity)
            num_output_channels = x_hat.size(1)
            dynamics_hat = self.encoder(
                paddle.concat(
                    [x[:, num_output_channels:], reconstruction_input], dim=1
                ),
                mesh_pos=mesh_pos,
                mesh_edges=mesh_edges,
                batch_idx=batch_idx,
                condition=next_condition,
                static_tokens=static_tokens,
            )
            outputs["dynamics_hat"] = dynamics_hat
        if reconstruct_prev_x:
            prev_timestep = paddle.nn.functional.relu(x=timestep - 1)
            prev_condition = self.conditioner(timestep=prev_timestep, velocity=velocity)
            prev_x_hat = self.decoder(
                prev_dynamics.detach() if detach_reconstructions else prev_dynamics,
                query_pos=query_pos,
                unbatch_idx=unbatch_idx,
                unbatch_select=unbatch_select,
                condition=prev_condition,
                static_tokens=static_tokens,
            )
            outputs["prev_x_hat"] = prev_x_hat
        return outputs

    @paddle.no_grad()
    def rollout(
        self,
        x,
        geometry2d,
        velocity,
        mesh_pos,
        query_pos,
        mesh_edges,
        batch_idx,
        unbatch_idx,
        unbatch_select,
        num_rollout_timesteps=None,
        mode="image",
        clip=None,
    ):
        max_timesteps = self.data_container.get_dataset().getdim_timestep()
        num_rollout_timesteps = num_rollout_timesteps or max_timesteps
        assert 0 < num_rollout_timesteps <= max_timesteps
        x_hats = []
        timestep = paddle.zeros(1, device=x.device, dtype=paddle.long)
        condition = None
        outputs = self(
            x,
            geometry2d=geometry2d,
            velocity=velocity,
            timestep=timestep,
            mesh_pos=mesh_pos,
            query_pos=query_pos,
            mesh_edges=mesh_edges,
            batch_idx=batch_idx,
            unbatch_idx=unbatch_idx,
            unbatch_select=unbatch_select,
        )
        x_hat = outputs["x_hat"]
        x_hats.append(x_hat)
        static_tokens = outputs.get("static_tokens", None)
        if mode == "latent":
            dynamics = outputs["dynamics"]
            for _ in range(num_rollout_timesteps - 1):
                if self.conditioner is not None:
                    timestep.add_(1)
                    condition = self.conditioner(timestep=timestep, velocity=velocity)
                dynamics = self.latent(
                    dynamics, condition=condition, static_tokens=static_tokens
                )
                x_hat = self.decoder(
                    dynamics,
                    query_pos=query_pos,
                    unbatch_idx=unbatch_idx,
                    unbatch_select=unbatch_select,
                    condition=condition,
                    static_tokens=static_tokens,
                )
                if clip is not None:
                    x_hat = x_hat.clip(-clip, clip)
                x_hats.append(x_hat)
        elif mode == "image":
            for _ in range(num_rollout_timesteps - 1):
                x = paddle.concat([x[:, x_hat.size(1) :], x_hat], dim=1)
                timestep.add_(1)
                outputs = self(
                    x,
                    geometry2d=geometry2d,
                    velocity=velocity,
                    timestep=timestep,
                    mesh_pos=mesh_pos,
                    query_pos=query_pos,
                    mesh_edges=mesh_edges,
                    batch_idx=batch_idx,
                    unbatch_idx=unbatch_idx,
                    unbatch_select=unbatch_select,
                )
                x_hat = outputs["x_hat"]
                if clip is not None:
                    x_hat = x_hat.clip(-clip, clip)
                x_hats.append(x_hat)
        else:
            raise NotImplementedError
        return paddle.stack(x_hats, dim=2)
