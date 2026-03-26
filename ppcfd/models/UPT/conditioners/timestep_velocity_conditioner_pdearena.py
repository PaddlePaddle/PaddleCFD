import paddle
from kappamodules.functional.pos_embed import get_sincos_1d_from_seqlen
from kappamodules.init import (init_truncnormal_zero_bias,
                               init_xavier_uniform_zero_bias)
from kappamodules.layers import ContinuousSincosEmbed
from models.base.single_model_base import SingleModelBase


class TimestepVelocityConditionerPdearena(SingleModelBase):
    def __init__(self, dim, cond_dim=None, init_weights="xavier_uniform", **kwargs):
        super().__init__(**kwargs)
        self.num_total_timesteps = self.data_container.get_dataset().getdim_timestep()
        self.dim = dim
        self.cond_dim = cond_dim or dim * 4
        self.init_weights = init_weights
        self.static_ctx["condition_dim"] = self.cond_dim
        self.register_buffer(
            "timestep_embed",
            get_sincos_1d_from_seqlen(seqlen=self.num_total_timesteps, dim=dim),
        )
        self.velocity_embed = ContinuousSincosEmbed(dim=dim, ndim=1)
        self.timestep_mlp = paddle.nn.Sequential(
            paddle.compat.nn.Linear(dim, dim * 4),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(dim * 4, self.cond_dim),
            paddle.nn.GELU(),
        )
        self.velocity_mlp = paddle.nn.Sequential(
            paddle.compat.nn.Linear(dim, dim * 4),
            paddle.nn.GELU(),
            paddle.compat.nn.Linear(dim * 4, self.cond_dim),
            paddle.nn.GELU(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
        elif self.init_weights == "truncnormal":
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError

    def forward(self, timestep, velocity):
        assert timestep.size == len(timestep)
        assert velocity.size == len(velocity)
        timestep = timestep.flatten()
        velocity = velocity.view(-1, 1).float()
        if timestep.size == 1:
            timestep = timestep.repeat(velocity.size)
        timestep_embed = self.timestep_mlp(self.timestep_embed[timestep])
        velocity_embed = self.velocity_mlp(self.velocity_embed(velocity))
        return timestep_embed + velocity_embed
