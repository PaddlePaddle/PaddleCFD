import paddle

from kappamodules.functional.pos_embed import get_sincos_1d_from_seqlen
from kappamodules.init import init_xavier_uniform_zero_bias


class TimestepEmbed(paddle.nn.Layer):
    """https://github.com/facebookresearch/DiT/blob/main/models.py#L27C1-L64C21 but more performant"""

    def __init__(self, num_total_timesteps, embed_dim, hidden_dim=None):
        super().__init__()
        self.num_total_timesteps = num_total_timesteps
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim * 4
        self.register_buffer(
            "embed",
            get_sincos_1d_from_seqlen(seqlen=num_total_timesteps, dim=embed_dim),
        )
        self.mlp = paddle.nn.Sequential(
            paddle.compat.nn.Linear(embed_dim, hidden_dim),
            paddle.nn.SiLU(),
            paddle.compat.nn.Linear(hidden_dim, hidden_dim),
            paddle.nn.SiLU(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(init_xavier_uniform_zero_bias)

    def forward(self, timestep):
        assert timestep.size == len(timestep)
        timestep = timestep.flatten()
        embed = self.embed[timestep]
        return self.mlp(embed)
