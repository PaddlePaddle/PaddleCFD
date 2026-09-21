import numpy as np
import paddle
from ppcfd.models.climax.arch import ClimaX
from ppcfd.models.climax.pos_embed import get_1d_sincos_pos_embed_from_grid


class ClimaXClimateBench(ClimaX):
    def __init__(
        self,
        default_vars,
        out_vars,
        img_size=[32, 64],
        time_history=1,
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path=0.1,
        drop_rate=0.1,
        parallel_patch_embed=False,
        freeze_encoder=False,
    ):
        assert out_vars is not None
        super().__init__(
            default_vars,
            img_size,
            patch_size,
            embed_dim,
            depth,
            decoder_depth,
            num_heads,
            mlp_ratio,
            drop_path,
            drop_rate,
            parallel_patch_embed,
        )
        self.out_vars = out_vars
        self.time_history = time_history
        self.freeze_encoder = freeze_encoder
        self.time_pos_embed = paddle.nn.Parameter(
            paddle.zeros(1, time_history, embed_dim), requires_grad=True
        )
        self.time_agg = paddle.compat.nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.time_query = paddle.nn.Parameter(
            paddle.zeros(1, 1, embed_dim), requires_grad=True
        )
        time_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.time_pos_embed.shape[-1], np.arange(self.time_history)
        )
        self.time_pos_embed.data.copy_(
            paddle.from_numpy(time_pos_embed).float().unsqueeze(0)
        )
        self.head = paddle.compat.nn.Linear(embed_dim, img_size[0] * img_size[1])
        if freeze_encoder:
            for name, p in self.blocks.named_parameters():
                name = name.lower()
                if "norm" in name:
                    continue
                else:
                    p.requires_grad_(False)

    def forward_encoder(self, x: paddle.Tensor, lead_times: paddle.Tensor, variables):
        if isinstance(variables, list):
            variables = tuple(variables)
        b, t, _, _, _ = x.shape
        x = x.flatten(0, 1)
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)
        if self.parallel_patch_embed:
            x = self.token_embeds(x, var_ids)
        else:
            # var_ids values are pure Python constants (var_map dict lookup);
            # recompute them as ints so the to_static tracer can index the
            # ModuleList without a tensor->int conversion, which dy2static
            # cannot resolve (KeyError: pir Value as _sub_layers key).
            # Numerically equivalent: same values, same loop order.
            id_list = [int(self.var_map[var]) for var in variables]
            for i in range(len(id_list)):
                embeds.append(self.token_embeds[id_list[i]](x[:, i : i + 1]))
            x = paddle.stack(embeds, dim=1)
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)
        x = self.aggregate_variables(x)
        x = x + self.pos_embed
        x = x.unflatten(0, sizes=(b, t))
        x = x + self.time_pos_embed.unsqueeze(2)
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))
        lead_time_emb = lead_time_emb.unsqueeze(1).unsqueeze(2)
        x = x + lead_time_emb
        x = x.flatten(0, 1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.unflatten(0, sizes=(b, t))
        x = x.mean(-2)
        time_query = self.time_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.time_agg(time_query, x, x)
        return x

    def forward(self, x, y, lead_times, variables, out_variables, metric, lat):
        x = self.forward_encoder(x, lead_times, variables)
        preds = self.head(x)
        preds = preds.reshape(-1, 1, self.img_size[0], self.img_size[1])
        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]
        return loss, preds
