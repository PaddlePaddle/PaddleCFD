import paddle
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import Mlp, PerceiverBlock
from models.base.single_model_base import SingleModelBase
# from paddle_geometric.utils import unbatch

def paddle_unbatch(x, batch):
    """
    PaddlePaddle版本的unbatch函数
    将批处理的数据根据batch索引拆分成单个样本
    """
    if batch is None:
        return [x]
    
    # 获取每个样本的节点数量
    batch_size = int(batch.max()) + 1
    counts = paddle.bincount(batch, minlength=batch_size)
    
    # 拆分数据
    unbatched = []
    start = 0
    for count in counts:
        count = int(count)
        if count > 0:
            unbatched.append(x[start:start+count])
        else:
            unbatched.append(paddle.empty([0, x.shape[1]], dtype=x.dtype))
        start += count
    
    return unbatched


class RansPerceiver(SingleModelBase):
    def __init__(
        self,
        dim,
        num_attn_heads,
        init_weights="xavier_uniform",
        init_last_proj_zero=False,
        use_last_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_attn_heads = num_attn_heads
        self.use_last_norm = use_last_norm
        _, input_dim = self.input_shape
        self.proj = LinearProjection(input_dim, dim, init_weights=init_weights)
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=self.static_ctx["ndim"])
        self.query_mlp = Mlp(in_dim=dim, hidden_dim=dim, init_weights=init_weights)
        self.perceiver = PerceiverBlock(
            dim=dim,
            num_heads=num_attn_heads,
            init_last_proj_zero=init_last_proj_zero,
            init_weights=init_weights,
        )
        _, output_dim = self.output_shape
        self.norm = (
            paddle.nn.LayerNorm(dim, eps=1e-06)
            if use_last_norm
            else paddle.nn.Identity()
        )
        self.pred = LinearProjection(dim, output_dim, init_weights=init_weights)

    def forward(self, x, query_pos, unbatch_idx, unbatch_select):
        x = self.proj(x)
        query_pos_embed = self.pos_embed(query_pos)
        query = self.query_mlp(query_pos_embed)
        x = self.perceiver(q=query, kv=x)
        x = self.norm(x)
        x = self.pred(x)
        x = paddle.flatten(x, start_axis=0, stop_axis=1)
        unbatched = paddle_unbatch(x, batch=unbatch_idx)
        x = paddle.concat([unbatched[i] for i in unbatch_select])
        return x
