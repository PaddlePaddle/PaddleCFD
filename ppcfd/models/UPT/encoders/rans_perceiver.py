import paddle
from kappamodules.layers import ContinuousSincosEmbed
from kappamodules.transformer import Mlp, PerceiverPoolingBlock
from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import \
    ExcludeFromWdByNameModifier
# from paddle_geometric.utils import to_dense_batch

def paddle_to_dense_batch(x, batch, fill_value=0, max_num_nodes=None):
    """
    Paddle 版 to_dense_batch
    将 [Total_Nodes, Dim] 的稀疏节点特征转换为 [Batch, Max_Nodes, Dim] 的密集张量
    """
    # 1. 计算每个 Batch 的节点数量
    # 统计 batch 中每个索引出现的次数
    batch_size = int(batch.max()) + 1
    
    # 获取每个样本的节点数
    num_nodes = paddle.bincount(batch, minlength=batch_size)
    
    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    # 2. 准备输出张量
    dim = x.shape[-1]
    out = paddle.full([batch_size, max_num_nodes, dim], fill_value, dtype=x.dtype)
    
    # 3. 计算每个节点在所属 batch 内部的偏移索引 (模拟 PyG 的逻辑)
    # 这步比较核心：生成每个节点在其 batch 内的 [0, 1, 2...] 索引
    # 利用 cumsum 来计算偏移
    cum_nodes = paddle.concat([paddle.zeros([1], dtype='int64'), paddle.cumsum(num_nodes)[:-1]])
    idx_in_batch = paddle.arange(len(batch)) - paddle.gather(cum_nodes, batch)

    # 4. 填充数据
    # 构造 scatter 的索引: [batch_id, node_id_in_batch]
    mask = idx_in_batch < max_num_nodes # 过滤超过 max_num_nodes 的点
    # 如果有过滤，需要筛选 x 和对应的索引
    safe_batch = paddle.masked_select(batch, mask)
    safe_idx_in_batch = paddle.masked_select(idx_in_batch, mask)
    safe_x = paddle.masked_select(x, mask.unsqueeze(-1)).reshape([-1, dim])
    
    # 构造坐标索引进行填充
    indices = paddle.stack([safe_batch, safe_idx_in_batch], axis=1)
    out = paddle.scatter_nd_add(out, indices, safe_x)

    # 5. 生成 Mask (True 表示真实存在节点的位置)
    out_mask = paddle.arange(max_num_nodes).unsqueeze(0) < num_nodes.unsqueeze(1)
    
    return out, out_mask

class RansPerceiver(SingleModelBase):
    def __init__(
        self,
        dim,
        num_attn_heads,
        num_output_tokens,
        add_type_token=False,
        init_weights="xavier_uniform",
        init_last_proj_zero=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_attn_heads = num_attn_heads
        self.num_output_tokens = num_output_tokens
        self.add_type_token = add_type_token
        _, ndim = self.input_shape
        self.static_ctx["ndim"] = ndim
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        self.mlp = Mlp(in_dim=dim, hidden_dim=dim * 4, init_weights=init_weights)
        self.block = PerceiverPoolingBlock(
            dim=dim,
            num_heads=num_attn_heads,
            num_query_tokens=num_output_tokens,
            perceiver_kwargs=dict(
                init_weights=init_weights, init_last_proj_zero=init_last_proj_zero
            ),
        )
        if add_type_token:
            # self.type_token = paddle.nn.Parameter(paddle.empty(size=(1, 1, dim)))
            self.type_token = self.create_parameter(
            shape=[1, 1, dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Normal(std=0.02)
        )
        else:
            self.type_token = None
        self.output_shape = num_output_tokens, dim

    def model_specific_initialization(self):
        if self.add_type_token:
            paddle.nn.initializer.TruncatedNormal(std=0.02)(self.type_token)

    def get_model_specific_param_group_modifiers(self):
        modifiers = [ExcludeFromWdByNameModifier(name="block.query")]
        if self.add_type_token:
            modifiers += [ExcludeFromWdByNameModifier(name="type_token")]
        return modifiers

    def forward(self, mesh_pos, batch_idx, mesh_edges=None):
        x = self.pos_embed(mesh_pos)
        x, mask = paddle_to_dense_batch(x, batch_idx)
        if paddle.all(mask):
            mask = None
        else:
            mask = mask.unsqueeze(1).unsqueeze(1)
        x = self.mlp(x)
        x = self.block(kv=x, attn_mask=mask)
        if self.add_type_token:
            x = x + self.type_token
        return x
