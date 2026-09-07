import paddle

from .perceiver_block import PerceiverBlock


class PerceiverPoolingBlock(paddle.nn.Layer):
    """
    implementation inspired by
    https://github.com/lucidrains/flamingo-paddle/blob/main/flamingo_paddle/flamingo_paddle.py#L74
    """

    def __init__(
        self,
        dim,
        num_heads,
        num_query_tokens,
        perceiver_kwargs=None,
        init_query="mean0std1",
    ):
        super().__init__()
        self.init_query = init_query
        # self.query = paddle.nn.Parameter(paddle.empty(size=(num_query_tokens, dim)))
        self.query = self.create_parameter(
            shape=[num_query_tokens, dim],
            dtype='float32',
            # 推荐使用正态分布初始化，std=0.02 是这类 learnable query 的常用值
            default_initializer=paddle.nn.initializer.Normal(std=0.02)
        )
        self.perceiver = PerceiverBlock(
            dim=dim, num_heads=num_heads, **perceiver_kwargs or {}
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_query == "mean0std1":
            # paddle.nn.init.trunc_normal_(self.query)
            paddle.nn.initializer.TruncatedNormal(std=0.02)(self.query)
        else:
            raise NotImplementedError

    def forward(self, kv, attn_mask=None):
        query = self.query.expand((len(kv), -1, -1))
        return self.perceiver(q=query, kv=kv, attn_mask=attn_mask)
