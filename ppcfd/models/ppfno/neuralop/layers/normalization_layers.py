import sys
# sys.path.append('/home/chenkai26/PaddleScience-AeroShapeOpt/paddle_project')
from .. import utils
import paddle


class AdaIN(paddle.nn.Layer):

    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-05):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps
        if mlp is None:
            mlp = paddle.nn.Sequential(paddle.nn.Linear(in_features=
                embed_dim, out_features=512), paddle.nn.GELU(), paddle.nn.
                Linear(in_features=512, out_features=2 * in_channels))
        self.mlp = mlp
        self.embedding = None

    def set_embedding(self, x):
        self.embedding = x.reshape(self.embed_dim)

    def forward(self, x):
        assert self.embedding is not None, 'AdaIN: update embeddding before running forward'
        weight, bias = utils.split(x=self.mlp(self.embedding),
            num_or_sections=self.in_channels, axis=0)
        return paddle.nn.functional.group_norm(x=x, num_groups=self.
            in_channels, weight=weight, bias=bias, epsilon=self.eps)
