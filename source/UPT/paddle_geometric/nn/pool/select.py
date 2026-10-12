from dataclasses import dataclass

import paddle


@dataclass
class SelectOutput:
    node_index: paddle.Tensor
    num_nodes: int
    cluster_index: paddle.Tensor
    num_clusters: int
    weight: paddle.Tensor | None = None


class Select(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        pass
