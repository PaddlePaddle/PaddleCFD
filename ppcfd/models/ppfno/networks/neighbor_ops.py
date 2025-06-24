import sys
# sys.path.append("/home/")
# sys.path.append("/home/src/networks")

import paddle
import unittest
from typing import Optional
from .net_utils import MLP
from .utilities3 import paddle_memory_usage, memory_usage, num_of_nans, show_tensor_range

import fused_segment_csr

# with customized paddle-backended open3d operators
# import src.custom_ops.return_types as return_types
# from src.custom_ops.neighbor_search import FixedRadiusSearch
# NeighborSearchReturnType = return_types.open3d_fixed_radius_search

# with paddle-backended open3d
import open3d.ml.paddle as ml3d
from open3d.ml.paddle.layers import FixedRadiusSearch
NeighborSearchReturnType = ml3d.python.return_types.open3d_fixed_radius_search


class NeighborSearchLayer(paddle.nn.Layer):

    def __init__(self, radius: float):
        super().__init__()
        self.radius = radius
        self.nsearch = FixedRadiusSearch(return_distances=True)

    def forward(self, inp_positions: paddle.Tensor,
                out_positions: paddle.Tensor, return_distances=True) ->NeighborSearchReturnType:
        paddle.device.synchronize()
        neighbors = self.nsearch(inp_positions, out_positions, self.radius)
        paddle.device.synchronize()
        return neighbors


class NeighborMLPConvLayerWeighted(paddle.nn.Layer):

    def __init__(self, mlp=None, in_channels=8, hidden_dim=32, out_channels
        =32, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        if mlp is None:
            mlp = MLP([2 * in_channels, hidden_dim, out_channels], paddle.nn.GELU)
        self.mlp = mlp

    def forward(self, in_features: paddle.Tensor,
                neighbors: NeighborSearchReturnType,
                out_features: Optional[paddle.Tensor]=None,
                in_weights: Optional[paddle.Tensor]=None
        ) ->paddle.Tensor:
        """
        in_features: [N,C]
        out_features: [M,C]
        in_weights: [N]
        neighbors: ml3d.layers.FixedRadiusSearchResult.
        """
        if out_features is None:
            out_features = in_features
        assert (in_features.shape[1] + out_features.shape[1]
            == self.mlp.layers[0].weight.shape[0])
        neighbors_index_int64 = neighbors.neighbors_index.astype(paddle.int64)
        # rep_features = in_features[neighbors.neighbors_index.astype(paddle.int32)]
        # self_features = paddle.repeat_interleave(x=out_features, repeats=
        #     num_reps, axis=0)

        # rep_csr = segment_csr(
        #     rep_features, neighbors.neighbors_row_splits, reduce=self.reduction
        # )
        # del rep_features
        rep_csr = fused_segment_csr.select_segment_csr(in_features,
                                                       neighbors_index_int64,
                                                       neighbors.neighbors_row_splits,
                                                       reduce=self.reduction)
        # self_csr = segment_csr(
        #     self_features, neighbors.neighbors_row_splits, reduce=self.reduction
        # )
        if self.reduction == 'sum':
            rs = neighbors.neighbors_row_splits
            num_reps = rs[1:] - rs[:-1]
            self_csr = out_features * num_reps.reshape([-1, 1])
        elif self.reduction == 'mean':
            self_csr = out_features
        else:
            raise NotImplementedError

        # del self_features
        agg_csr = paddle.concat([rep_csr, self_csr], axis=1)
        del rep_csr
        del self_csr

        if in_weights is None:
            rep_weights = 1
            weights_csr = segment_csr(
                rep_weights, neighbors.neighbors_row_splits, reduce=self.reduction
            )
        else:
            # rep_weights = in_weights[neighbors.neighbors_index.astype(paddle.int32)].unsqueeze(axis=-1)
            weights_csr = fused_segment_csr.select_segment_csr(in_weights.unsqueeze(axis=-1),
                                                               neighbors_index_int64,
                                                               neighbors.neighbors_row_splits,
                                                               reduce=self.reduction)
        out_features = weights_csr * self.mlp(agg_csr)
        return out_features


class NeighborMLPConvLayer(paddle.nn.Layer):

    def __init__(self, mlp=None, in_channels=8, hidden_dim=32, out_channels
        =32, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        if mlp is None:
            mlp = MLP([2 * in_channels, hidden_dim, out_channels], paddle.nn.GELU)
        self.mlp = mlp

    def forward(self, in_features: paddle.Tensor, neighbors: NeighborSearchReturnType,
                out_features: Optional[paddle.Tensor]=None) ->paddle.Tensor:
        """
        inp_features: [N,C]
        outp_features: [M,C]
        neighbors: ml3d.layers.FixedRadiusSearchResult.
        """
        if out_features is None:
            out_features = in_features
        assert tuple(in_features.shape)[1] + tuple(out_features.shape)[1
            ] == self.mlp.layers[0].in_features
        rep_features = in_features[neighbors.neighbors_index.long()]
        rs = neighbors.neighbors_row_splits
        num_reps = rs[1:] - rs[:-1]
        self_features = paddle.repeat_interleave(x=out_features, repeats=
            num_reps, axis=0)
        agg_features = paddle.concat(x=[rep_features, self_features], axis=1)
        rep_features = self.mlp(agg_features)
        out_features = segment_csr(rep_features, neighbors.
            neighbors_row_splits, reduce=self.reduction)
        return out_features


class NeighborMLPConvLayerLinear(paddle.nn.Layer):

    def __init__(self, mlp=None, in_channels=8, hidden_dim=32, out_channels
        =32, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        if mlp is None:
            mlp = MLP([2 * in_channels, hidden_dim, out_channels], paddle.
                nn.GELU)
        self.mlp = mlp

    def forward(self, x_in: paddle.Tensor, neighbors:
        NeighborSearchReturnType, in_features: paddle.Tensor, x_out:
        Optional[paddle.Tensor]=None) ->paddle.Tensor:
        """
        inp_features: [N,C]
        outp_features: [M,C]
        neighbors: ml3d.layers.FixedRadiusSearchResult.
        """
        if x_out is None:
            x_out = x_in
        assert x_in.shape[1] + x_out.shape[1] == self.mlp.layers[0].weight.shape[0]
        neighbors_index_int64 = neighbors.neighbors_index.astype(paddle.int64)
        # rep_features = x_in[neighbors_index_int64]

        # agg_features = paddle.concat(x=[rep_features, self_features], axis=1)
        # del self_features

        # agg_features = segment_csr(agg_features, neighbors.
        #     neighbors_row_splits, reduce=self.reduction)

        rep_features = fused_segment_csr.select_segment_csr(x_in,
                                                            neighbors_index_int64,
                                                            neighbors.neighbors_row_splits,
                                                            reduce=self.reduction)

        # self_features = paddle.repeat_interleave(x=x_out, repeats=num_reps,
        #     axis=0)
        if self.reduction == 'sum':
            rs = neighbors.neighbors_row_splits
            num_reps = rs[1:] - rs[:-1]
            self_features = x_out * num_reps.reshape([-1, 1])
        elif self.reduction == 'mean':
            self_features = x_out
        else:
            raise NotImplementedError

        agg_features = paddle.concat(x=[rep_features, self_features], axis=1)

        rep_features = self.mlp(agg_features)
        del agg_features
        # in_features = in_features[neighbors_index_int64]
        # in_features = segment_csr(in_features, neighbors.
        #     neighbors_row_splits, reduce=self.reduction)
        in_features = fused_segment_csr.select_segment_csr(in_features.contiguous(),
                                                           neighbors_index_int64,
                                                           neighbors.neighbors_row_splits,
                                                           reduce=self.reduction)

        out_features = rep_features * in_features
        return out_features


def segment_mean_csr(src: paddle.Tensor, indptr: paddle.Tensor, out):
    num_seg = indptr.shape[0] - 1
    segment_ids = paddle.arange(num_seg)
    repeats = indptr[1:] - indptr[:-1]
    segment_ids = paddle.repeat_interleave(segment_ids, repeats,)
    res = paddle.geometric.segment_mean(src, segment_ids)

    if res.shape[0] < num_seg:
        zero = paddle.zeros([num_seg - res.shape[0], res.shape[1]])
        res = paddle.concat([res, zero], axis=0)
    res = paddle.reshape(res, [num_seg, res.shape[1]])
    return res


def segment_sum_csr(src: paddle.Tensor, indptr: paddle.Tensor, out):
    num_seg = indptr.shape[0] - 1
    segment_ids = paddle.arange(num_seg)
    repeats = indptr[1:] - indptr[:-1]
    segment_ids = paddle.repeat_interleave(segment_ids, repeats)
    res = paddle.geometric.segment_sum(src, segment_ids)

    if res.shape[0] < num_seg:
        zero = paddle.zeros([num_seg - res.shape[0], res.shape[1]])
        res = paddle.concat([res, zero], axis=0)
    res = paddle.reshape(res, [num_seg, -1])
    return res


def segment_csr(
    src: paddle.Tensor,
    indptr: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
    reduce: str = "sum",
) -> paddle.Tensor:
    if reduce == "mean":
        return segment_mean_csr(src, indptr, out)
    elif reduce == "sum":
        return segment_sum_csr(src, indptr, out)
    else:
        raise NotImplementedError


class TestNeighborSearch(unittest.TestCase):

    def setUp(self) ->None:
        self.N = 10000
        self.device = 'cuda:0'
        return super().setUp()

    def test_neighbor_search(self):
        inp_positions = paddle.randn(shape=[self.N, 3]).to(self.device) * 10
        inp_features = paddle.randn(shape=[self.N, 8]).to(self.device)
        out_positions = inp_positions
        neighbors = NeighborSearchLayer(1.2)(inp_positions, out_positions)
        pool = NeighborPoolingLayer(reduction='mean')
        out_features = pool(inp_features, neighbors)

    def test_mlp_conv(self):
        out_N = 1000
        radius = 1.2
        in_positions = paddle.randn(shape=[self.N, 3]).to(self.device) * 10
        out_positions = paddle.randn(shape=[out_N, 3]).to(self.device) * 10
        in_features = paddle.randn(shape=[self.N, 8]).to(self.device)
        out_features = paddle.randn(shape=[out_N, 8]).to(self.device)
        neighbors = NeighborSearchLayer(radius)(in_positions, out_positions)
        conv = NeighborMLPConvLayer(reduction='mean').to(self.device)
        out_features = conv(in_features, neighbors, out_features=out_features)


if __name__ == '__main__':
    unittest.main()
