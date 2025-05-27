import sys

sys.path.append("/home/chenkai26/PaddleScience-AeroShapeOpt/paddle_project")
import paddle

import utils

"""
Python implementation of neighbor-search algorithm for use on CPU to avoid
breaking torch_cluster's CPU version.
"""


def simple_neighbor_search(data: paddle.Tensor, queries: paddle.Tensor, radius: float):
    """

    Parameters
    ----------
    Density-Based Spatial Clustering of Applications with Noise
    data : torch.Tensor
        vector of data points from which to find neighbors
    queries : torch.Tensor
        centers of neighborhoods
    radius : float
        size of each neighborhood
    """
    dists = paddle.cdist(x=queries, y=data).to(queries.place)
    in_nbr = paddle.where(condition=dists <= radius, x=1.0, y=0.0)
    nbr_indices = in_nbr.nonzero()[:, 1:].reshape(-1)
    nbrhd_sizes = paddle.cumsum(x=paddle.sum(x=in_nbr, axis=1), axis=0)
    splits = paddle.concat(
        x=(paddle.to_tensor(data=[0.0]).to(queries.place), nbrhd_sizes)
    )
    nbr_dict = {}
    nbr_dict["neighbors_index"] = nbr_indices.astype(dtype="int64").to(queries.place)
    nbr_dict["neighbors_row_splits"] = splits.astype(dtype="int64")
    return nbr_dict
