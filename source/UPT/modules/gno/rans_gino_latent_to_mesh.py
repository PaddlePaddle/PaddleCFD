import einops
import paddle
from torch_scatter import segment_csr

from .gino_grid_to_mesh import GinoGridToMesh


class RansGinoLatentToMesh(GinoGridToMesh):
    def forward(self, x, query_pos):
        assert query_pos.ndim == 3
        _, seqlen, _ = x.shape
        x = einops.rearrange(x, "batch_size seqlen dim -> (batch_size seqlen) dim")
        x = self.proj(x)
        query_pos = einops.rearrange(
            query_pos,
            "batch_size num_query_points ndim -> (batch_size num_query_points) ndim",
        )
        query_pos = self.pos_embed(query_pos)
        query_idx = paddle.arange(len(query_pos), device=x.device).repeat_interleave(
            seqlen
        )
        latent_idx = paddle.arange(seqlen, device=x.device).repeat(len(query_pos))
        x = paddle.concat([x[latent_idx], query_pos[query_idx]], dim=1)
        x = self.message(x)
        dst_indices, counts = query_idx.unique(return_counts=True)
        padded_counts = paddle.zeros(
            len(query_pos) + 1, device=counts.device, dtype=counts.dtype
        )
        padded_counts[dst_indices + 1] = counts
        indptr = padded_counts.cumsum(dim=0)
        x = segment_csr(src=x, indptr=indptr, reduce="mean")
        x = self.pred(x)
        return x
