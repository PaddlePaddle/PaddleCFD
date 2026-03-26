import unittest

import torch

from kappamodules.attention import PerceiverAttention1d
from original_modules.original_perceiver_attention import OriginalPerceiverAttention


class TestPerceiverAttention(unittest.TestCase):
    def test_shape(self):
        torch.manual_seed(9823)
        dim = 24
        seqlen_q = 6
        seqlen_kv = 7
        attn = PerceiverAttention1d(dim=dim, num_heads=4)
        q = torch.randn(1, seqlen_q, dim)
        kv = torch.randn(1, seqlen_kv, dim)
        y = attn(q=q, kv=kv)
        self.assertEqual(q.shape, y.shape)

    def test_shape_masked(self):
        torch.manual_seed(9823)
        dim = 24
        seqlen_q = 6
        seqlen_kv = 7
        attn = PerceiverAttention1d(dim=dim, num_heads=4)
        q = torch.randn(1, seqlen_q, dim)
        kv = torch.randn(1, seqlen_kv, dim)
        attn_mask = torch.rand(1, seqlen_q, seqlen_kv) > 0.5
        y = attn(q=q, kv=kv, attn_mask=attn_mask)
        self.assertEqual(q.shape, y.shape)

    def test_equal_to_original(self):
        dim = 4
        seqlen_q = 3
        seqlen_kv = 2
        num_heads = 1
        torch.manual_seed(9823)
        attn_kc = PerceiverAttention1d(
            dim=dim,
            num_heads=num_heads,
            bias=False,
            init_weights="torch",
            concat_query_to_kv=True,
        )
        torch.manual_seed(9823)
        attn_og = OriginalPerceiverAttention(dim=dim, dim_head=dim // num_heads, heads=num_heads)
        torch.manual_seed(9823)
        q = torch.randn(1, seqlen_q, dim)
        kv = torch.randn(1, seqlen_kv, dim)
        y_kc = attn_kc(q=q, kv=kv)
        y_og = attn_og(latents=q, x=kv)
        self.assertEqual(q.shape, y_og.shape)
        self.assertEqual(y_kc.shape, y_og.shape)
        self.assertTrue(torch.allclose(y_kc, y_og))

    def test_init_last_proj_zero(self):
        attn = PerceiverAttention1d(dim=8, num_heads=4, init_last_proj_zero=True)
        q = torch.randn(2, 6, 8)
        kv = torch.randn(2, 7, 8)
        y = attn(q=q, kv=kv)
        self.assertTrue(torch.all(y == 0))
