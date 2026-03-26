import unittest

import torch

from kappamodules.attention import MMDiTDotProductAttention, DotProductAttention1d


class TestMmditDotProductAttention(unittest.TestCase):
    def test_equal_dotproductattention(self):
        dim = 16
        torch.manual_seed(98234)
        attn1 = MMDiTDotProductAttention(
            dim=dim,
            num_heads=2,
            num_modalities=1,
            qkv_bias=True,
            init_weights="truncnormal002",
            channel_first=False,
        )
        torch.manual_seed(98234)
        attn2 = DotProductAttention1d(
            dim=dim,
            num_heads=2,
            qkv_bias=True,
            init_weights="truncnormal002",
            channel_first=False,
        )
        seqlen = 13
        x = torch.randn(2, seqlen, dim)
        y1, = attn1(x)
        y2 = attn2(x)
        self.assertEqual(x.shape, y1.shape)
        self.assertEqual(x.shape, y2.shape)
        self.assertTrue(torch.allclose(y1, y2))

    def test_bimodal_shapes(self):
        dim = 16
        torch.manual_seed(98234)
        attn = MMDiTDotProductAttention(
            dim=dim,
            num_heads=2,
            num_modalities=2,
            qkv_bias=True,
            init_weights="truncnormal002",
            channel_first=False,
        )
        seqlens = [13, 8]
        x1, x2 = [torch.randn(2, seqlen, dim) for seqlen in seqlens]
        y1, y2 = attn(x1, x2)
        self.assertEqual(x1.shape, y1.shape)
        self.assertEqual(x2.shape, y2.shape)

    def test_trimodal_shapes(self):
        dim = 16
        torch.manual_seed(98234)
        attn = MMDiTDotProductAttention(
            dim=dim,
            num_heads=2,
            num_modalities=3,
            qkv_bias=True,
            init_weights="truncnormal002",
            channel_first=False,
        )
        seqlens = [6, 8, 2]
        x1, x2, x3 = [torch.randn(2, seqlen, dim) for seqlen in seqlens]
        y1, y2, y3 = attn(x1, x2, x3)
        self.assertEqual(x1.shape, y1.shape)
        self.assertEqual(x2.shape, y2.shape)
        self.assertEqual(x3.shape, y3.shape)
