import unittest

import torch

from kappamodules.attention import EfficientAttention1d


class TestEfficientAttention(unittest.TestCase):
    def test_1d_channellast(self):
        torch.manual_seed(98234)
        dim = 16
        attn = EfficientAttention1d(
            dim=dim,
            num_heads=2,
            qkv_bias=True,
            init_weights="xavier_uniform",
            channel_first=False,
        )
        seqlen = 13
        x = torch.randn(2, seqlen, dim)
        y = attn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(-0.007668268401175737)))
