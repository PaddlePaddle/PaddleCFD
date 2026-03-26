import unittest

import torch

from kappamodules.attention import LinformerAttention1d


class TestLinformerAttention(unittest.TestCase):
    def test_1d_channellast(self):
        torch.manual_seed(98234)
        dim = 16
        seqlen = 13
        attn = LinformerAttention1d(
            dim=dim,
            input_seqlen=seqlen,
            kv_seqlen=4,
            num_heads=2,
            qkv_bias=True,
            init_weights="xavier_uniform",
            channel_first=False,
        )
        x = torch.randn(2, seqlen, dim)
        y = attn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(0.023179292678833008)))
