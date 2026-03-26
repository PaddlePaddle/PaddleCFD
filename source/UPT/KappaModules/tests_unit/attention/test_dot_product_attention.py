import unittest

import einops
import numpy as np
import torch

from kappamodules.attention import DotProductAttention1d, DotProductAttention2d, DotProductAttention3d


class TestDotProductAttention(unittest.TestCase):
    def test_1d_channellast(self):
        torch.manual_seed(98234)
        dim = 16
        attn = DotProductAttention1d(
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
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(0.010636774823069572)))

    def test_1d_channellast_relpos5x5(self):
        torch.manual_seed(98234)
        dim = 16
        seqlens = (5, 5)
        attn = DotProductAttention1d(
            dim=dim,
            num_heads=2,
            qkv_bias=True,
            init_weights="xavier_uniform",
            channel_first=False,
            rel_pos_bias="learnable",
            seqlens=seqlens,
        )
        seqlen = int(np.prod(seqlens))
        x = torch.randn(2, seqlen + 1, dim)
        y = attn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(-0.028891844674944878)))

    def test_1d_channelfirst(self):
        torch.manual_seed(98234)
        dim = 16
        attn = DotProductAttention1d(
            dim=dim,
            num_heads=2,
            qkv_bias=True,
            init_weights="xavier_uniform",
            channel_first=True,
        )
        seqlen = 13
        x = torch.randn(2, dim, seqlen)
        y = attn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(-0.00038123130798339844)))

    def test_1d_channelfirst_equals_channellast(self):
        # setup
        torch.manual_seed(98234)
        dim = 16
        seqlen = 13
        # channelfirst
        attn1 = DotProductAttention1d(
            dim=dim,
            num_heads=2,
            qkv_bias=True,
            init_weights="xavier_uniform",
            channel_first=True,
        )
        x1 = torch.randn(2, dim, seqlen)
        y1 = attn1(x1)
        # channellast
        x2 = einops.rearrange(x1, "bs dim seqlen -> bs seqlen dim")
        attn2 = DotProductAttention1d(
            dim=dim,
            num_heads=2,
            qkv_bias=True,
            init_weights="xavier_uniform",
            channel_first=False,
        )
        attn2.load_state_dict(attn1.state_dict())
        y2 = attn2(x2)
        y2 = einops.rearrange(y2, "bs seqlen dim -> bs dim seqlen")
        # check
        self.assertEqual(y1.shape, y2.shape)
        self.assertTrue(torch.allclose(y1, y2))

    def test_2d_channelfirst(self):
        torch.manual_seed(98234)
        dim = 16
        attn = DotProductAttention2d(
            dim=dim,
            num_heads=2,
            qkv_bias=True,
            init_weights="xavier_uniform",
            channel_first=True,
        )
        x = torch.randn(2, dim, 2, 3)
        y = attn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(0.047952961176633835)))

    def test_3d_channelfirst(self):
        torch.manual_seed(98234)
        dim = 16
        attn = DotProductAttention3d(
            dim=dim,
            num_heads=2,
            qkv_bias=True,
            init_weights="xavier_uniform",
            channel_first=True,
        )
        x = torch.randn(2, dim, 2, 3, 4)
        y = attn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(-0.019655022770166397)))
