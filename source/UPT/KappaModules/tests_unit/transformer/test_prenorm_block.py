import unittest

import torch

from kappamodules.transformer import PrenormBlock


class TestPrenormBlock(unittest.TestCase):
    def test_shape_mask(self):
        dim = 8
        seqlen = 5
        attn = PrenormBlock(dim=dim, num_heads=2)
        x = torch.randn(1, seqlen, dim)
        attn_mask = torch.rand(1, seqlen, seqlen) > 0.5
        y = attn(x, attn_mask=attn_mask)
        self.assertEqual(x.shape, y.shape)
