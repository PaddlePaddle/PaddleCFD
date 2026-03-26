import unittest
from functools import partial

import torch

from kappamodules.attention import TranssolverAttention
from kappamodules.transformer import DitBlock


class TestDitBlock(unittest.TestCase):
    def test_shape(self):
        torch.manual_seed(9845)
        dim = 8
        seqlen = 5
        block = DitBlock(dim=dim, num_heads=2, cond_dim=dim)
        x = torch.randn(4, seqlen, dim)
        cond = torch.randn(4, dim)
        y = block(x, cond=cond)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(torch.tensor(21.843421936035156), y.sum()))

    def test_shape_droppath(self):
        torch.manual_seed(9845)
        dim = 8
        seqlen = 5
        block = DitBlock(dim=dim, num_heads=2, cond_dim=dim, drop_path=0.25)
        x = torch.randn(4, seqlen, dim)
        cond = torch.randn(4, dim)
        y = block(x, cond=cond)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(torch.tensor(-28.474021911621094), y.sum()))

    def test_gate0(self):
        torch.manual_seed(9845)
        dim = 8
        seqlen = 5
        block = DitBlock(dim=dim, num_heads=2, cond_dim=dim, init_gate_zero=True)
        x = torch.randn(4, seqlen, dim)
        cond = torch.randn(4, dim)
        y = block(x, cond=cond)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.all(x == y))

    def test_transsolver(self):
        torch.manual_seed(9845)
        dim = 8
        seqlen = 5
        block = DitBlock(dim=dim, num_heads=2, cond_dim=dim, attn_ctor=partial(TranssolverAttention, num_slices=3))
        x = torch.randn(4, seqlen, dim)
        cond = torch.randn(4, dim)
        y = block(x, cond=cond)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(torch.tensor(40.46888732910156), y.sum()), y.sum().item())
