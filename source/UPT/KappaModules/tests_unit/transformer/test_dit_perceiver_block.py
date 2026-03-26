import unittest

import torch

from kappamodules.transformer import DitPerceiverBlock


class TestDitPerceiverBlock(unittest.TestCase):
    def test_shape(self):
        batch_size = 4
        dim = 8
        seqlen_q = 5
        seqlen_kv = 6
        block = DitPerceiverBlock(dim=dim, num_heads=2, cond_dim=dim)
        q = torch.randn(batch_size, seqlen_q, dim)
        kv = torch.randn(batch_size, seqlen_kv, dim)
        cond = torch.randn(batch_size, dim)
        y = block(q=q, kv=kv, cond=cond)
        self.assertEqual(q.shape, y.shape)

    def test_shape_droppath(self):
        batch_size = 4
        dim = 8
        seqlen_q = 5
        seqlen_kv = 6
        block = DitPerceiverBlock(dim=dim, num_heads=2, cond_dim=dim, drop_path=0.25)
        q = torch.randn(batch_size, seqlen_q, dim)
        kv = torch.randn(batch_size, seqlen_kv, dim)
        cond = torch.randn(batch_size, dim)
        y = block(q=q, kv=kv, cond=cond)
        self.assertEqual(q.shape, y.shape)
