import unittest

import torch

from kappamodules.transformer import PerceiverBlock


class TestPerceiverBlock(unittest.TestCase):
    def test_shape(self):
        dim = 8
        block = PerceiverBlock(dim=dim, num_heads=2)
        q = torch.ones(1, 3, dim)
        kv = torch.ones(1, 6, dim)
        y = block(q=q, kv=kv)
        self.assertEqual(q.shape, y.shape)

    def test_DontConcatQtoKV(self):
        dim = 8
        block = PerceiverBlock(dim=dim, num_heads=2, concat_query_to_kv=False)
        q = torch.ones(1, 3, dim)
        kv = torch.ones(1, 6, dim)
        y = block(q=q, kv=kv)
        self.assertEqual(q.shape, y.shape)
