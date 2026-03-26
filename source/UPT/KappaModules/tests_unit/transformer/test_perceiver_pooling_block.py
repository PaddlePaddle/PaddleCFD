import unittest

import torch

from kappamodules.transformer import PerceiverPoolingBlock


class TestPerceiverPoolingBlock(unittest.TestCase):
    def test_shape(self):
        dim = 8
        num_query_tokens = 3
        batch_size = 2
        block = PerceiverPoolingBlock(dim=dim, num_query_tokens=num_query_tokens, num_heads=2)
        kv = torch.ones(batch_size, 6, dim)
        y = block(kv=kv)
        self.assertEqual((batch_size, num_query_tokens, dim), y.shape)
