import unittest

import torch

from kappamodules.pooling import CrossAttentionPooling


class TestCrossAttentionPooling(unittest.TestCase):
    def test_learnable_queries(self):
        seqlen = 10
        dim = 4
        num_heads = 2
        x = torch.randn(size=(1, seqlen, dim))
        pooling = CrossAttentionPooling(dim=dim, num_heads=num_heads, num_query_tokens=1)
        y = pooling(x)
        self.assertEqual((1, 1, dim), y.shape)

    def test_provided_queries(self):
        seqlen = 10
        dim = 4
        num_heads = 2
        x = torch.randn(size=(1, seqlen, dim))
        q = torch.randn(size=(1, 2, dim))
        pooling = CrossAttentionPooling(dim=dim, num_heads=num_heads, num_query_tokens=0)
        y = pooling(x, query_tokens=q)
        self.assertEqual((1, 2, dim), y.shape)
