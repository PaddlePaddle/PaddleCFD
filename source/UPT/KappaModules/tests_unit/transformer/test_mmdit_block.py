import unittest

import torch

from kappamodules.transformer import MMDitBlock


class TestMMDiTBlock(unittest.TestCase):
    def test_shape(self):
        torch.manual_seed(9845)
        dim = 8
        seqlen1 = 5
        seqlen2 = 10
        block = MMDitBlock(dim=dim, num_heads=2, cond_dim=dim)
        x1 = torch.randn(4, seqlen1, dim)
        x2 = torch.randn(4, seqlen2, dim)
        cond = torch.randn(4, dim)
        y1, y2 = block([x1, x2], cond=[cond, cond])
        self.assertEqual(x1.shape, y1.shape)
        self.assertEqual(x2.shape, y2.shape)
        self.assertTrue(torch.allclose(y1.mean(), torch.tensor(0.022029364481568336)))
        self.assertTrue(torch.allclose(y2.mean(), torch.tensor(0.0025057464372366667)))
