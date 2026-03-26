import unittest

import torch

from kappamodules.transformer import MMMDitBlock


class TestMMMDiTBlock(unittest.TestCase):
    def test_shape_mainonly(self):
        torch.manual_seed(9845)
        dim = 8
        seqlen1 = 5
        seqlen2 = 10
        block = MMMDitBlock(dim=dim, num_heads=2, num_main_modalities=2, num_optional_modalities=0, cond_dim=dim)
        x1 = torch.randn(4, seqlen1, dim)
        x2 = torch.randn(4, seqlen2, dim)
        cond = torch.randn(4, dim)
        y1, y2 = block([x1, x2], cond=[cond, cond])
        self.assertEqual(x1.shape, y1.shape)
        self.assertEqual(x2.shape, y2.shape)
        self.assertTrue(torch.allclose(y1.mean(), torch.tensor(0.022029364481568336)))
        self.assertTrue(torch.allclose(y2.mean(), torch.tensor(0.0025057464372366667)))

    def test_shape(self):
        torch.manual_seed(9845)
        dim = 8
        seqlen1 = 5
        seqlen2 = 10
        seqlen3 = 7
        block = MMMDitBlock(dim=dim, num_heads=2, num_main_modalities=2, num_optional_modalities=1, cond_dim=dim)
        x1 = torch.randn(4, seqlen1, dim)
        x2 = torch.randn(4, seqlen2, dim)
        x3 = torch.randn(4, seqlen3, dim)
        cond = torch.randn(4, dim)
        y1, y2, y3 = block([x1, x2, x3], cond=[cond, cond, cond])
        self.assertEqual(x1.shape, y1.shape)
        self.assertEqual(x2.shape, y2.shape)
        self.assertEqual(x3.shape, y3.shape)
        self.assertTrue(torch.allclose(y1.mean(), torch.tensor(0.11030294001102448)))
        self.assertTrue(torch.allclose(y2.mean(), torch.tensor(-0.007555273827165365)))
        self.assertTrue(torch.allclose(y3.mean(), torch.tensor(0.1332601010799408)))
