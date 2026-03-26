import unittest

import torch

from kappamodules.layers import LearnedBatchNorm, LearnedBatchNorm1d, LearnedBatchNorm3d, LearnedBatchNorm2d


class TestLearnedBatchNorm(unittest.TestCase):
    def test_lbatchnorm(self):
        x = torch.rand(size=(3, 10), generator=torch.Generator().manual_seed(843))
        lbn = LearnedBatchNorm(dim=10)
        y = lbn(x)
        self.assertEqual(x.shape, y.shape)

    def test_lbatchnorm1d(self):
        x = torch.rand(size=(3, 2, 10), generator=torch.Generator().manual_seed(843))
        lbn = LearnedBatchNorm1d(dim=2)
        y = lbn(x)
        self.assertEqual(x.shape, y.shape)

    def test_lbatchnorm2d(self):
        x = torch.rand(size=(3, 2, 10, 10), generator=torch.Generator().manual_seed(843))
        lbn = LearnedBatchNorm2d(dim=2)
        y = lbn(x)
        self.assertEqual(x.shape, y.shape)

    def test_lbatchnorm3d(self):
        x = torch.rand(size=(3, 2, 10, 10, 10), generator=torch.Generator().manual_seed(843))
        lbn = LearnedBatchNorm3d(dim=2)
        y = lbn(x)
        self.assertEqual(x.shape, y.shape)
