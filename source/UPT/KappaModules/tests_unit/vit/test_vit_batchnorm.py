import unittest

import torch

from kappamodules.vit import VitBatchNorm


class TestVitBatchNorm(unittest.TestCase):
    def test_0d(self):
        dim = 4
        bn = VitBatchNorm(num_features=dim, eps=0)
        x = torch.rand(10, dim, generator=torch.Generator().manual_seed(89543))
        y = bn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.allclose(y.mean(dim=0), torch.zeros(size=(dim,)), atol=1e-6))
        self.assertTrue(torch.allclose(y.std(dim=0), torch.ones(size=(dim,)), atol=1e-1))

    def test_1d(self):
        dim = 4
        bn = VitBatchNorm(num_features=dim, eps=0)
        x = torch.rand(5, 4, dim, generator=torch.Generator().manual_seed(89543))
        y = bn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.allclose(y.mean(dim=[0, 1]), torch.zeros(size=(dim,)), atol=1e-6))
        self.assertTrue(torch.allclose(y.std(dim=[0, 1]), torch.ones(size=(dim,)), atol=1e-1))

    def test_2d(self):
        dim = 4
        bn = VitBatchNorm(num_features=dim, eps=0)
        x = torch.rand(5, 4, 3, dim, generator=torch.Generator().manual_seed(89543))
        y = bn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.allclose(y.mean(dim=[0, 1, 2]), torch.zeros(size=(dim,)), atol=1e-6))
        self.assertTrue(torch.allclose(y.std(dim=[0, 1, 2]), torch.ones(size=(dim,)), atol=1e-1))

    def test_3d(self):
        dim = 4
        bn = VitBatchNorm(num_features=dim, eps=0)
        x = torch.rand(5, 4, 3, 2, dim, generator=torch.Generator().manual_seed(89543))
        y = bn(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.allclose(y.mean(dim=[0, 1, 2, 3]), torch.zeros(size=(dim,)), atol=1e-6))
        self.assertTrue(torch.allclose(y.std(dim=[0, 1, 2, 3]), torch.ones(size=(dim,)), atol=1e-1))
