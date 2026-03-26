import unittest

import torch
from torch import nn

from kappamodules.layers import AsyncBatchNorm


class TestAsyncBatchNorm(unittest.TestCase):
    def test_abn_shape(self):
        x = torch.rand(size=(3, 10), generator=torch.Generator().manual_seed(843))
        asb = AsyncBatchNorm(dim=10)
        y = asb(x)
        self.assertEqual(x.shape, y.shape)

    def test_abn_converges(self):
        x = torch.rand(size=(3, 10), generator=torch.Generator().manual_seed(843))
        asb = AsyncBatchNorm(dim=10, momentum=0.0)
        bn = torch.nn.BatchNorm1d(10)
        # momentum=0 -> track stats
        _ = asb(x)
        # normalize with stats
        y2 = asb(x)
        # compate to batchnorm
        y3 = bn(x)
        self.assertTrue(torch.allclose(y2.std(), y3.std()))

    def test_abn_backward(self):
        x = torch.rand(size=(3, 10), generator=torch.Generator().manual_seed(843))
        asb = AsyncBatchNorm(dim=10)
        y = asb(x)
        y.mean().backward()
        self.assertEqual(x.shape, y.shape)

    def test_abn_backward_nowhiten(self):
        x = torch.rand(size=(3, 10), generator=torch.Generator().manual_seed(843))
        asb = AsyncBatchNorm(dim=10, whiten=False)
        y = asb(x)
        y.mean().backward()
        self.assertEqual(x.shape, y.shape)

    def test_convert_nostats(self):
        syncbn = nn.Sequential(nn.Linear(5, 10), nn.BatchNorm1d(10))
        asyncbn = AsyncBatchNorm.convert_async_batchnorm(syncbn)
        self.assertIsInstance(asyncbn[1], AsyncBatchNorm)

    def test_convert_withstats(self):
        bn = nn.BatchNorm1d(10)
        bn(torch.rand(4, 10, generator=torch.Generator().manual_seed(84)))
        syncbn = nn.Sequential(nn.Linear(5, 10), bn)
        asyncbn = AsyncBatchNorm.convert_async_batchnorm(syncbn)
        self.assertIsInstance(asyncbn[1], AsyncBatchNorm)
        self.assertTrue(torch.all(bn.running_mean == asyncbn[1].mean))
        self.assertTrue(torch.all(bn.running_var == asyncbn[1].var))
