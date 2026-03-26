import unittest

import torch

from kappamodules.unet import UnetDenoisingDiffusion


class TestUnetDenoisingDiffusion(unittest.TestCase):
    def test_1d_uncond(self):
        torch.manual_seed(9823)
        num_channels = 1
        seqlen = 8
        model = UnetDenoisingDiffusion(dim=8, ndim=1, dim_in=num_channels, dim_out=num_channels, num_heads=2, depth=2)
        x = torch.randn(2, num_channels, seqlen)
        y = model(x)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(-0.04462967440485954)))

    def test_1d_cond(self):
        torch.manual_seed(9823)
        num_channels = 1
        seqlen = 8
        dim_cond = 4
        model = UnetDenoisingDiffusion(
            dim=8,
            ndim=1,
            dim_in=num_channels,
            dim_out=num_channels,
            num_heads=2,
            depth=2,
            dim_cond=dim_cond,
        )
        x = torch.randn(2, num_channels, seqlen)
        cond = torch.randn(2, 4)
        y = model(x, cond=cond)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(0.3150939643383026)))

    def test_2d_cond_noattn(self):
        torch.manual_seed(9823)
        num_channels = 1
        seqlen = 8
        dim_cond = 4
        model = UnetDenoisingDiffusion(
            dim=8,
            ndim=1,
            dim_in=num_channels,
            dim_out=num_channels,
            depth=2,
            dim_cond=dim_cond,
            num_heads=None,
        )
        x = torch.randn(2, num_channels, seqlen)
        cond = torch.randn(2, 4)
        y = model(x, cond=cond)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(-0.6672834753990173)))
