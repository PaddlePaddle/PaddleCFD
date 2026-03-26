import unittest

import torch

from kappamodules.unet.unet_pdearena import UnetPdearena


class TestUnetPdearena(unittest.TestCase):
    def test_shape(self):
        model = UnetPdearena(input_dim=2, output_dim=2, hidden_channels=3, cond_dim=4)
        x = torch.randn(1, 2, 32, 32)
        cond = torch.randn(1, 4)
        y = model(x, emb=cond)
        self.assertEqual(y.shape, x.shape)
