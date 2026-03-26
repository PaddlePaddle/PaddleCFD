import unittest

import torch

from kappamodules.vit import VitBlock


class TestVitBlock(unittest.TestCase):
    def test(self):
        dim = 4
        block = VitBlock(dim=dim, num_heads=2)
        x = torch.randn(2, 6, dim, generator=torch.Generator().manual_seed(9834))
        y = block(x)
        self.assertEqual(x.shape, y.shape)
