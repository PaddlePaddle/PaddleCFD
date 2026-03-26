import unittest

import torch

from kappamodules.layers import LinearProjection


class TestLinearProjection(unittest.TestCase):
    def test_optional(self):
        proj = LinearProjection(input_dim=5, output_dim=5, optional=True)
        x = torch.randn(1, 5)
        y = proj(x)
        self.assertTrue(torch.all(x == y))

    def test_samedim_nooptional(self):
        proj = LinearProjection(input_dim=5, output_dim=5)
        x = torch.randn(1, 5)
        y = proj(x)
        self.assertTrue(torch.all(x != y))
