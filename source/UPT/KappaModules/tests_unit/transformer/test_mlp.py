import unittest

import torch

from kappamodules.transformer import Mlp


class TestMlp(unittest.TestCase):
    def test_init_last_proj_zero(self):
        mlp = Mlp(in_dim=4, init_last_proj_zero=True)
        x = torch.randn(2, 4)
        y = mlp(x)
        self.assertTrue(torch.all(y == 0))
