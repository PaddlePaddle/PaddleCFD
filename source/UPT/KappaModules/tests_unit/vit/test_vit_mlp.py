import unittest

import torch
from timm.layers import Mlp

from kappamodules.vit import VitMlp


class TestVitMlp(unittest.TestCase):
    def test_equal_to_timm(self):
        seed = 89453
        torch.manual_seed(seed)
        mlp0 = Mlp(in_features=4, hidden_features=12)
        torch.manual_seed(seed)
        mlp1 = VitMlp(in_dim=4, hidden_dim=12, init_weights="torch")
        torch.manual_seed(seed)
        x = torch.randn(3, 4)
        y0 = mlp0(x)
        y1 = mlp1(x)
        self.assertTrue(torch.all(y0 == y1))
