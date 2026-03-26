import unittest

import torch
from torch import nn

from kappamodules.layers import Residual


class TestResidual(unittest.TestCase):
    def test_single_module(self):
        # original
        torch.manual_seed(0)
        og_x = torch.randn(2, 4)
        og_layer = nn.Linear(4, 4)
        og_y = og_x + og_layer(og_x)
        og_y.mean().backward()

        # Residual
        torch.manual_seed(0)
        res_x = torch.randn(2, 4)
        res_layer = nn.Linear(4, 4)
        res_module = Residual(res_layer)
        res_y = res_module(res_x)
        res_y.mean().backward()

        # checks
        self.assertTrue(torch.all(og_y == res_y))
        self.assertTrue(torch.all(og_layer.weight.grad == res_layer.weight.grad))
        self.assertTrue(torch.all(og_layer.bias.grad == res_layer.bias.grad))

    def test_expand_module(self):
        # original
        torch.manual_seed(0)
        og_x = torch.randn(2, 4)
        og_layer0 = nn.Linear(4, 8)
        og_layer1 = nn.Linear(8, 4)
        og_layer = nn.Sequential(og_layer0, og_layer1)
        og_y = og_x + og_layer(og_x)
        og_y.mean().backward()

        # Residual
        torch.manual_seed(0)
        res_x = torch.randn(2, 4)
        res_layer0 = nn.Linear(4, 8)
        res_layer1 = nn.Linear(8, 4)
        res_module = Residual(res_layer0, res_layer1)
        res_y = res_module(res_x)
        res_y.mean().backward()

        # checks
        self.assertTrue(torch.all(og_y == res_y))
        self.assertTrue(torch.all(og_layer0.weight.grad == res_layer0.weight.grad))
        self.assertTrue(torch.all(og_layer0.bias.grad == res_layer0.bias.grad))
        self.assertTrue(torch.all(og_layer1.weight.grad == res_layer1.weight.grad))
        self.assertTrue(torch.all(og_layer1.bias.grad == res_layer1.bias.grad))
