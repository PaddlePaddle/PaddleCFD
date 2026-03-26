import unittest

import torch

from kappamodules.unet import UnetGino


class TestUnetGino(unittest.TestCase):
    def test_forward(self):
        torch.manual_seed(9823)
        unet = UnetGino(
            input_dim=3,
            hidden_dim=64,
            depth=4,
            num_groups=8,
        )
        x = torch.randn(1, 3, 32, 32, 32)
        y = unet(x)
        expected_shape = list(x.shape)
        expected_shape[1] = 64
        self.assertEqual(tuple(expected_shape), y.shape)
        self.assertTrue(torch.isclose(y.mean(), torch.tensor(0.01520983874797821)))
