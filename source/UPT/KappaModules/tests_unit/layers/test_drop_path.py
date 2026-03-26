import unittest

import torch
from torch import nn

from kappamodules.layers import DropPath
from original_modules.original_drop_path import OriginalDropPath


class TestEfficientDropPath(unittest.TestCase):
    def is_equal_to_original(self, batch_size, dim, drop_prob, stochastic_drop_prob, scale_by_keep, training, seed):
        # create layers
        torch.manual_seed(seed)
        custom_sequential_layer = nn.Linear(dim, dim)
        torch.manual_seed(seed)
        custom_standalone_layer = nn.Linear(dim, dim)
        torch.manual_seed(seed)
        original_layer = nn.Linear(dim, dim)
        # create DropPath modules
        kwargs = dict(drop_prob=drop_prob, stochastic_drop_prob=stochastic_drop_prob, scale_by_keep=scale_by_keep)
        custom_sequential = DropPath(custom_sequential_layer, **kwargs)
        custom_standalone = DropPath(**kwargs)
        original = OriginalDropPath(**kwargs)
        # prepare forward
        if not training:
            custom_sequential = custom_sequential.eval()
            custom_standalone = custom_standalone.eval()
            original = original.eval()

        # create data (and make sure that DropPath does nothing inplace)
        x = torch.randn(batch_size, dim, generator=torch.Generator().manual_seed(seed))
        og_x = x.clone()
        # original forward + backward
        torch.manual_seed(seed)
        original_y = x + original(original_layer(x))
        original_y.mean().backward()
        # custom sequential forward
        torch.manual_seed(seed)
        custom_sequential_y = custom_sequential(x)
        custom_sequential_y.mean().backward()
        # custom standalone forward
        torch.manual_seed(seed)
        custom_standalone_y = custom_standalone(x, residual_path=custom_standalone_layer)
        custom_standalone_y.mean().backward()
        # check general
        self.assertTrue(torch.all(x == og_x))
        # check custom_sequential == original
        self.assertTrue(torch.allclose(original_y, custom_sequential_y))
        self.assertTrue(torch.allclose(original_layer.weight.grad, custom_sequential_layer.weight.grad))
        self.assertTrue(torch.allclose(original_layer.bias.grad, custom_sequential_layer.bias.grad))
        # check custom_standalone == original
        self.assertTrue(torch.allclose(original_y, custom_standalone_y))
        self.assertTrue(torch.allclose(original_layer.weight.grad, custom_standalone_layer.weight.grad))
        self.assertTrue(torch.allclose(original_layer.bias.grad, custom_standalone_layer.bias.grad))

    def test_standalone_or_sequential(self):
        layer = nn.Linear(4, 4)
        drop_path = DropPath(layer, drop_prob=0.2)
        with self.assertRaises(AssertionError):
            drop_path(torch.randn(10, 4), residual_path=layer)

    def test_03_stochasticsize_scalebykeep_training(self):
        self.is_equal_to_original(
            batch_size=10,
            dim=4,
            drop_prob=0.3,
            stochastic_drop_prob=True,
            scale_by_keep=True,
            training=True,
            seed=4389,
        )

    def test_03_fixedsize_scalebykeep_training(self):
        self.is_equal_to_original(
            batch_size=10,
            dim=4,
            drop_prob=0.3,
            stochastic_drop_prob=False,
            scale_by_keep=True,
            training=True,
            seed=4389,
        )

    def test_03_stochasticsize_noscalebykeep_training(self):
        self.is_equal_to_original(
            batch_size=10,
            dim=4,
            drop_prob=0.3,
            stochastic_drop_prob=True,
            scale_by_keep=False,
            training=True,
            seed=4389,
        )

    def test_03_stochasticsize_scalebykeep_eval(self):
        self.is_equal_to_original(
            batch_size=10,
            dim=4,
            drop_prob=0.3,
            stochastic_drop_prob=True,
            scale_by_keep=True,
            training=False,
            seed=4389,
        )

    def test_03_fixedsize_noscalebykeep_training(self):
        self.is_equal_to_original(
            batch_size=10,
            dim=4,
            drop_prob=0.3,
            stochastic_drop_prob=False,
            scale_by_keep=False,
            training=True,
            seed=4389,
        )

    def test_03_fixedsize_scalebykeep_eval(self):
        self.is_equal_to_original(
            batch_size=10,
            dim=4,
            drop_prob=0.3,
            stochastic_drop_prob=False,
            scale_by_keep=True,
            training=False,
            seed=4389,
        )

    def test_03_fixedsize_noscalebykeep_eval(self):
        self.is_equal_to_original(
            batch_size=10,
            dim=4,
            drop_prob=0.3,
            stochastic_drop_prob=False,
            scale_by_keep=False,
            training=False,
            seed=4389,
        )

    def test_099_stoch_scalebykeep(self):
        self.is_equal_to_original(
            batch_size=10,
            dim=4,
            drop_prob=0.99,
            stochastic_drop_prob=True,
            scale_by_keep=True,
            training=True,
            seed=4389,
        )
