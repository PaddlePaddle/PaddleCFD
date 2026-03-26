import unittest

import torch
from torch import nn

from kappamodules.layers import WeightNormLinear


class TestWeightNormLinear(unittest.TestCase):
    def is_equal_to_native(self, dim_in, dim_out, bias, fixed_g, seed=49385):
        # native layer
        torch.manual_seed(seed)
        native = nn.utils.weight_norm(nn.Linear(dim_in, dim_out, bias=bias))
        if fixed_g:
            native.weight_g.data.fill_(1)
        # custom layer
        torch.manual_seed(seed)
        custom = WeightNormLinear(dim_in, dim_out, bias=bias, fixed_g=fixed_g, init_weights="torch")
        # checks
        self.assertTrue(torch.all(native.weight_g == custom.weight_g))
        self.assertTrue(torch.all(native.weight_v == custom.weight_v))
        if bias:
            self.assertTrue(torch.all(native.bias == custom.bias))
        x = torch.randn(2, dim_in, generator=torch.Generator().manual_seed(seed))
        self.assertTrue(torch.all(native(x) == custom(x)))

    def test_is_equal_to_native_nobias_nofixedg(self):
        self.is_equal_to_native(dim_in=4, dim_out=8, bias=False, fixed_g=False)

    def test_is_equal_to_native_bias_nofixedg(self):
        self.is_equal_to_native(dim_in=4, dim_out=8, bias=True, fixed_g=False)

    def test_is_equal_to_native_nobias_fixedg(self):
        self.is_equal_to_native(dim_in=4, dim_out=8, bias=False, fixed_g=True)

    def test_is_equal_to_native_bias_fixedg(self):
        self.is_equal_to_native(dim_in=4, dim_out=8, bias=True, fixed_g=True)
