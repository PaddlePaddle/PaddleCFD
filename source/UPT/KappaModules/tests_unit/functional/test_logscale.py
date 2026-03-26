import unittest

import torch

from kappamodules.functional.logscale import to_logscale, from_logscale


class TestLogscale(unittest.TestCase):
    def test_identity(self):
        x = torch.randn(5, generator=torch.Generator().manual_seed(0))
        y = to_logscale(x)
        z = from_logscale(y)
        self.assertTrue(torch.allclose(x, z))
