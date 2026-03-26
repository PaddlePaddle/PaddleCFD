import unittest

import torch

from kappamodules.modulation import Film


class TestFilm(unittest.TestCase):
    def test_channel_last(self):
        film = Film(dim_cond=5, dim_out=4, channel_first=False)
        cond = torch.randn(2, 5, generator=torch.Generator().manual_seed(0))
        x = torch.randn(2, 3, 4, generator=torch.Generator().manual_seed(0))
        x = film(x, cond)
        self.assertEqual((2, 3, 4), x.shape)
        self.assertTrue(torch.all(x[0] != x[1]))
