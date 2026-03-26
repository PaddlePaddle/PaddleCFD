import unittest

import torch

from kappamodules.vit import VitPosEmbed1d, VitPosEmbed2d, VitPosEmbed3d


class TestVitPosEmbed(unittest.TestCase):
    def _test1d(self, is_learnable):
        seqlen = 4
        dim = 8
        x = torch.zeros(2, seqlen, dim)
        pos_embed = VitPosEmbed1d(seqlens=(seqlen,), dim=dim, is_learnable=is_learnable)
        y = pos_embed(x)
        self.assertEqual(x.shape, y.shape)
        # check learnable initialization
        if is_learnable:
            self.assertTrue(torch.all(y != 0))

    def test_1d_fixed(self):
        self._test1d(is_learnable=False)

    def test_1d_learnable(self):
        self._test1d(is_learnable=True)

    def _test2d(self, is_learnable):
        seqlens = (4, 5)
        dim = 8
        x = torch.zeros(2, *seqlens, dim)
        pos_embed = VitPosEmbed2d(seqlens=seqlens, dim=dim, is_learnable=is_learnable)
        y = pos_embed(x)
        self.assertEqual(x.shape, y.shape)

    def test_2d_fixed(self):
        self._test2d(is_learnable=False)

    def test_2d_learnable(self):
        self._test2d(is_learnable=True)

    def _test3d(self, is_learnable):
        seqlens = (4, 5, 6)
        dim = 9
        x = torch.zeros(2, *seqlens, dim)
        pos_embed = VitPosEmbed3d(seqlens=seqlens, dim=dim, is_learnable=is_learnable)
        y = pos_embed(x)
        self.assertEqual(x.shape, y.shape)

    def test_3d_fixed(self):
        self._test3d(is_learnable=False)

    def test_3d_learnable(self):
        self._test3d(is_learnable=True)

    def test_interpolate_2d(self):
        seqlens = (8, 12)
        dim = 64
        pos_embed = VitPosEmbed2d(seqlens=seqlens, dim=dim, is_learnable=False)
        # forward pass with full resolution
        x_full = torch.zeros(2, *seqlens, dim)
        y_full = pos_embed(x_full)
        # forward pass with half resolution
        x_half = torch.zeros(2, *[seqlen // 2 for seqlen in seqlens], dim)
        y_half = pos_embed(x_half)
        self.assertEqual(x_half.shape, y_half.shape)
