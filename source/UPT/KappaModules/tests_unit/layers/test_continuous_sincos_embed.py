import unittest

import einops
import torch

from kappamodules.functional.pos_embed import get_sincos_pos_embed_from_seqlens
from kappamodules.layers import ContinuousSincosEmbed


class TestContinuousSincosEmbed(unittest.TestCase):
    def test_shape(self):
        batch_size = 3
        num_points = 32
        ndim = 2
        dim = 16
        # dense
        dense_coords = torch.rand(batch_size, num_points, ndim, generator=torch.Generator().manual_seed(8943))
        embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        dense_result = embed(dense_coords)
        self.assertEqual((batch_size, num_points, dim), dense_result.shape)
        # sparse
        sparse_coords = torch.rand(batch_size * num_points, ndim, generator=torch.Generator().manual_seed(8943))
        embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        sparse_result = embed(sparse_coords)
        self.assertEqual((batch_size * num_points, dim), sparse_result.shape)

    def test_dtype(self):
        ndim = 2
        coords = torch.rand(3, 32, ndim, generator=torch.Generator().manual_seed(8943))
        embed = ContinuousSincosEmbed(dim=16, ndim=ndim)
        result = embed(coords.double())
        self.assertEqual(torch.double, result.dtype)

    def test_equals_regulargrid_square2d(self):
        seqlens = (4, 4)
        dim = 16
        expected = get_sincos_pos_embed_from_seqlens(seqlens=seqlens, dim=dim, indexing="ij")
        grids = torch.meshgrid(*[torch.arange(seqlen, dtype=torch.float32) for seqlen in seqlens], indexing="ij")
        coords = einops.rearrange(torch.stack(grids), "ndim ... -> (...) ndim")
        actual = ContinuousSincosEmbed(dim=dim, ndim=len(seqlens), dtype=torch.double)(coords).reshape(*expected.shape)
        self.assertTrue(torch.all(expected == actual))

    def test_equals_regulargrid_rect2d(self):
        seqlens = (3, 4)
        dim = 16
        expected = get_sincos_pos_embed_from_seqlens(seqlens=seqlens, dim=dim, indexing="ij")
        coords = torch.meshgrid(*[torch.arange(seqlen, dtype=torch.float32) for seqlen in seqlens], indexing="ij")
        coords = einops.rearrange(torch.stack(coords), "ndim ... -> (...) ndim")
        actual = ContinuousSincosEmbed(dim=dim, ndim=len(seqlens), dtype=torch.double)(coords).reshape(*expected.shape)
        self.assertTrue(torch.all(expected == actual))

    def test_equals_regulargrid_rect3d(self):
        seqlens = (3, 4, 5)
        dim = 18
        expected = get_sincos_pos_embed_from_seqlens(seqlens=seqlens, dim=dim, indexing="ij")
        coords = torch.meshgrid(*[torch.arange(seqlen, dtype=torch.float32) for seqlen in seqlens], indexing="ij")
        coords = einops.rearrange(torch.stack(coords), "ndim ... -> (...) ndim")
        actual = ContinuousSincosEmbed(dim=dim, ndim=len(seqlens), dtype=torch.double)(coords).reshape(*expected.shape)
        self.assertTrue(torch.all(expected == actual))

    def test_shape_3d_dim256(self):
        dim = 256
        actual = ContinuousSincosEmbed(dim=dim, ndim=3)(torch.rand(1, 3))
        self.assertEqual((1, dim), actual.shape)

    def test_padding_equals_regulargridpadding(self):
        seqlens = (3, 4)
        dim = 15
        expected = get_sincos_pos_embed_from_seqlens(seqlens=seqlens, dim=dim, indexing="ij")
        coords = torch.meshgrid(*[torch.arange(seqlen, dtype=torch.float32) for seqlen in seqlens], indexing="ij")
        coords = einops.rearrange(torch.stack(coords), "ndim ... -> (...) ndim")
        actual = ContinuousSincosEmbed(dim=dim, ndim=len(seqlens), dtype=torch.double)(coords).reshape(*expected.shape)
        # continuous has all padding as last dimensions, static embed has the sincos padding in the middle
        # continuous padded dims are: 12, 13, 14
        # fixed padded dims are: 6, 13, 14
        self.assertTrue(torch.all(expected[:, :, :5] == actual[:, :, :5]))
        self.assertTrue(torch.all(expected[:, :, 6] == actual[:, :, 12]))
        self.assertTrue(torch.all(expected[:, :, 7:13] == actual[:, :, 6:12]))
        self.assertTrue(torch.all(expected[:, :, 13:] == actual[:, :, 13:]))
        self.assertTrue(torch.all(actual[:, :, 13:] == 0))

    def test_padding(self):
        ndim = 2
        coords = torch.rand(3, 32, ndim, generator=torch.Generator().manual_seed(8943))
        result = ContinuousSincosEmbed(dim=15, ndim=ndim)(coords)
        self.assertEqual((3, 32, 15), result.shape)
