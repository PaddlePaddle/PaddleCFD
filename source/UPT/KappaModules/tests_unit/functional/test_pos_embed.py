import unittest

import einops
import torch

from kappamodules.functional.pos_embed import get_sincos_pos_embed_from_seqlens, relative_position_indices
from original_modules.mae_pos_embed import get_2d_sincos_pos_embed


class TestPosEmbed(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual((10, 5), get_sincos_pos_embed_from_seqlens(seqlens=(10,), dim=5).shape)
        self.assertEqual((10, 20, 5), get_sincos_pos_embed_from_seqlens(seqlens=(10, 20), dim=5).shape)
        self.assertEqual((10, 20, 30, 5), get_sincos_pos_embed_from_seqlens(seqlens=(10, 20, 30), dim=5).shape)

    def test_padding_sincospad_2d(self):
        pos_embed_padded = get_sincos_pos_embed_from_seqlens(seqlens=(2, 3), dim=15)
        # pos_embed_unpadded is still padded because dim needs to be divided per ndim and then also between sin/cos
        pos_embed_unpadded = get_sincos_pos_embed_from_seqlens(seqlens=(2, 3), dim=14)
        self.assertEqual((2, 3, 15), pos_embed_padded.shape)
        self.assertTrue(torch.all(pos_embed_padded[:, :, -1] == 0))
        self.assertTrue(torch.all(pos_embed_padded[:, :, :-1] == pos_embed_unpadded))

    def test_nodimpadding_nosincospad_2d(self):
        pos_embed_padded = get_sincos_pos_embed_from_seqlens(seqlens=(2, 3), dim=9)
        pos_embed_unpadded = get_sincos_pos_embed_from_seqlens(seqlens=(2, 3), dim=8)
        self.assertEqual((2, 3, 9), pos_embed_padded.shape)
        self.assertTrue(torch.all(pos_embed_padded[:, :, -1] == 0))
        self.assertTrue(torch.all(pos_embed_padded[:, :, :-1] == pos_embed_unpadded))

    def test_nodimpadding_nosincospad_3d(self):
        pos_embed_padded = get_sincos_pos_embed_from_seqlens(seqlens=(2, 3, 4), dim=26)
        pos_embed_unpadded = get_sincos_pos_embed_from_seqlens(seqlens=(2, 3, 4), dim=24)
        self.assertEqual((2, 3, 4, 26), pos_embed_padded.shape)
        self.assertTrue(torch.all(pos_embed_padded[:, :, :, -1] == 0))
        self.assertTrue(torch.all(pos_embed_padded[:, :, :, -2] == 0))
        self.assertTrue(torch.all(pos_embed_padded[:, :, :, :-2] == pos_embed_unpadded))

    def test_equals_mae_square_xy(self):
        pos_embed_og = torch.from_numpy(get_2d_sincos_pos_embed(embed_dim=16, grid_size=4))
        pos_embed_og = pos_embed_og.reshape(4, 4, 16).float()
        pos_embed = get_sincos_pos_embed_from_seqlens(seqlens=(4, 4), dim=16, indexing="xy")
        self.assertTrue(torch.all(pos_embed == pos_embed_og))

    def test_equals_mae_square_ij(self):
        pos_embed_og = torch.from_numpy(get_2d_sincos_pos_embed(embed_dim=16, grid_size=4))
        pos_embed_og = einops.rearrange(pos_embed_og, "(width height) dim -> height width dim", width=4).float()
        pos_embed = get_sincos_pos_embed_from_seqlens(seqlens=(4, 4), dim=16, indexing="ij")
        self.assertTrue(torch.all(pos_embed == pos_embed_og))

    def test_equals_mae_rect_xy(self):
        pos_embed_og = torch.from_numpy(get_2d_sincos_pos_embed(embed_dim=16, grid_size=(3, 4)))
        pos_embed_og = pos_embed_og.reshape(3, 4, 16).float()
        pos_embed = get_sincos_pos_embed_from_seqlens(seqlens=(3, 4), dim=16, indexing="xy")
        self.assertTrue(torch.all(pos_embed == pos_embed_og))

    def test_relative_position_indices_5x5cls(self):
        seqlens = (5, 5)
        # [0, 0]: cls-cls
        # first row: cls-patch
        # first column: patch-cls
        expected = [
            [51, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49],
            [50, 24, 23, 22, 21, 17, 16, 15, 14, 10, 9, 8, 7, 3, 2, 1, 0],
            [50, 25, 24, 23, 22, 18, 17, 16, 15, 11, 10, 9, 8, 4, 3, 2, 1],
            [50, 26, 25, 24, 23, 19, 18, 17, 16, 12, 11, 10, 9, 5, 4, 3, 2],
            [50, 27, 26, 25, 24, 20, 19, 18, 17, 13, 12, 11, 10, 6, 5, 4, 3],
            [50, 31, 30, 29, 28, 24, 23, 22, 21, 17, 16, 15, 14, 10, 9, 8, 7],
            [50, 32, 31, 30, 29, 25, 24, 23, 22, 18, 17, 16, 15, 11, 10, 9, 8],
            [50, 33, 32, 31, 30, 26, 25, 24, 23, 19, 18, 17, 16, 12, 11, 10, 9],
            [50, 34, 33, 32, 31, 27, 26, 25, 24, 20, 19, 18, 17, 13, 12, 11, 10],
            [50, 38, 37, 36, 35, 31, 30, 29, 28, 24, 23, 22, 21, 17, 16, 15, 14],
            [50, 39, 38, 37, 36, 32, 31, 30, 29, 25, 24, 23, 22, 18, 17, 16, 15],
            [50, 40, 39, 38, 37, 33, 32, 31, 30, 26, 25, 24, 23, 19, 18, 17, 16],
            [50, 41, 40, 39, 38, 34, 33, 32, 31, 27, 26, 25, 24, 20, 19, 18, 17],
            [50, 45, 44, 43, 42, 38, 37, 36, 35, 31, 30, 29, 28, 24, 23, 22, 21],
            [50, 46, 45, 44, 43, 39, 38, 37, 36, 32, 31, 30, 29, 25, 24, 23, 22],
            [50, 47, 46, 45, 44, 40, 39, 38, 37, 33, 32, 31, 30, 26, 25, 24, 23],
            [50, 48, 47, 46, 45, 41, 40, 39, 38, 34, 33, 32, 31, 27, 26, 25, 24],
        ]
        actual, num_distinct_distances = relative_position_indices(seqlens=seqlens, num_aux_tokens=1)
        self.assertTrue(expected, actual.tolist())
        self.assertTrue(num_distinct_distances, 84)
