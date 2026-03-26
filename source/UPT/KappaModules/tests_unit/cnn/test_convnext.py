import unittest
from functools import partial

import torch
from timm import create_model

from kappamodules.convolution.convnext import ConvNextBlock, ConvNext
from kappamodules.layers import LayerNorm2d


class TestConvnext(unittest.TestCase):
    def test_block_eqals_convnextv2small(self):
        torch.manual_seed(894)
        og = create_model("convnextv2_small")
        og_block0 = og.stages[0].blocks[0]
        block0 = ConvNextBlock(dim=96, norm_ctor=partial(LayerNorm2d, eps=1e-6))
        self.assertEqual(
            sum(p.numel() for p in block0.parameters()),
            sum(p.numel() for p in og_block0.parameters()),
        )
        with torch.no_grad():
            for og_param, param in zip(og_block0.parameters(), block0.parameters()):
                param.copy_(og_param.view(*param.shape))
        x = torch.randn(1, 96, 7, 7)
        og_y = og_block0(x)
        y = block0(x)
        self.assertTrue(torch.allclose(og_y, y))

    def test_forward2d_shape(self):
        convnext = ConvNext(
            patch_size=2,
            input_dim=4,
            dims=[6, 12, 18],
            depths=[2, 3, 4],
        )
        x = torch.randn(1, 4, 16, 16)
        y = convnext(x)
        # 16x16 -> patchify -> 8x8 -> 2 downsampling stages -> 2x2
        self.assertEqual((1, 18, 2, 2), y.shape)

    def test_forward2d_droppath_shape(self):
        convnext = ConvNext(
            patch_size=2,
            input_dim=4,
            dims=[6, 12, 18],
            depths=[2, 3, 4],
            drop_path_rate=0.1,
        )
        x = torch.randn(1, 4, 16, 16)
        y = convnext(x)
        # 16x16 -> patchify -> 8x8 -> 2 downsampling stages -> 2x2
        self.assertEqual((1, 18, 2, 2), y.shape)

    def test_forward2d_droppathunif_shape(self):
        convnext = ConvNext(
            patch_size=2,
            input_dim=4,
            dims=[6, 12, 18],
            depths=[2, 3, 4],
            drop_path_rate=0.1,
            drop_path_decay=False,
        )
        x = torch.randn(1, 4, 16, 16)
        y = convnext(x)
        # 16x16 -> patchify -> 8x8 -> 2 downsampling stages -> 2x2
        self.assertEqual((1, 18, 2, 2), y.shape)
