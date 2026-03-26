import unittest

import torch

from kappamodules.vit import DitBlock
from original_modules.original_dit_block import OriginalDitBlock


class TestDitBlock(unittest.TestCase):
    def test_equal_to_original(self):
        dim = 12
        seed = 239084
        torch.manual_seed(seed)
        block = DitBlock(dim=dim, num_heads=2, init_weights="torch")
        torch.manual_seed(seed)
        block_og = OriginalDitBlock(hidden_size=dim, num_heads=2)

        x = torch.randn(2, 6, dim, generator=torch.Generator().manual_seed(9834))
        cond = torch.randn(2, dim, generator=torch.Generator().manual_seed(564))
        y = block(x, cond=cond)
        y_og = block_og(x, c=cond)
        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.allclose(y_og, y))

    def test_masked(self):
        dim = 12
        torch.manual_seed(239084)
        block = DitBlock(dim=dim, num_heads=2, init_weights="torch")
        x = torch.randn(2, 6, dim, generator=torch.Generator().manual_seed(9834))
        cond = torch.randn(2, dim, generator=torch.Generator().manual_seed(564))
        mask = torch.randn(2, 6, 6, generator=torch.Generator().manual_seed(324514)) > 0
        y = block(x, cond=cond, attn_mask=mask)
        self.assertEqual(x.shape, y.shape)
