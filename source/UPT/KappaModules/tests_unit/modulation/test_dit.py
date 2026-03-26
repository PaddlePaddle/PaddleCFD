import unittest

import torch

from kappamodules.modulation import Dit


class TestDit(unittest.TestCase):
    def test_init_gate_zero(self):
        dit = Dit(cond_dim=5, out_dim=4, init_gate_zero=True, num_outputs=6, gate_indices=[2, 5])
        cond = torch.arange(5.).unsqueeze(0)
        scale1, shift1, gate1, scale2, shift2, gate2 = dit(cond)
        self.assertNotEqual(0, scale1.sum())
        self.assertNotEqual(0, shift1.sum())
        self.assertEqual(0, gate1.sum())
        self.assertNotEqual(0, scale2.sum())
        self.assertNotEqual(0, shift2.sum())
        self.assertEqual(0, gate2.sum())
