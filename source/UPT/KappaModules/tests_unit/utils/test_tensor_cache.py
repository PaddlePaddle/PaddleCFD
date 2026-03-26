import unittest
from functools import partial

import torch

import kappamodules.utils.tensor_cache as tc


class TestTensorCache(unittest.TestCase):
    def _test(self, ctor, expected):
        tensor = ctor()
        self.assertTrue(torch.all(tensor == expected))
        self.assertEqual(expected.device, tensor.device)
        tensor2 = ctor()
        self.assertEqual(id(tensor), id(tensor2))
        tensor[0] = 3
        with self.assertRaises(AssertionError) as ex:
            ctor()
        self.assertTrue(str(ex.exception).startswith(f"cached tensor with key="))
        self.assertTrue(str(ex.exception).endswith(f"was modified inplace"))

    def test_zeros(self):
        self._test(
            ctor=partial(tc.zeros, size=(5,)),
            expected=torch.zeros(5),
        )

    def test_ones(self):
        self._test(
            ctor=partial(tc.ones, size=(5,)),
            expected=torch.ones(5),
        )

    def test_full(self):
        self._test(
            ctor=partial(tc.full, size=(5,), fill_value=3),
            expected=torch.full(size=(5,), fill_value=3),
        )

    def test_arange(self):
        self._test(
            ctor=partial(tc.arange, start=0, end=3),
            expected=torch.arange(3),
        )

    def test_arange_nokwargs(self):
        self._test(
            ctor=partial(tc.arange, 3),
            expected=torch.arange(3),
        )
