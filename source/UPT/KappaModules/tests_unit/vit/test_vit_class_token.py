import unittest

import einops
import torch

from kappamodules.vit import VitClassTokens


class TestVitClassToken(unittest.TestCase):
    def test_first(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 3)
        cls_token = VitClassTokens(dim=3, num_tokens=1, location="first")
        y = cls_token(x)
        pooled = cls_token.pool(y)
        self.assertTrue(torch.all(einops.repeat(cls_token.tokens, "1 1 dim -> bs dim", bs=len(x)) == pooled))
        self.assertTrue(torch.all(y[:, 0] == pooled))

    def test_middle(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 3)
        cls_token = VitClassTokens(dim=3, num_tokens=1, location="middle")
        y = cls_token(x)
        pooled = cls_token.pool(y)
        self.assertTrue(torch.all(einops.repeat(cls_token.tokens, "1 1 dim -> bs dim", bs=len(x)) == pooled))
        self.assertTrue(torch.all(y[:, 2] == pooled))

    def test_last(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 3)
        cls_token = VitClassTokens(dim=3, num_tokens=1, location="last")
        y = cls_token(x)
        pooled = cls_token.pool(y)
        self.assertTrue(torch.all(einops.repeat(cls_token.tokens, "1 1 dim -> bs dim", bs=len(x)) == pooled))
        self.assertTrue(torch.all(y[:, -1] == pooled))

    def test_middle_vit(self):
        torch.manual_seed(0)
        x = torch.randn(2, 196, 3)
        cls_token = VitClassTokens(dim=3, num_tokens=1, location="middle")
        y = cls_token(x)
        pooled = cls_token.pool(y)
        self.assertTrue(torch.all(einops.repeat(cls_token.tokens, "1 1 dim -> bs dim", bs=len(x)) == pooled))
        self.assertTrue(torch.all(y[:, 98] == pooled))

    def test_bilateral_vit(self):
        torch.manual_seed(0)
        x = torch.randn(2, 196, 3)
        cls_token = VitClassTokens(dim=3, num_tokens=2, location="bilateral")
        y = cls_token(x)
        first, last = cls_token.pool(y).chunk(chunks=2, dim=1)
        self.assertTrue(torch.all(y[:, 0] == first))
        self.assertTrue(torch.all(y[:, -1] == last))

    def test_bilateral_vit_mean(self):
        torch.manual_seed(0)
        x = torch.randn(2, 196, 3)
        cls_token = VitClassTokens(dim=3, num_tokens=2, location="bilateral", aggregate="mean")
        y = cls_token(x)
        expected = cls_token.tokens.mean(dim=1).expand(len(y), -1)
        actual = cls_token.pool(y)
        self.assertTrue(torch.all(actual == expected))

    def test_uniform_vit(self):
        torch.manual_seed(0)
        x = torch.randn(2, 196, 3)
        cls_token = VitClassTokens(dim=3, num_tokens=4, location="uniform")
        y = cls_token(x)
        pooled = cls_token.pool(y)
        rep_tokens = einops.repeat(cls_token.tokens, "1 num_tokens dim -> bs (num_tokens dim)", bs=len(x))
        self.assertTrue(torch.all(rep_tokens == pooled))
        cls0, cls1, cls2, cls3 = pooled.chunk(chunks=4, dim=1)
        self.assertTrue(torch.all(y[:, 40] == cls0))
        self.assertTrue(torch.all(y[:, 81] == cls1))
        self.assertTrue(torch.all(y[:, 122] == cls2))
        self.assertTrue(torch.all(y[:, 163] == cls3))
