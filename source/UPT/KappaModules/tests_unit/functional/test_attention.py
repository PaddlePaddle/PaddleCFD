import unittest

import torch

from kappamodules.functional.attention import scaled_dot_product_attention


class TestAttention(unittest.TestCase):
    def _run_test(self, **kwargs):
        results = []
        for backend in ["flash", "math", "vanilla"]:
            results.append(scaled_dot_product_attention(**kwargs, backend=backend))
        assert len(results) == 3
        self.assertTrue(torch.allclose(results[0], results[1], atol=1e-6))
        self.assertTrue(torch.allclose(results[0], results[2], atol=1e-6))

    def test_sdpa_selfattn_nomask(self):
        batch_size = 2
        seqlen = 8
        num_heads = 3
        head_dim = 4
        q = torch.randn(batch_size, num_heads, seqlen, head_dim, generator=torch.Generator().manual_seed(0))
        k = torch.randn(batch_size, num_heads, seqlen, head_dim, generator=torch.Generator().manual_seed(1))
        v = torch.randn(batch_size, num_heads, seqlen, head_dim, generator=torch.Generator().manual_seed(2))
        self._run_test(query=q, key=k, value=v)

    def test_sdpa_selfattn_boolmask(self):
        batch_size = 2
        seqlen = 8
        num_heads = 3
        head_dim = 4
        q = torch.randn(batch_size, num_heads, seqlen, head_dim, generator=torch.Generator().manual_seed(0))
        k = torch.randn(batch_size, num_heads, seqlen, head_dim, generator=torch.Generator().manual_seed(1))
        v = torch.randn(batch_size, num_heads, seqlen, head_dim, generator=torch.Generator().manual_seed(2))
        attn_mask = torch.randn(batch_size, 1, seqlen, seqlen, generator=torch.Generator().manual_seed(4)) < 0
        self._run_test(query=q, key=k, value=v, attn_mask=attn_mask)

    def test_sdpa_selfattn_floatmask(self):
        batch_size = 2
        seqlen = 8
        num_heads = 3
        head_dim = 4
        q = torch.randn(batch_size, num_heads, seqlen, head_dim, generator=torch.Generator().manual_seed(0))
        k = torch.randn(batch_size, num_heads, seqlen, head_dim, generator=torch.Generator().manual_seed(1))
        v = torch.randn(batch_size, num_heads, seqlen, head_dim, generator=torch.Generator().manual_seed(2))
        attn_mask = torch.randn(batch_size, 1, seqlen, seqlen, generator=torch.Generator().manual_seed(4))
        self._run_test(query=q, key=k, value=v, attn_mask=attn_mask)

    def test_sdpa_selfattn_causal(self):
        batch_size = 2
        seqlen = 8
        num_heads = 3
        head_dim = 4
        q = torch.randn(batch_size, num_heads, seqlen, head_dim, generator=torch.Generator().manual_seed(0))
        k = torch.randn(batch_size, num_heads, seqlen, head_dim, generator=torch.Generator().manual_seed(1))
        v = torch.randn(batch_size, num_heads, seqlen, head_dim, generator=torch.Generator().manual_seed(2))
        self._run_test(query=q, key=k, value=v, is_causal=True)

    def test_sdpa_crossattn_floatmask(self):
        batch_size = 2
        seqlen_q = 8
        seqlen_kv = 7
        num_heads = 3
        head_dim = 4
        q = torch.randn(batch_size, num_heads, seqlen_q, head_dim, generator=torch.Generator().manual_seed(0))
        k = torch.randn(batch_size, num_heads, seqlen_kv, head_dim, generator=torch.Generator().manual_seed(1))
        v = torch.randn(batch_size, num_heads, seqlen_kv, head_dim, generator=torch.Generator().manual_seed(2))
        attn_mask = torch.randn(batch_size, 1, seqlen_q, seqlen_kv, generator=torch.Generator().manual_seed(4))
        self._run_test(query=q, key=k, value=v, attn_mask=attn_mask)
