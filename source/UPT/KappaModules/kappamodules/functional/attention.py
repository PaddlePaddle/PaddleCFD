import einops
import paddle
import paddle.nn.functional as F

def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    backend="auto",
    **kwargs,
):
    sdpa_kwargs = dict(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        **kwargs,
    )
    if backend == "auto":
        return F.scaled_dot_product_attention(**sdpa_kwargs)
    if backend == "flash":
        return F.scaled_dot_product_attention(**sdpa_kwargs)
    if backend == "math":
        return F.scaled_dot_product_attention(**sdpa_kwargs)
    if backend in ["vanilla", "flops"]:
        assert len(kwargs) == 0
        batch_size, _, seqlen_q, head_dim = query.shape
        if is_causal:
            assert attn_mask is None
            assert query.shape == key.shape
            attn_mask = paddle.tril(
                paddle.ones(
                    (seqlen_q, seqlen_q), dtype=paddle.bool, device=query.device
                )
            )
            attn_mask = einops.rearrange(
                attn_mask, "seqlen1 seqlen2 -> 1 1 seqlen1 seqlen2"
            )
        scale = head_dim**-0.5
        scaled_query = query * scale
        attn = scaled_query @ key.transpose(-2, -1)
        if attn_mask is not None:
            assert attn_mask.ndim == attn.ndim
            if attn_mask.dtype == paddle.bool:
                attn[~attn_mask.expand(*attn.shape)] = float("-inf")
            else:
                attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = paddle.nn.functional.dropout(attn, p=dropout_p)
        x = attn @ value
        return x
    raise NotImplementedError(f"invalid backend {backend}")
