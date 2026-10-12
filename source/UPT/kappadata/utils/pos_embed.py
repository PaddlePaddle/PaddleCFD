import einops
import paddle


def get_2d_sincos_pos_embed(embed_dim, h, w):
    """ returns 2d sincos positional embedding with shape (embed_dim, h, w) """
    assert embed_dim % 4 == 0
    grid_h = paddle.arange(h, dtype=paddle.float32)
    grid_w = paddle.arange(w, dtype=paddle.float32)
    mesh_h, mesh_w = paddle.meshgrid(grid_h, grid_w, indexing="ij")

    # half for height, half for width
    # NOTE:
    half_dim = embed_dim // 2
    emb_h = get_1d_sincos_pos_embed_from_grid(half_dim, mesh_h.flatten())
    emb_w = get_1d_sincos_pos_embed_from_grid(half_dim, mesh_w.flatten())

    # reshape back to image shape
    emb_h = einops.rearrange(emb_h, "(h w) dh -> dh h w", h=h, w=w, dh=half_dim)
    emb_w = einops.rearrange(emb_w, "(h w) dh -> dh h w", h=h, w=w, dh=half_dim)

    return paddle.concat([emb_h, emb_w])


def get_1d_sincos_pos_embed_from_grid(embed_dim, positions):
    """
    embed_dim: output dimension for each position
    positions: a list of positions to be encoded: size (L,)
    out: (L, D)
    """
    assert embed_dim % 2 == 0
    assert positions.ndim == 1

    omega = paddle.arange(embed_dim // 2, dtype=paddle.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega

    # d is d // 2 here
    out = paddle.einsum('l,d->ld', positions, omega)

    emb_sin = out.sin()  # (L, D/2)
    emb_cos = out.cos()  # (L, D/2)
    emb = paddle.concat([emb_sin, emb_cos], dim=1)  # (L, D)
    return emb
