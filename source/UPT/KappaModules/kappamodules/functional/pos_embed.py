import einops
import paddle


def get_sincos_1d_from_seqlen(seqlen: int, dim: int, max_wavelength: int = 10000):
    grid = paddle.arange(seqlen, dtype=paddle.float64)
    return get_sincos_1d_from_grid(grid=grid, dim=dim, max_wavelength=max_wavelength)


def get_sincos_1d_from_grid(grid, dim: int, max_wavelength: int = 10000):
    if dim % 2 == 0:
        padding = None
    else:
        padding = paddle.zeros(*grid.shape, 1)
        dim -= 1
    omega = 1.0 / max_wavelength ** (
        paddle.arange(0, dim, 2, dtype=paddle.float64) / dim
    )
    out = grid.unsqueeze(-1) @ omega.unsqueeze(0)
    emb_sin = paddle.sin(out)
    emb_cos = paddle.cos(out)
    emb = paddle.concat([emb_sin, emb_cos], dim=-1).float()
    if padding is None:
        return emb
    else:
        return paddle.concat([emb, padding], dim=-1)


def get_sincos_2d_from_seqlens(seqlens, dim: int, max_wavelength: int = 10000):
    seqlen_h, seqlen_w = seqlens
    grid_h = paddle.arange(seqlen_h, dtype=paddle.float64)
    grid_w = paddle.arange(seqlen_w, dtype=paddle.float64)
    grid = paddle.meshgrid(grid_h, grid_w, indexing="xy")
    grid = paddle.stack(grid).reshape(2, seqlen_h, seqlen_w)
    return get_2d_sincos_pos_embed_from_grid(
        grid=grid, dim=dim, max_wavelength=max_wavelength
    )


def get_2d_sincos_pos_embed_from_grid(grid, dim: int, max_wavelength: int = 10000):
    assert dim % 2 == 0
    grid_h, grid_w = grid
    emb_h = get_sincos_1d_from_grid(
        grid=grid_h, dim=dim // 2, max_wavelength=max_wavelength
    )
    emb_w = get_sincos_1d_from_grid(
        grid=grid_w, dim=dim // 2, max_wavelength=max_wavelength
    )
    return paddle.concat([emb_h, emb_w], dim=-1)


def get_sincos_3d_from_seqlens(seqlens, dim: int, max_wavelength: int = 10000):
    seqlen_x, seqlen_y, seqlen_z = seqlens
    grid_x = paddle.arange(seqlen_x, dtype=paddle.float64)
    grid_y = paddle.arange(seqlen_y, dtype=paddle.float64)
    grid_z = paddle.arange(seqlen_z, dtype=paddle.float64)
    grid = paddle.meshgrid(grid_x, grid_y, grid_z, indexing="xy")
    grid = paddle.stack(grid).reshape(3, seqlen_x, seqlen_y, seqlen_z)
    return get_3d_sincos_pos_embed_from_grid(
        grid=grid, dim=dim, max_wavelength=max_wavelength
    )


def get_3d_sincos_pos_embed_from_grid(grid, dim: int, max_wavelength: int = 10000):
    assert dim % 3 == 0
    grid_x, grid_y, grid_z = grid
    emb_x = get_sincos_1d_from_grid(
        grid=grid_x, dim=dim // 3, max_wavelength=max_wavelength
    )
    emb_y = get_sincos_1d_from_grid(
        grid=grid_y, dim=dim // 3, max_wavelength=max_wavelength
    )
    emb_z = get_sincos_1d_from_grid(
        grid=grid_z, dim=dim // 3, max_wavelength=max_wavelength
    )
    return paddle.concat([emb_x, emb_y, emb_z], dim=-1)


def get_sincos_pos_embed_from_seqlens(
    seqlens, dim: int, max_wavelength: int = 10000, indexing="ij"
):
    assert isinstance(seqlens, (tuple, list))
    grids = [paddle.arange(seqlen, dtype=paddle.float64) for seqlen in seqlens]
    if indexing == "xy":
        grids = reversed(grids)
    grid = paddle.stack(paddle.meshgrid(*grids, indexing=indexing))
    return get_sincos_pos_embed_from_grid(
        grid=grid, dim=dim, max_wavelength=max_wavelength
    )


def get_sincos_pos_embed_from_grid(grid, dim: int, max_wavelength: int = 10000):
    ndim = grid.size(0)
    if dim % ndim == 0:
        padding = None
    else:
        padding_dim = dim % ndim
        padding = paddle.zeros(*grid.shape[1:], padding_dim)
        dim -= padding_dim
    pos_embed = paddle.concat(
        [
            get_sincos_1d_from_grid(
                grid=grid[i], dim=dim // ndim, max_wavelength=max_wavelength
            )
            for i in range(ndim)
        ],
        dim=-1,
    )
    if padding is None:
        return pos_embed
    else:
        return paddle.concat([pos_embed, padding], dim=-1)


def interpolate_sincos(
    embed, seqlens, mode: str = "bicubic", interpolate_offset: float = None
):
    old_dtype = embed.dtype
    assert embed.ndim - 2 == len(seqlens)
    embed = einops.rearrange(embed, "1 ... dim -> 1 dim ...").float()
    if interpolate_offset:
        scale_factor = [
            ((seqlens[i] + interpolate_offset) / embed.size(i + 2))
            for i in range(len(seqlens))
        ]
        embed = paddle.nn.functional.interpolate(
            embed, scale_factor=scale_factor, mode=mode
        )
    else:
        embed = paddle.nn.functional.interpolate(embed, size=seqlens, mode=mode)
    embed = einops.rearrange(embed, "1 dim ... -> 1 ... dim")
    return embed.to(old_dtype)


def relative_position_indices(seqlens, num_aux_tokens):
    """creates a bias for each relative distance"""
    assert len(seqlens) == 2
    assert num_aux_tokens == 1
    seqlen0, seqlen1 = seqlens
    num_distinct_distances = (2 * seqlen0 - 1) * (2 * seqlen1 - 1) + 3
    abs_coords = paddle.stack(
        paddle.meshgrid([paddle.arange(seqlen0), paddle.arange(seqlen1)], indexing="ij")
    )
    abs_coords_flat = einops.rearrange(abs_coords, "ndim ... -> ndim (...)")
    rel_coords = abs_coords_flat[:, :, None] - abs_coords_flat[:, None, :]
    rel_coords = einops.rearrange(rel_coords, "ndim ... -> ... ndim").contiguous()
    rel_coords[:, :, 0] += seqlen0 - 1
    rel_coords[:, :, 1] += seqlen1 - 1
    rel_coords[:, :, 0] *= 2 * seqlen1 - 1
    rel_pos_index = rel_coords.new_zeros(
        size=(seqlen0 * seqlen1 + 1, seqlen0 * seqlen1 + 1)
    )
    rel_pos_index[1:, 1:] = rel_coords.sum(-1)
    rel_pos_index[0, 0] = num_distinct_distances - 1
    rel_pos_index[0:, 0] = num_distinct_distances - 2
    rel_pos_index[0, 0:] = num_distinct_distances - 3
    return rel_pos_index, num_distinct_distances
