import paddle

ALL_BATCHNORMS = (
    paddle.nn.BatchNorm1D,
    paddle.nn.BatchNorm2D,
    paddle.nn.BatchNorm3D,
    paddle.nn.SyncBatchNorm,
)
ALL_NORMS = (
    *ALL_BATCHNORMS,
    paddle.nn.LayerNorm,
    paddle.nn.InstanceNorm1D,
    paddle.nn.InstanceNorm2D,
    paddle.nn.InstanceNorm3D,
    paddle.nn.GroupNorm,
    paddle.nn.LocalResponseNorm,
)
ALL_CONVS = (
    paddle.nn.Conv1D,
    paddle.nn.Conv2D,
    paddle.nn.Conv3D,
    paddle.nn.Conv1DTranspose,
    paddle.nn.Conv2DTranspose,
    paddle.nn.Conv3DTranspose,
)
ALL_LAYERS = paddle.nn.Linear, *ALL_CONVS


def init_with_scheme(module, scheme):
    if scheme == "paddle":
        pass
    elif scheme in ["truncnormal", "truncnormal002"]:
        module.apply(init_truncnormal_zero_bias)
    elif scheme == "xavier_uniform":
        module.apply(init_xavier_uniform_zero_bias)
    else:
        raise NotImplementedError


def init_norm_as_noaffine(m):
    if isinstance(m, ALL_NORMS):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)
        if m.weight is not None:
            paddle.nn.init.constant_(m.weight, 1.0)


def init_norms_as_noaffine(m):
    if isinstance(m, ALL_NORMS):
        if m.bias is not None:
            # paddle.nn.init.constant_(m.bias, 0.0)
            paddle.nn.initializer.Constant(0.0)(m.bias)
        if m.weight is not None:
            # paddle.nn.init.constant_(m.weight, 1.0)
            paddle.nn.initializer.Constant(1.0)(m.weight)


def init_layernorm_as_noaffine(m):
    if isinstance(m, paddle.nn.LayerNorm):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)
        if m.weight is not None:
            paddle.nn.init.constant_(m.weight, 1.0)


def init_batchnorm_as_noaffine(m):
    if isinstance(m, ALL_BATCHNORMS):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)
        if m.weight is not None:
            paddle.nn.init.constant_(m.weight, 1.0)


def init_norm_as_identity(m):
    if isinstance(m, ALL_NORMS):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)
        if m.weight is not None:
            paddle.nn.init.constant_(m.weight, 0.0)


def init_norms_as_identity(m):
    if isinstance(m, ALL_NORMS):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)
        if m.weight is not None:
            paddle.nn.init.constant_(m.weight, 0.0)


def init_layernorm_as_identity(m):
    if isinstance(m, paddle.nn.LayerNorm):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)
        if m.weight is not None:
            paddle.nn.init.constant_(m.weight, 0.0)
    else:
        raise NotImplementedError


def init_batchnorm_as_identity(m):
    if isinstance(m, ALL_BATCHNORMS):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)
        if m.weight is not None:
            paddle.nn.init.constant_(m.weight, 0.0)
    else:
        raise NotImplementedError


def init_bias_to_zero(m):
    if isinstance(m, ALL_LAYERS):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)


def init_zero(m):
    if isinstance(m, ALL_LAYERS):
        paddle.nn.init.zeros_(m.weight)
        if m.bias is not None:
            paddle.nn.init.zeros_(m.bias)


def init_linear_bias_to_zero(m):
    if isinstance(m, paddle.compat.nn.Linear):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)


def init_conv_bias_to_zero(m):
    if isinstance(m, ALL_CONVS):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)


def init_xavier_uniform_zero_bias(m, gain: float = 1.0):
    if isinstance(m, ALL_LAYERS):
        # paddle.nn.init.xavier_uniform_(m.weight, gain=gain)
        initializer = paddle.nn.initializer.XavierUniform(fan_in=None, fan_out=None, gain=gain)
        initializer(m.weight)
        if m.bias is not None:
            # paddle.nn.init.constant_(m.bias, 0.0)
            paddle.nn.initializer.Constant(0.0)(m.bias)


def init_truncnormal_zero_bias(m, std=0.02):
    if isinstance(m, ALL_LAYERS):
        # paddle.nn.init.trunc_normal_(m.weight, std=std)
        paddle.nn.initializer.TruncatedNormal(std=std)(m.weight)
        if m.bias is not None:
            # paddle.nn.init.constant_(m.bias, 0.0)
            paddle.nn.initializer.Constant(0.0)(m.bias)


def init_linear_truncnormal_zero_bias(m, std=0.02):
    if isinstance(m, paddle.compat.nn.Linear):
        # paddle.nn.init.trunc_normal_(m.weight, std=std)
        paddle.nn.initializer.TruncatedNormal(std=std)(m.weight)
        if m.bias is not None:
            # paddle.nn.init.constant_(m.bias, 0.0)
            paddle.nn.initializer.Constant(0.0)(m.bias)



def init_xavier_uniform_merged_linear(module, num_layers):
    assert isinstance(module, paddle.compat.nn.Linear)
    assert module.weight.shape[0] % num_layers == 0
    val = (6 / (module.weight.shape[0] // num_layers + module.weight.shape[1])) ** 0.5
    paddle.nn.init.uniform_(module.weight, -val, val)
    if module.bias is not None:
        paddle.nn.init.constant_(module.bias, 0.0)
