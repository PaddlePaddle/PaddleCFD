import paddle

ALL_BATCHNORMS = (
    paddle.nn.BatchNorm1D,
    paddle.nn.BatchNorm2D,
    paddle.nn.BatchNorm3D,
    paddle.nn.SyncBatchNorm,
)
_ALL_NORMS = (
    *ALL_BATCHNORMS,
    paddle.nn.LayerNorm,
    paddle.nn.InstanceNorm1D,
    paddle.nn.InstanceNorm2D,
    paddle.nn.InstanceNorm3D,
    paddle.nn.GroupNorm,
    paddle.nn.LocalResponseNorm,
)


def initialize_norms_as_noaffine(m):
    if isinstance(m, _ALL_NORMS):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)
        if m.weight is not None:
            paddle.nn.init.constant_(m.weight, 1.0)


def initialize_norms_as_identity(m):
    if isinstance(m, _ALL_NORMS):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)
        if m.weight is not None:
            paddle.nn.init.constant_(m.weight, 0.0)


def initialize_layernorm_as_noaffine(m):
    if isinstance(m, paddle.nn.LayerNorm):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)
        if m.weight is not None:
            paddle.nn.init.constant_(m.weight, 1.0)


def initialize_layernorm_as_identity(m):
    if isinstance(m, paddle.nn.LayerNorm):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)
        if m.weight is not None:
            paddle.nn.init.constant_(m.weight, 0.0)
    else:
        raise NotImplementedError


def initialize_batchnorm_as_noaffine(m):
    if isinstance(
        m, (paddle.nn.BatchNorm1D, paddle.nn.BatchNorm2D, paddle.nn.BatchNorm3D)
    ):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)
        if m.weight is not None:
            paddle.nn.init.constant_(m.weight, 1.0)


def initialize_batchnorm_as_identity(m):
    if isinstance(
        m, (paddle.nn.BatchNorm1D, paddle.nn.BatchNorm2D, paddle.nn.BatchNorm3D)
    ):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)
        if m.weight is not None:
            paddle.nn.init.constant_(m.weight, 0.0)
    else:
        raise NotImplementedError


def initialize_linear_bias_to_zero(m):
    if isinstance(m, paddle.compat.nn.Linear):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)


def initialize_conv_bias_to_zero(m):
    if isinstance(m, (paddle.nn.Conv1d, paddle.nn.Conv2d, paddle.nn.Conv3d)):
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)


def initialize_xavier_uniform_zero_bias(m):
    if isinstance(
        m,
        (paddle.compat.nn.Linear, paddle.nn.Conv1d, paddle.nn.Conv2d, paddle.nn.Conv3d),
    ):
        paddle.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            paddle.nn.init.constant_(m.bias, 0.0)


def initialize_qkv_seperately(model):
    for full_name, module in model.named_modules():
        last_name = full_name.split(".")[-1]
        if last_name == "qkv":
            val = (6 / (module.weight.shape[0] // 3 + module.weight.shape[1])) ** 0.5
            paddle.nn.init.uniform_(module.weight, -val, val)
        if last_name == "qkv_mlpin":
            input_dim = module.weight.shape[1]
            assert module.weight.shape[0] == 7 * input_dim
            qkv_bound = (3 / input_dim) ** 0.5
            mlpin_bound = (6 / (5 * input_dim)) ** 0.5
            paddle.nn.init.uniform_(
                module.weight[: 3 * input_dim], -qkv_bound, qkv_bound
            )
            paddle.nn.init.uniform_(
                module.weight[3 * input_dim :], -mlpin_bound, mlpin_bound
            )


def initialize_modulation_seperately(model):
    for full_name, module in model.named_modules():
        last_name = full_name.split(".")[-1]
        if last_name == "modulation":
            val = (6 / (module.weight.shape[0] // 2 + module.weight.shape[1])) ** 0.5
            paddle.nn.init.uniform_(module.weight, -val, val)


def initialize_seperately(model, name, denominator):
    for full_name, module in model.named_modules():
        last_name = full_name.split(".")[-1]
        if last_name == name:
            val = (
                6 / (module.weight.shape[0] // denominator + module.weight.shape[1])
            ) ** 0.5
            paddle.nn.init.uniform_(module.weight, -val, val)
