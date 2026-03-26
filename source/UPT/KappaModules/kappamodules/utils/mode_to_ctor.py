import paddle

from kappamodules.layers import AsyncBatchNorm, Identity, LearnedBatchNorm


def mode_to_norm_ctor(mode):
    if mode is None:
        return Identity, True
    mode = mode.lower().replace("_", "")
    if mode == "none":
        return Identity, True
    if mode in ["bn", "batchnorm", "batchnorm1d"]:
        return paddle.nn.BatchNorm1D, False
    if mode in ["batchnorm2d"]:
        return paddle.nn.BatchNorm2D, False
    if mode in ["batchnorm3d"]:
        return paddle.nn.BatchNorm3D, False
    if mode in ["ln", "layernorm"]:
        return paddle.nn.LayerNorm, True
    if mode in ["instancenorm1d"]:
        return paddle.nn.InstanceNorm1D, True
    if mode in ["instancenorm2d"]:
        return paddle.nn.InstanceNorm2D, True
    if mode in ["instancenorm3d"]:
        return paddle.nn.InstanceNorm3D, True
    if mode in ["gn", "groupnorm"]:
        return paddle.nn.GroupNorm, True
    if mode in ["lrn", "localresponsenorm"]:
        return paddle.nn.LocalResponseNorm, True
    if mode in ["learned", "learenedbatchnorm"]:
        return LearnedBatchNorm, False
    if mode in ["abn", "asyncbatchnorm"]:
        return AsyncBatchNorm, False
    raise NotImplementedError(f"no suitable norm constructor found for '{mode}'")
