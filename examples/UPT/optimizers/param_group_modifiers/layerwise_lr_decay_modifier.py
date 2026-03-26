from .base.param_group_modifier import ParamGroupModifier


class LayerwiseLrDecayModifier(ParamGroupModifier):
    def __init__(self, decay, skip_layers=None):
        self.decay = decay
        self.skip_layers = skip_layers

    def get_properties(self, model, name, param):
        num_layers = len(model.blocks) + 1
        scales = list(self.decay ** (num_layers - i) for i in range(num_layers))
        if self.skip_layers is not None:
            scales = scales[self.skip_layers :] + [1.0] * self.skip_layers
        if name in ["cls_token", "mask_token", "pos_embed"] or name.startswith(
            "patch_embed"
        ):
            return dict(lr_scale=scales[0])
        elif name.startswith("block"):
            layer = int(name.split(".")[1]) + 1
            return dict(lr_scale=scales[layer])
        elif name.startswith("norm."):
            return {}
        elif name.startswith("head."):
            return {}
        else:
            raise NotImplementedError

    def __str__(self):
        return f"{type(self).__name__}(decay={self.decay})"
