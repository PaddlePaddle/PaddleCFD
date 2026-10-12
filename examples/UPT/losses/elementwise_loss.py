import paddle
from losses import basic_loss_fn_from_kwargs
from utils.factory import create
from utils.loss_utils import apply_reduction
from utils.vit_util import patchify_as_1d

from .basic.mse_loss import MseLoss


class ElementwiseLoss(paddle.nn.Layer):
    def __init__(self, loss_function):
        super().__init__()
        self.loss_function = create(loss_function, basic_loss_fn_from_kwargs)

    def forward(self, prediction, target, mask=None, reduction="mean"):
        assert prediction.shape == target.shape
        loss = self.loss_function(prediction, target, reduction="none")
        if mask is not None:
            assert mask.dtype == paddle.bool and loss.shape == mask.shape
            loss = loss[mask]
        loss = apply_reduction(loss, reduction=reduction)
        return loss
