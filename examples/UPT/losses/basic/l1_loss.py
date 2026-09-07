import paddle


class L1Loss(paddle.nn.Layer):
    @staticmethod
    def forward(pred, target, reduction="mean"):
        return paddle.nn.functional.l1_loss(
            input=pred, label=target, reduction=reduction
        )
