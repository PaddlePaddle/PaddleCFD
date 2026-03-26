import paddle


class MSELoss(paddle.nn.Layer):
    @staticmethod
    def forward(pred, target, reduction="mean"):
        return paddle.nn.functional.mse_loss(
            input=pred, label=target, reduction=reduction
        )
