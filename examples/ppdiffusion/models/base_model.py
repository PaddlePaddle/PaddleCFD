from contextlib import contextmanager
from typing import Optional

import paddle


class BaseModel(paddle.nn.Layer):
    """Base class for all models.
    - :func:`__init__`
    - :func:`forward`
    """

    def __init__(
        self,
        num_input_channels: int,
        num_output_channels: int,
        num_conditional_channels: int = 0,
    ):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.num_conditional_channels = num_conditional_channels

        # get dropout layers
        self.dropout_layers = [
            m for m in self.sublayers() if isinstance(m, paddle.nn.Dropout) or isinstance(m, paddle.nn.Dropout2D)
        ]

    @property
    def num_params(self):
        """Returns the number of parameters in the model"""
        return sum(p.size for p in self.parameters() if not p.stop_gradient)

    def forward(self, X: paddle.Tensor, condition: Optional[paddle.Tensor] = None, **kwargs) -> paddle.Tensor:
        r"""Forward

        Args:
            X (Tensor): Input data tensor of shape :math:`(B, *, C_{in})`
        Shapes:
            - Input: :math:`(B, *, C_{in})`,

            where :math:`B` is the batch size, :math:`*` is the spatial dimension(s) of the data,
            and :math:`C_{in}` is the number of input features/channels.
        """
        raise NotImplementedError

    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.stop_gradient = True
        self.eval()

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.stop_gradient = False
        self.train()

    def enable_infer_dropout(self):
        for layer in self.dropout_layers:
            layer.training = True

    def disable_infer_dropout(self):
        for layer in self.dropout_layers:
            layer.training = False

    @contextmanager
    def dropout_controller(self, enable):
        if enable:
            self.enable_infer_dropout()
        try:
            yield
        finally:
            if enable:
                self.disable_infer_dropout()
