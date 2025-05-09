# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from typing import Optional

import paddle


class BaseModel(paddle.nn.Layer):
    """Base class for all models including some basic functions.

    Args:
        num_input_channels (int): number of input channels.
        num_output_channels (int): number of output channels.
        num_cond_channels (int, optional): number of condition channels. Defaults to 0.
    """

    def __init__(
        self,
        num_input_channels: int,
        num_output_channels: int,
        num_cond_channels: int = 0,
    ):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.num_cond_channels = num_cond_channels
        self.dropout_layers = None

    @property
    def num_params(self):
        """Returns the number of parameters in the model"""
        return sum(p.size for p in self.parameters() if not p.stop_gradient)

    def forward(self, X: paddle.Tensor, condition: Optional[paddle.Tensor] = None, **kwargs) -> paddle.Tensor:
        """Forward

        Args:
            X (paddle.Tensor): input data tensor of shape (B, *, C_{in}).
            condition (Optional[paddle.Tensor], optional): condition data tensor of shape (B, *, C_{in}). Defaults to None.
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

    def get_dropout_layers(self):
        self.dropout_layers = [m for m in self.sublayers() if isinstance(m, (paddle.nn.Dropout, paddle.nn.Dropout2D))]

    def enable_infer_dropout(self):
        """Adjust all dropout layers to training state"""
        for layer in self.dropout_layers:
            layer.training = True

    def disable_infer_dropout(self):
        """Adjust all dropout layers to non-training state"""
        for layer in self.dropout_layers:
            layer.training = False

    @contextmanager
    def dropout_controller(self, enable):
        """Controls the temporary enable/disable state of the Dropout layer during the model inference phase.

        Args:
            enable (bool): Whether to turn on the controller.
        """
        if self.dropout_layers is None:
            self.get_dropout_layers()

        if enable:
            self.enable_infer_dropout()
        try:
            yield
        finally:
            if enable:
                self.disable_infer_dropout()
