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


class LitEma(paddle.nn.Layer):
    """Exponential Moving Average (EMA) module."""

    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")
        self.m_name2s_name = {}
        self.register_buffer(name="decay", tensor=paddle.to_tensor(data=decay, dtype="float32"))
        self.register_buffer(
            name="num_updates",
            tensor=(
                paddle.to_tensor(data=0, dtype="int32") if use_num_upates else paddle.to_tensor(data=-1, dtype="int32")
            ),
        )
        for name, p in model.named_parameters():
            if not p.stop_gradient:
                # remove as '.'-character is not allowed in buffers
                s_name = name.replace(".", "")
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(name=s_name, tensor=p.clone().detach().data)
        self.collected_params = []

    def forward(self, model):
        decay = self.decay
        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with paddle.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())
            for key in m_param:
                if not m_param[key].stop_gradient:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].astype(dtype=m_param[key].dtype)
                    shadow_params[sname].subtract_(
                        y=paddle.to_tensor(one_minus_decay * (shadow_params[sname] - m_param[key]))
                    )
                else:
                    assert key not in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if not m_param[key].stop_gradient:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert key not in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of paddle parameters; the parameters to be temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of paddle parameters; the parameters to be updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    @contextmanager
    def ema_scope(
        self,
        use_ema: bool = False,
        context: Optional[str] = None,
        force_non_ema: bool = False,
        condition: Optional[bool] = None,
    ):
        """Context manager to switch to EMA weights."""
        condition = use_ema if condition is None else condition
        if not (condition and not force_non_ema):
            yield
            return

        original_params = {name: param.clone() for name, param in self.model.named_parameters()}

        self.copy_to(self.model)
        if context:
            print(f"{context}: Switched to EMA weights")

        try:
            yield
        finally:
            for name, param in self.model.named_parameters():
                if name in original_params:
                    param.set_value(original_params[name])
            if context:
                print(f"{context}: Restored training weights")
