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

from functools import wraps

from ppcfd.models.ppdiffusion import metrics
from ppcfd.models.ppdiffusion import modules
from ppcfd.models.ppdiffusion import process
from ppcfd.models.ppdiffusion.base_model import BaseModel
from ppcfd.models.ppdiffusion.unet_simple import UNet as SimpleUnet


__all__ = [
    "BaseModel",
    "SimpleUnet",
    "modules",
    "process",
    "metrics",
]


def auto_adapt_dataparallel(model, custom_funcs=[]):
    if not hasattr(model, "_layers"):
        return model

    original_class = model._layers.__class__

    for name in custom_funcs:
        if hasattr(model._layers, name):
            original_method = getattr(original_class, name)

            @wraps(original_method)
            def wrapper(*args, **kwargs):
                real_self = model._layers
                return original_method(real_self, *args[1:], **kwargs)

            setattr(model.__class__, name, wrapper)

    return model
