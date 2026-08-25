# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multiple Physics Pretraining (MPP) models for PaddleCFD.

This module provides the AViT (Axial Vision Transformer) architecture for
spatiotemporal surrogate modeling with multiple physics pretraining.

Reference:
    "Multiple Physics Pretraining for Spatiotemporal Surrogate Models"
    Michael McCabe et al., NeurIPS 2024.
"""

from .avit import AViT, build_avit

__all__ = [
    "AViT",
    "build_avit",
]
