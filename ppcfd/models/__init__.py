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

# Import models with graceful fallback for missing dependencies
__all__ = []

# CoNFILD - Conditional Neural Field Latent Diffusion (always available)
try:
    from ppcfd.models import confild

    __all__.append("confild")
except ImportError as e:
    import warnings

    warnings.warn(f"Failed to import confild: {e}")

# ppFNO - Fourier Neural Operator (requires custom C++ extensions)
try:
    from ppcfd.models import ppfno

    __all__.append("ppfno")
except ImportError:
    pass  # Optional dependency

# ppKAN - Kolmogorov-Arnold Networks
try:
    from ppcfd.models import ppkan

    __all__.append("ppkan")
except ImportError:
    pass  # Optional dependency

# ppTransformer
try:
    from ppcfd.models import pptransformer

    __all__.append("pptransformer")
except ImportError:
    pass  # Optional dependency

# ppDiffusion
try:
    from ppcfd.models import ppdiffusion

    __all__.append("ppdiffusion")
except ImportError:
    pass  # Optional dependency

# ppDeepONet
try:
    from ppcfd.models import ppdeeponet

    __all__.append("ppdeeponet")
except ImportError:
    pass  # Optional dependency

# Symbolic Graph Networks
try:
    from ppcfd.models import symbolic_gn

    __all__.append("symbolic_gn")
except ImportError:
    pass  # Optional dependency

# Poseidon - Scientific Operator Transformer
try:
    from ppcfd.models import poseidon

    __all__.append("poseidon")
except ImportError:
    pass  # Optional dependency
