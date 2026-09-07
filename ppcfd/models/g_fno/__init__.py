"""G-FNO model package for PaddleCFD."""

from .FNO import FNO2d
from .FNO import FNO3d
from .GCNN import GCNN2d
from .GCNN import GCNN3d
from .GFNO import GFNO2d
from .GFNO import GFNO3d
from .Ghybrid import Ghybrid2d
from .paddle_utils import resolve_runtime_device
from .paddle_utils import set_runtime_device
from .radialNO import radialNO2d
from .radialNO import radialNO3d


__all__ = [
    "FNO2d",
    "FNO3d",
    "GCNN2d",
    "GCNN3d",
    "GFNO2d",
    "GFNO3d",
    "Ghybrid2d",
    "radialNO2d",
    "radialNO3d",
    "resolve_runtime_device",
    "set_runtime_device",
]
