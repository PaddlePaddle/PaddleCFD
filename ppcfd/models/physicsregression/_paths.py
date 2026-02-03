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

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[2]
EXAMPLES_ROOT = PROJECT_ROOT / "examples" / "physics_regression"
EXAMPLE_DATA_ROOT = EXAMPLES_ROOT / "data"
ORACLE_MODEL_ROOT = EXAMPLES_ROOT / "Oracle_model"
FEYNMAN_EQUATIONS_PATH = EXAMPLE_DATA_ROOT / "FeynmanEquations.xlsx"
UNITS_CSV_PATH = EXAMPLE_DATA_ROOT / "units.csv"
PHYSICAL_DATA_ROOT = EXAMPLES_ROOT / "physical" / "data"


def example_data_path(filename: str) -> str:
    return str(EXAMPLE_DATA_ROOT / filename)


def oracle_model_path(name: str, expr_idx: int) -> str:
    return str(ORACLE_MODEL_ROOT / name / f"{name}_{expr_idx}.pth")
