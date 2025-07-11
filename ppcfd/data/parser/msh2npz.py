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


import re
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ppcfd.data.parser.base_parser import BaseTransition


class MeshSource(Enum):
    """Mesh data source identifier.

    Attributes:
        AUTO: (Default) Enable automatic format detection.
        GMSH: Gmsh open-source mesher.
        ANSYS: ANSYS Mechanical APDL generated mesh.
        OPENFOAM: OpenFOAM polyMesh format.
        GENERAL: Generic format requiring explicit specification of mesh
            topology and nodal coordinates. Use with custom parsers for
            unsupported formats.
    """

    AUTO = "auto"
    GMSH = "gmsh"
    ANSYS = "ansys"
    OPENFOAM = "openfoam"
    GENERAL = "general"


class MshTransition(BaseTransition):
    """Transition class for .msh file.

    Args:
        file_path (Union[str, Path]): path of .msh file.
        save_path (Optional[str], optional): path of saved .npz file. Defaults to None.
        save_data (Optional[bool], optional): Whether to save the .npz file. Defaults to False.
        source (Optional[Union[MeshSource, str]], optional): source of .msh file such as openfoam. Defaults to MeshSource.AUTO.
    """

    file_format = "msh"
    # be empty because `file_path` could be set by loader and all other prarmeters have default value
    required_params = []

    def __init__(
        self,
        file_path: Union[str, Path],
        save_path: Optional[Union[str, Path]] = None,
        save_data: bool = False,
        source: Optional[Union[MeshSource, str]] = MeshSource.AUTO,
        **kwargs,
    ):
        super().__init__(file_path)
        if save_path is not None:
            self.output_path = Path(save_path).absolute()
        else:
            self.output_path = self.file_path

        if isinstance(source, str):
            try:
                self.source = next(src for src in MeshSource if src.value == source.lower())
            except StopIteration:
                raise ValueError(f"Invalid 'source': {source}, optinal values are {[s.value for s in MeshSource]}")
        elif source is None:
            self.source = MeshSource.AUTO

        try:
            self.data = self.load_data()
        except Exception as e:
            raise ValueError(f"Failed to parse the file: {e}")

        if save_data:
            self.save_npz()

    def load_data(self):
        with open(self.file_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        data = {"nodes": None, "elements": None, "element_types": None}
        self.nodes, self.elements, self.elem_types = [], [], []

        if self.source == MeshSource.AUTO:
            self.source = self._auto_source_detection(lines)

        if self.source == MeshSource.GMSH:
            self._parse_gmsh(lines)
        elif self.source == MeshSource.ANSYS:
            self._parse_ansys(lines)
        elif self.source == MeshSource.OPENFOAM:
            self._parse_openfoam(lines)
        else:
            self._parse_general(lines)

        assert (
            len(self.nodes) != 0 or len(self.elements) != 0
        ), "No nodes and elements found. Please check your input file or specify the correct source."

        if len(self.nodes):
            data["nodes"] = np.array(self.nodes, dtype=np.float64)
        if len(self.elements):
            try:
                data["elements"] = np.array(self.elements, dtype=np.int32)
            except Exception:
                print(
                    "Warning: Failed to convert elements to integers, "
                    "try to use object type to allow contain cells of different lengths."
                )
                data["elements"] = np.array(self.elements, dtype=object)
        if len(self.elem_types):
            data["element_types"] = np.array(self.elem_types, dtype=np.int32)

        return data

    def _auto_source_detection(self, lines):
        if any(re.match(r"\$Nodes", line) for line in lines):
            print("Detected mesh source: gmsh")
            return MeshSource.GMSH
        elif any(re.match(r"/\w+,\d+,\w+/", line) for line in lines):
            print("Detected mesh source: ansys")
            return MeshSource.ANSYS
        elif any(line.startswith("FoamFile") for line in lines):
            print("Detected mesh source: openfoam")
            return MeshSource.OPENFOAM
        else:
            print("Warning: Unable to automatically identify the format, try general parsing")
            return MeshSource.GENERAL

    def _parse_gmsh(self, lines):
        section = None
        node_count, elem_count = 0, 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("$Nodes"):
                section = "nodes"
                node_count = int(lines[lines.index(line) + 1])
                self.nodes = np.zeros((node_count, 3), dtype=np.float64)
                continue
            elif line.startswith("$Elements"):
                section = "elements"
                elem_count = int(lines[lines.index(line) + 1])
                continue
            elif line.startswith("$End"):
                section = None
                continue

            if section == "nodes" and line:
                parts = list(map(float, line.split()))
                if len(parts) == 4:  # format: [node number, x, y, z]
                    node_id = int(parts) - 1  # 0-based
                    self.nodes[node_id] = parts[1:]
            elif section == "elements" and line:
                parts = list(map(int, line.split()))

                # can be skipped
                # elem_id = parts  # element ID
                # num_tags = parts
                # tags = parts[3:3 + num_tags]  # tags

                elem_type = parts  # typesuch as 2 = triangle
                nodes_start = 3 + parts  # node starting pos
                elem_nodes = [n - 1 for n in parts[nodes_start:]]  # 0-based

                self.elements.append(elem_nodes)
                self.elem_types.append(elem_type)
        if len(self.elements) != elem_count:
            print(f"Warning: Mismatch between expected elem_count {elem_count} and actual got {len(self.elements)}.")

    def _parse_ansys(self, lines):
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if re.match(r"^\d+\s+-?\d+\.?\d*", line):
                parts = list(map(float, line.split()))
                if len(parts) == 4:  # node line: [node number, x, y, z]
                    self.nodes.append(parts[1:])
            elif re.match(r"^\d+\.0\s+\d+", line):  # cell line: [type, node1, node2,...]
                parts = list(map(float, line.split()))
                elem_type = int(parts)  # type such as 230 = quadrangle
                elem_nodes = [int(n) - 1 for n in parts[1:]]  # 0-based

                self.elements.append(elem_nodes)
                self.elem_types.append(elem_type)

    def _parse_openfoam(self, lines):
        in_vertices = False
        in_cells = False
        current_cell_type = None
        current_cell_nodes = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("vertices"):
                in_vertices = True
                continue
            elif line.startswith("cells"):
                in_cells = True
                continue
            elif line == ")":
                in_vertices = False
                in_cells = False
                if current_cell_type and current_cell_nodes:
                    self.elements.append(current_cell_nodes)
                    self.elem_types.append(current_cell_type)
                current_cell_type = None  # reset
                current_cell_nodes = []

            if in_vertices:
                if line.startswith("("):
                    continue
                coords = list(map(float, line.replace("(", "").replace(")", "").split()))
                for i in range(0, len(coords), 3):
                    self.nodes.append(coords[i : i + 3])
            elif in_cells:
                if line.startswith(("hex", "tet", "prism", "pyramid")):
                    # append last
                    if current_cell_type and current_cell_nodes:
                        self.elements.append(current_cell_nodes)
                        self.elem_types.append(current_cell_type)
                    # init new
                    parts = line.split("(")
                    current_cell_type = parts.strip()
                    nodes_str = parts.split(")") if len(parts) > 1 else ""
                    current_cell_nodes = list(map(int, nodes_str.split()))
                elif current_cell_type and line not in ("(", ")"):
                    nodes_str = line.replace(")", "").strip()
                    current_cell_nodes.extend(map(int, nodes_str.split()))

        # append last cell
        if current_cell_type and current_cell_nodes:
            self.elements.append(current_cell_nodes)
            self.elem_types.append(current_cell_type)

    def _parse_general(self, lines):
        for line in lines:
            if not line or line.isalpha():
                continue
            parts = line.split()
            if 3 <= len(parts) <= 4:
                try:
                    self.nodes.append(list(map(float, parts)))
                except ValueError:
                    pass
            elif len(parts) > 1:
                try:
                    self.elements.append(list(map(int, parts)))
                except ValueError:
                    pass

    def save_npz(self):
        output_path = self.output_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **self.data)

    def get_data(self):
        return self.data
