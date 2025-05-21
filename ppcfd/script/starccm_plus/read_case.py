# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def read_case(input_file):
    reader = vtk.vtkEnSightGoldBinaryReader()
    reader.SetCaseFileName(input_file)
    reader.Update()
    multi_block = reader.GetOutput()
    paddledict = {}
    check_partial = []

    # find out non-partial block
    for i in range(multi_block.GetNumberOfBlocks()):
        block = multi_block.GetBlock(i)
        check_partial.append(block.GetCellData().GetNumberOfArrays())
    max_element = np.max(check_partial)
    max_index = np.argmax(check_partial)
    no_partial_block = multi_block.GetBlock(max_index)
    for i in range(max_element):
        key = no_partial_block.GetCellData().GetArrayName(i)
        paddledict[key] = None

    def get_centroids(polydata):
        cell_centers = vtk.vtkCellCenters()
        cell_centers.SetInputData(polydata)
        cell_centers.Update()
        numpy_cell_centers = vtk_to_numpy(cell_centers.GetOutput().GetPoints().GetData()).astype(np.float32)
        return numpy_cell_centers

    centroid_list = []
    blockname_list = []
    # collect all block into paddledict
    for i in range(multi_block.GetNumberOfBlocks()):
        block = multi_block.GetBlock(i)
        block_info = multi_block.GetMetaData(i)
        block_name = block_info.Get(vtk.vtkCompositeDataSet.NAME())
        n = block.GetNumberOfCells()
        centroid = get_centroidsd(block)
        centroid_list.append(centroid)
        blockname_list.append(np.full(len(centroid), block_name))
        for key in paddledict.keys():
            val = block.GetCellData().GetArray(key)
            if val is None:  # partial to non-partial
                no_partial_val = no_partial_block.GetCellData().GetArray(key)
                no_partial_val = vtk_to_numpy(no_partial_val)
                no_partial_block_shape = no_partial_val.shape
                if len(no_partial_block_shape) == 1:  # N,1
                    val = np.zeros(
                        [
                            n,
                        ]
                    )
                elif len(no_partial_block_shape) == 2:  # N,3
                    val = np.zeros([n, 3])
                else:
                    raise NotImplementedError(f"Unknown shape of [{key}] with shape {no_partial_block_shape}.")
            else:
                val = vtk_to_numpy(val).astype(np.float32)
            if i == 0:
                paddledict[key] = val
            else:
                paddledict[key] = np.concatenate([paddledict[key], val])
    assert sum([c.shape[0] for c in centroid_list]) == paddledict[key].shape[0]
    new_paddledict = {}
    new_paddledict["centroids"] = np.vstack(centroid_list)
    new_paddledict["blocknames"] = np.hstack(blockname_list)
    if "WallShearStressi" in paddledict.keys():
        new_paddledict["wss"] = np.vstack(
            (
                [
                    paddledict["WallShearStressi"],
                    paddledict["WallShearStressj"],
                    paddledict["WallShearStressk"],
                ]
            )
        ).T
    else:
        raise NotImplementedError

    if "Normali" in paddledict.keys():
        normali = paddledict["Normali"][:, np.newaxis]
        normalj = paddledict["Normalj"][:, np.newaxis]
        normalk = paddledict["Normalk"][:, np.newaxis]
        new_paddledict["normal"] = np.concatenate((normali, normalj, normalk), axis=-1)
    else:
        raise NotImplementedError

    new_paddledict["pressure"] = paddledict["Pressure"]

    if "AreaMagnitude" in paddledict.keys():
        new_paddledict["areas"] = paddledict["AreaMagnitude"]
    else:
        raise NotImplementedError

    return new_paddledict, multi_block, None
