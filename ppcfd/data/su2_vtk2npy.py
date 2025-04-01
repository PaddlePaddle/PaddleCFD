import sys
from os import PathLike
from typing import Sequence, Dict, Union, Tuple, List
import vtk
from vtk.numpy_interface import dataset_adapter as dsa

import numpy as np
import paddle

UnionTensor = Union[np.ndarray, paddle.Tensor]
"""
convert su2 simulation results to numpy array
reference:  https://github.com/locuslab/cfd-gcn/blob/master/mesh_utils.py
author: guhaohao(guhaohao@baidu.com)
"""
# default arguments values
SU2_SHAPE_IDS = {
    "line": 3,
    "triangle": 5,
    "quad": 9,
} # standard identifier for SU2 grid file format, which is defined in the SU2 official documents.
default_dtype = paddle.float32
default_keys = ['x', 'y', 'z'] # default keys of values to extract from su2 mesh file and vtk flow field data file.

def su2_to_graph(data, mesh, **kwargs):
    """
    将SU2网格和VTK数据转换为图结构数据
    convert SU2 mesh and VTK data to graph structure data
    
    Args:
        data (str): path to vtk data file(.vtu/.vtk), which contains flow field data
        mesh (str): path to SU2 mesh file(.su2), define the topological structure of the mesh
        **kwargs: optional arguments, including:
            - dtype: default paddle.float32
            - keys: intresting variables name keys, like ['x','y','z']
            - device: the device to be used for the tensor data. default is 'cpu'
    
    Returns:
        List: [nodes, edges, keys, values]
              - nodes: number of nodes (int)
              - edges: edges connection matrix (np.ndarray, shape=[2, num_edges])
              - keys: extracted variable names list (List[str])
              - values: extracted variable value tensors (List[paddle.Tensor])
    
    Example:
        >>> graph = su2_to_graph("path/to/flow.vtu", "path/to/mesh.su2", 
        >>>                     dtype=paddle.float64, 
        >>>                     keys=['x','y','pressure'],
        >>>                     device='gpu:0')
    """  
    dtype = kwargs.get('dtype') if 'dtype' in kwargs else default_dtype
    device = kwargs.get('device') if 'device' in kwargs else 'cpu'
    paddle.set_device(device)
    nodes, edges, _, _ = get_su2mesh_graph(mesh)
    nodes = len(nodes)
    if not isinstance(data, str):
        raise TypeError(f"Expected a string path to .vtu or .vtk file, got {type(data)}")
    if not isinstance(mesh, str):
        raise TypeError(f"Expected a string path to .su2 file, got {type(mesh)}")

    # Read VTK data file
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(data)
    reader.Update()
    output = reader.GetOutput()
    out = dsa.WrapDataObject(output)

    keys = kwargs.get('keys', default_keys)
    keys += out.PointData.keys() # add all point data keys to keys list
    values = []
    for key in keys:
        if key == 'x':
            tensor = paddle.to_tensor(out.Points[:, 0], dtype=dtype)
        elif key == 'y':
            tensor = paddle.to_tensor(out.Points[:, 1], dtype=dtype)
        elif key == 'z':
            tensor = paddle.to_tensor(out.Points[:, 2], dtype=dtype)
        else:
            tensor = paddle.to_tensor(out.PointData[key], dtype=dtype)
        values.append(value)

    return [nodes, edges, keys, values]

def get_su2mesh_graph(mesh_filename: Union[PathLike, str],
                      dtype: np.dtype = np.float32
                      ) -> Tuple[np.ndarray, np.ndarray, List[List[List[int]]], Dict[str, List[List[int]]]]:
    """
    从SU2网格文件中读取节点、边信息并返回相应的数据结构
    get nodes, edges information from SU2 mesh file and return corresponding graph data structure
    
    Args:
        mesh_filename (Union[PathLike, str]):  path to su2 mesh file
    
    Returns:
        Tuple[List[int], np.ndarray, List[str], List[str]]: 
                                                            - nodes: nodes array (np.ndarray, [N, 2])
                                                            - edges: edge connection matrix, ids of 2 connected nodes (np.ndarray, [2, E])
                                                            - elems: element topology list. 
                                                                out_list: element types. [triangles, quads], mid_list: element of single type, in_list: single element nodes id list. 
                                                                e.g. [
                                                                      [[0, 1, 2], [1, 2, 3]],  # 2 triangle elements
                                                                      [[4, 5, 6, 7]]           # 1 quad element
                                                                      ]
                                                            - marker_dict: boudary marder dictionary. 
                                                              keys (str): boundary marker names (e.g. 'wall', 'inlet')
                                                              values (List[List[int]]): edges lists of each marker
    """
    def get_rhs(s: str) -> str:
        return s.split('=')[-1]

    marker_dict = {}
    with open(mesh_filename) as f:
        for line in f:
            if line.startswith('NPOIN'):
                num_points = int(get_rhs(line))
                mesh_points = [[float(p) for p in f.readline().split()[:2]]
                               for _ in range(num_points)]
                nodes = np.array(mesh_points, dtype=dtype)

            if line.startswith('NMARK'):
                num_markers = int(get_rhs(line))
                for _ in range(num_markers):
                    line = f.readline()
                    assert line.startswith('MARKER_TAG')
                    marker_tag = get_rhs(line).strip()
                    num_elems = int(get_rhs(f.readline()))
                    marker_elems = [[int(e) for e in f.readline().split()[-2:]]
                                    for _ in range(num_elems)]
                    # marker_dict[marker_tag] = np.array(marker_elems, dtype=np.long).transpose()
                    marker_dict[marker_tag] = marker_elems

            if line.startswith('NELEM'):
                edges = []
                triangles = []
                quads = []
                num_edges = int(get_rhs(line))
                for _ in range(num_edges):
                    elem = [int(p) for p in f.readline().split()]
                    if elem[0] == SU2_SHAPE_IDS['triangle']:
                        n = 3
                        triangles.append(elem[1:1+n])
                    elif elem[0] == SU2_SHAPE_IDS['quad']:
                        n = 4
                        quads.append(elem[1:1+n])
                    else:
                        raise NotImplementedError
                    elem = elem[1:1+n]
                    edges += [[elem[i], elem[(i+1) % n]] for i in range(n)]
                edges = np.array(edges, dtype=np.int32).transpose()
                # triangles = np.array(triangles, dtype=np.long)
                # quads = np.array(quads, dtype=np.long)
                elems = [triangles, quads]

    return nodes, edges, elems, marker_dict