from __future__ import annotations

from typing import Tuple
from typing import Union

import paddle
import paddle.nn as nn

from ppsci.arch import activation as act_mod
from ppsci.arch import base
from ppsci.arch import mlp
from ppsci.arch import kan

import numpy as np


class ChipKANONet(base.Arch):

    def __init__(
        self,
        branch_input_keys: Tuple[str, ...],
        BCtype_input_keys: Tuple[str, ...],
        BC_input_keys: Tuple[str, ...],
        trunk_input_keys: Tuple[str, ...],
        output_keys: Tuple[str, ...],
        num_loc: int,
        bctype_loc: int,
        BC_num_loc: int,
        num_features: int,
        branch_hidden_layers: Tuple[int, ...],
        BCtype_hidden_layers: Tuple[int, ...],
        BC_hidden_layers: Tuple[int, ...],
        trunk_hidden_layers: Tuple[int, ...],
        branch_grid_size: int,
        branch_grid_range: Tuple[float, float],
        BCtype_grid_size: int,
        BCtype_grid_range: Tuple[float, float],
        BC_grid_size: int,
        BC_grid_range: Tuple[float, float],
        trunk_grid_size: int,
        trunk_grid_range: Tuple[float, float],
        branch_activation: str = "tanh",
        BC_activation: str = "tanh",
        trunk_activation: str = "tanh",
        use_bias: bool = True,
        sup_res: bool = False,
        N_sup: int = 20,
    ):
        super().__init__()
        self.trunk_input_keys = trunk_input_keys
        self.branch_input_keys = branch_input_keys
        self.BCtype_input_keys = BCtype_input_keys
        self.BC_input_keys = BC_input_keys
        self.input_keys = (
            self.trunk_input_keys 
            + self.branch_input_keys
            + self.BC_input_keys 
            + self.BCtype_input_keys,
        )
        self.output_keys = output_keys

        self.sup_res = sup_res
        self.N_sup = N_sup
        
        branch_layers = [num_loc, *branch_hidden_layers, num_features]
        self.branch_act = act_mod.get_activation(branch_activation)
        self.branch_net = kan.KAN(
            layers_hidden=branch_layers,
            input_keys=self.branch_input_keys,
            output_keys=("b",),
            grid_size=branch_grid_size,
            grid_range=branch_grid_range,
            #base_activation=self.branch_act,
        )
        
        BCtype_layers = [bctype_loc, *BCtype_hidden_layers, num_features]
        self.bc_act = act_mod.get_activation(BC_activation)
        self.BCtype_net = kan.KAN(
            layers_hidden=BCtype_layers,
            input_keys=self.BCtype_input_keys,
            output_keys=("bctype",),
            grid_size=BCtype_grid_size,
            grid_range=BCtype_grid_range,
            #base_activation=self.bc_act,
        )
        
        BC_layers = [BC_num_loc, *BC_hidden_layers, num_features]
        self.BC_net = kan.KAN(
            layers_hidden=BC_layers,
            input_keys=self.BC_input_keys,
            output_keys=("bc",),
            grid_size=BC_grid_size,
            grid_range=BC_grid_range,
            #base_activation=self.bc_act,
        )

        trunk_layers = [len(self.trunk_input_keys), *trunk_hidden_layers, num_features]
        self.trunk_act = act_mod.get_activation(trunk_activation)
        self.trunk_net = kan.KAN(
            layers_hidden=trunk_layers,
            input_keys=self.trunk_input_keys,
            output_keys=("t",),
            grid_size=trunk_grid_size,
            grid_range=trunk_grid_range,
            #base_activation=self.trunk_act,
        )
        self.use_bias = use_bias
        if use_bias:
            self.b = self.create_parameter(
                shape=(1, ), 
                attr=nn.initializer.Constant(0.0),
            )

    def forward(self, x):
        if self._input_transform is not None:
            x = self._input_transform(x)

        # Encode the input functions with branch nets
        # if Super resolution is used, downsample the grid and boundary conditions
        if self.sup_res:
            x[self.branch_input_keys[0]] = downsample_grid(x[self.branch_input_keys[0]], self.N_sup, 20)
            x[self.BC_input_keys[0]] = downsample_boundary(x[self.BC_input_keys[0]], self.N_sup, 20)
        u_features = self.branch_net(x)[self.branch_net.output_keys[0]]
        bc_features = self.BC_net(x)[self.BC_net.output_keys[0]]
        bctype_features = self.BCtype_net(x)[self.BCtype_net.output_keys[0]]
        # Encode the domain of the output function with trunk net
        y_features = self.trunk_net(x)[self.trunk_net.output_keys[0]]
        y_features = self.trunk_act(y_features)

        # Decode the output function with the combination of the branch nets
        G_u = paddle.sum(
            u_features * y_features * bc_features * bctype_features,
            axis=1,
            keepdim=True,
        )
        # Add the bias term if needed
        if self.use_bias:
            G_u += self.b

        result_dict = {
            self.output_keys[0]: G_u,
        }
        if self._output_transform is not None:
            result_dict = self._output_transform(x, result_dict)

        return result_dict

def downsample_grid(tensor, N, n):
    
    batch_size = tensor.shape[0]
    
    assert N > 2 and n > 2 and n <= N, "Grid size must satisfy n > 2 and n <= N"
    
    if n == N:
        return tensor
    
    grid = tensor.reshape([batch_size, N-2, N-2])
    
    if n - 2 == 1:
        i_indices = [(N-2) // 2]
    else:
        i_indices = np.linspace(0, N-3, n-2, dtype=int)
    
    sampled_points = []
    for i in i_indices:
        for j in i_indices:
            sampled_points.append((i, j))

    result = paddle.zeros([batch_size, (n-2)*(n-2)], dtype=tensor.dtype)
    for idx, (i, j) in enumerate(sampled_points):
        result[:, idx] = grid[:, i, j]
    
    return result

def downsample_boundary(tensor, N, n):
    batch_size = tensor.shape[0]
    assert N > 1 and n > 1 and n <= N, "Grid size must satisfy n > 1 and N >= n"
    if n == N:
        return tensor
    boundary = tensor.reshape([batch_size, 4, N-1])
    if n-1 == 1:
        indices = [(N-1) // 2]
    else:
        indices = np.linspace(0, N-2, n-1, dtype=int)
    
    result = paddle.zeros([batch_size, 4, n-1], dtype=tensor.dtype)
    for edge in range(4):
        for i, idx in enumerate(indices):
            result[:, edge, i] = boundary[:, edge, idx]

    result = result.reshape([batch_size, (n-1)*4])
    
    return result

