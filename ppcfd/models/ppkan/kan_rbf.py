import sys
import paddle
from typing import List
from .kan import *

class RadialBasisFunctionLayer(paddle.nn.Layer):

    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 min_grid: float=-1.0, 
                 max_grid: float=1.0, 
                 grid_count: int=5, 
                 apply_base_update:bool=True, 
                 activation: paddle.nn.Layer=paddle.nn.Silu(), 
                 grid_opt:bool=False, 
                 noise_scale: float=0.1, dtype: paddle.dtype='float32'
                 ):
        super().__init__()
        self.apply_base_update = apply_base_update
        self.activation = activation
        self.min_grid = min_grid
        self.max_grid = max_grid
        self.grid_count = grid_count
        self.grid = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.linspace(start=min_grid, stop=max_grid, num=grid_count, dtype=dtype), trainable=grid_opt)
        if not grid_opt:
            self.grid.stop_gradient=True
        self.rbf_weight = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.empty(shape=[in_features * grid_count, out_features], dtype=dtype))
        
        self.scale_rbf = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.ones(shape=out_features, dtype=dtype))

        init_XavierNormal = paddle.nn.initializer.XavierNormal()
        init_XavierNormal(self.rbf_weight)
        self.rbf_weight.data += paddle.randn(shape=self.rbf_weight.shape, dtype=self.rbf_weight.dtype) * noise_scale

        self.scale_base = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.ones(shape=out_features, dtype=dtype))
        self.base_weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=[out_features, in_features], dtype=dtype))
        init_XavierNormal(self.base_weight)
        if not apply_base_update:
            self.scale_base.stop_gradient=True
            self.base_weight.stop_gradient=True
        self.base_activation = paddle.nn.Silu() if apply_base_update else None

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x_unsqueezed = x.unsqueeze(axis=-1)
        rbf_basis = paddle.exp(x=-((x_unsqueezed - self.grid) / ((self.
            max_grid - self.min_grid) / (self.grid_count - 1))) ** 2) # Shape: [batch_size, in_features， grid_count]
        rbf_basis = rbf_basis.contiguous().view(rbf_basis.shape[0], -1) # Shape: [batch_size, in_features * grid_count]
        
        rbf_output = paddle.mm(input=rbf_basis, mat2=self.rbf_weight) # [batch_size, out_features]

        # Apply base update if specified
        if self.apply_base_update:
            base_output = paddle.nn.functional.linear(
                x=self.base_activation(x), weight=self.base_weight.T, bias=None)
            #base_output = base_output.mean(axis=-1, keepdim=True)
            output = (self.scale_base * base_output + self.scale_rbf * rbf_output)
        else:
            output = self.scale_rbf * rbf_output

        return output

class RBF_KAN(paddle.nn.Layer):
    def __init__(self, 
                 hidden_layers: List[int], 
                 min_grid: float=-1.0,
                 max_grid: float=1.0, 
                 grid_count: int=5, 
                 apply_base_update: bool=False, 
                 activation: paddle.nn.Layer=paddle.nn.Silu(), 
                 grid_opt: bool=False, 
                 dtype: paddle.dtype='float32', 
                 noise_scale: float=0.1, 
                 ):

        super().__init__()
        self.layers = paddle.nn.LayerList()
        #First layer with specified min and max grid values
        self.layers.append(RadialBasisFunctionLayer(
            in_features = hidden_layers[0],
            out_features = hidden_layers[1], 
            min_grid = min_grid, 
            max_grid = max_grid, 
            grid_count = grid_count,
            apply_base_update = apply_base_update, 
            activation = activation, 
            grid_opt = grid_opt, 
            noise_scale = noise_scale,
            dtype = dtype))
        # All other layers with default min and max grid values (-1, 1)
        for in_dim, out_dim in zip(hidden_layers[1:-1], hidden_layers[2:]):
            self.layers.append(RadialBasisFunctionLayer(in_dim, out_dim, -
                1.0, 1.0, grid_count, apply_base_update, activation,
                grid_opt, noise_scale, dtype))
        self.dtype = dtype

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        for layer in self.layers[:-1]:
            x = paddle.nn.functional.tanh(x=layer(x))
        x = self.layers[-1](x)
        return x

class KANONet(paddle.nn.Layer):
    def __init__(self, width_trunk, width_branch,  
                 trunk_min_grid, trunk_max_grid,
                 branch_min_grid, branch_max_grid,
                 grid_count, grid_opt, 
                 branch_kan_func, trunk_kan_func,
                 apply_base_update, noise_scale, dtype):
        super(KANONet, self).__init__()
        if trunk_kan_func == 'rbf':
            self.trunk_net = RBF_KAN(hidden_layers=width_trunk, dtype=dtype,
                                    min_grid=trunk_min_grid,
                                    max_grid=trunk_max_grid,
                                    grid_count=grid_count,
                                    grid_opt=grid_opt,
                                    apply_base_update=apply_base_update,
                                    noise_scale=noise_scale
                                    )
        elif trunk_kan_func == 'bspline':
            self.trunk_net = KAN(layers_hidden=width_trunk, 
                                 grid_size=grid_count,
                                 spline_order=3,
                                 scale_noise=noise_scale,
                                 scale_base=0.0,
                                 scale_spline=1.0,
                                 base_activation=paddle.nn.Silu,
                                 grid_range=[trunk_min_grid, trunk_max_grid])
        if branch_kan_func == 'rbf':
            self.branch_net = RBF_KAN(hidden_layers=width_branch, dtype=dtype,
                                    min_grid=branch_min_grid,
                                    max_grid=branch_max_grid,
                                    grid_count=grid_count,
                                    grid_opt=grid_opt,
                                    apply_base_update=apply_base_update,
                                    noise_scale=noise_scale)
        elif branch_kan_func == 'bspline':
            self.branch_net = KAN(layers_hidden=width_branch, 
                                 grid_size=grid_count,
                                 spline_order=3,
                                 scale_noise=noise_scale,
                                 scale_base=0.0,
                                 scale_spline=1.0,
                                 base_activation=paddle.nn.Silu,
                                 grid_range=[branch_min_grid, branch_max_grid])

    def forward(self, x_branch, x_trunk):
        y_branch = self.branch_net(x_branch)
        y_trunk = self.trunk_net(x_trunk)
        if (y_branch.shape)[-1] != (y_trunk.shape)[-1]:
            raise AssertionError("The output dimension of branch and trunk networks are not equal.")
        Y = paddle.einsum('bk,nk->bn', y_branch, y_trunk)
        return Y

        
class RMSLoss(paddle.nn.Layer):
    
    def __init__(self):
        super(RMSLoss, self).__init__()
 
    def forward(self, Y_pred, Y_true):
        if Y_pred.shape != Y_true.shape:
            raise ValueError("The shape of Y_pred and Y_true must be the same.")
        RMSloss = paddle.sqrt(paddle.mean(paddle.square(Y_pred - Y_true)))
        return RMSloss

def view(self, *args, **kwargs):
    if args:
        if len(args)==1 and isinstance(args[0], (tuple, list, str)):
            return paddle.view(self, args[0])
        else:
            return paddle.view(self, list(args))
    elif kwargs:
        return paddle.view(self, shape_or_dtype = list(kwargs.values())[0])

setattr(paddle.Tensor, 'view', view)

def reshape(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert "shape" in kwargs
        return paddle.reshape(self, shape=kwargs["shape"])

setattr(paddle.Tensor, "reshape", reshape)
