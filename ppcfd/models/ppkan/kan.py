import paddle
import math
from typing import Optional, List
"""
This is the paddle implementation of Korogonov-Arnold-Network (KAN) 
which is based on the [efficient-kan] by Blealtan and akkashdash 
please refer to their work (https://github.com/Blealtan/efficient-kan)
"""
class KANLinear(paddle.nn.Layer):

    def __init__(
        self, 
        in_features, 
        out_features, 
        grid_size=5, 
        spline_order=3, 
        scale_noise=0.1, 
        scale_base=1.0, 
        scale_spline=1.0,
        enable_standalone_scale_spline=True, 
        base_activation=paddle.nn.Silu,
        grid_eps=0.02, 
        grid_range=[-1, 1]
        ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            paddle.arange(start=-spline_order, end=grid_size + spline_order + 1) * h
             + grid_range[0]
             ).expand(shape=[in_features, -1]).contiguous()
        self.register_buffer(name='grid', tensor=grid)

        self.base_weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=[out_features, in_features]))
        self.spline_weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=[out_features, in_features, grid_size + spline_order]))
        
        if enable_standalone_scale_spline:
            self.spline_scaler = (paddle.base.framework.EagerParamBase.
                from_tensor(tensor=paddle.empty(shape=[out_features,
                in_features])))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        init_KaimingUniform = paddle.nn.initializer.KaimingUniform(
            negative_slope=math.sqrt(5) * self.scale_base, nonlinearity=
            'leaky_relu')
        init_KaimingUniform(self.base_weight)
        with paddle.no_grad():
            noise = (
                paddle.rand(shape=[self.grid_size + 1, self.in_features, self.out_features]) - 1 / 2
                ) * self.scale_noise / self.grid_size

            paddle.assign(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                 * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise),
                output=self.spline_weight.data)

            if self.enable_standalone_scale_spline:
                init_KaimingUniform = paddle.nn.initializer.KaimingUniform(
                    negative_slope=math.sqrt(5) * self.scale_spline,
                    nonlinearity='leaky_relu')
                init_KaimingUniform(self.spline_scaler)

    def b_splines(self, x: paddle.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (paddle.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            paddle.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.shape[1] == self.in_features
        grid: paddle.Tensor = self.grid
        x = x.unsqueeze(axis=-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1] \
             + (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k]) * bases[:, :, 1:]

        assert tuple(bases.shape) == (
            x.shape[0], 
            self.in_features, 
            self.grid_size + self.spline_order)

        return bases.contiguous()

    def curve2coeff(self, x: paddle.Tensor, y: paddle.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (paddle.Tensor): Input tensor of shape (batch_size, in_features).
            y (paddle.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            paddle.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.shape[1] == self.in_features
        assert tuple(y.shape) == (x.shape[0], self.in_features, self.out_features)

        A = self.b_splines(x).transpose(perm=dim2perm(self.b_splines(x).
            ndim, 0, 1)) # [in_features, batch_size, grid_size + spline_order]
        B = y.transpose(perm=dim2perm(y.ndim, 0, 1)) # [in_features, batch_size, out_features]
        solution = paddle.linalg.lstsq(x=A, y=B)[0] # [in_features, grid_size + spline_order, out_features]
        if A.shape[0] == 1:
            solution = solution.unsqueeze(axis=0)
        
        result = solution.transpose([2, 0, 1])
        assert tuple(result.shape) == (
            self.out_features, 
            self.in_features,
            self.grid_size + self.spline_order)

        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(axis=-1) 
            if self.enable_standalone_scale_spline 
            else 1.0)

    def forward(self, x: paddle.Tensor):
        assert x.dim() == 2 and x.shape[1] == self.in_features

        base_output = paddle.nn.functional.linear(
            x=self.base_activation(x),
            weight=self.base_weight.T)
            
        spline_output = paddle.nn.functional.linear(
            x=self.b_splines(x).reshape([x.shape[0],-1]).contiguous(), 
            weight=self.scaled_spline_weight.reshape([self.out_features, -1]).T.contiguous())
        
        return base_output + spline_output

    @paddle.no_grad()
    def update_grid(self, x: paddle.Tensor, margin=0.01):
        assert x.dim() == 2 and x.shape[1] == self.in_features
        batch = x.shape[0]

        splines = self.b_splines(x) # [batch, in, coeff]
        splines = splines.transpose(perm=[1, 0, 2]) # [in, batch, coeff]
        orig_coeff = self.scaled_spline_weight # [out, in, coeff]
        orig_coeff = orig_coeff.transpose(perm=[1, 2, 0]) # [in, coeff, out]
        unreduced_spline_output = paddle.bmm(x=splines, y=orig_coeff) # [in, batch, out]
        unreduced_spline_output = unreduced_spline_output.transpose(perm=[1, 0, 2]) # [batch, in, out]

        # sort each channel individually to collect data distribution
        x_sorted = (paddle.sort(x=x, axis=0), paddle.argsort(x=x, axis=0))[0]
        grid_adaptive = x_sorted[
                                paddle.linspace(start=0, stop=batch - 1, 
                                num=self.grid_size + 1, dtype='int64')
                                ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = paddle.arange(
            dtype='float32', end=self.grid_size + 1
            ).unsqueeze(axis=1) * uniform_step + x_sorted[0] - margin

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = paddle.concat(x=[grid[:1] - uniform_step * paddle.arange(start=
            self.spline_order, end=0, step=-1).unsqueeze(axis=1), grid, grid[-1
            :] + uniform_step * paddle.arange(start=1, end=self.spline_order + 
            1).unsqueeze(axis=1)], axis=0)

        paddle.assign(grid.T, output=self.grid)
        paddle.assign(self.curve2coeff(x, unreduced_spline_output), output=self
            .spline_weight.data)

    def regularization_loss(self, regularize_activation=1.0,
        regularize_entropy=1.0):
        """
        Compute the regularization loss.

        L1 and the entropy loss is for the feature selection, i.e., let the weight of the activation function be small.
        """
        l1_fake = self.spline_weight.abs().mean(axis=-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -paddle.sum(x=p * p.log())
        return (regularize_activation * regularization_loss_activation + 
            regularize_entropy * regularization_loss_entropy)


class KAN(paddle.nn.Layer):

    def __init__(
        self, 
        layers_hidden, 
        grid_size=5, 
        spline_order=3,
        scale_noise=0.1, 
        scale_base=1.0, 
        scale_spline=1.0, 
        base_activation=paddle.nn.Silu, 
        grid_eps=0.02, 
        grid_range=[-1, 1]
        ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.layers = paddle.nn.LayerList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features, 
                    out_features,
                    grid_size=grid_size, 
                    spline_order=spline_order, 
                    scale_noise=scale_noise, 
                    scale_base=scale_base, 
                    scale_spline=scale_spline, 
                    base_activation=base_activation, 
                    grid_eps=grid_eps, 
                    grid_range=grid_range))

    def forward(self, x: paddle.Tensor, update_grid=False):
        for index, layer in enumerate(self.layers):
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
            if index < len(self.layers) - 1:
                x = paddle.nn.functional.tanh(x=x)
        return x

    def regularization_loss(self, regularize_activation=1.0,
        regularize_entropy=1.0):
        return sum(layer.regularization_loss(regularize_activation,
            regularize_entropy) for layer in self.layers)


class KANONet(paddle.nn.Layer):

    def __init__(
        self, 
        width_branch1,
        grid_range_b1,
        grid_size_b1,
        width_trunk,
        grid_range_t,
        grid_size_t,
        spline_order=3,
        width_branch2: Optional[List[int]] = None,
        grid_range_b2=None,
        grid_size_b2=None,
        width_branch3: Optional[List[int]] = None,
        grid_range_b3=None,
        grid_size_b3=None,
        **kwargs
        ):
        super(KANONet, self).__init__()
        self.hidden_out = width_branch1[-1]
        self.branch2 = False
        self.branch3 = False
        self.branch_net1 = KAN(
            layers_hidden=width_branch1,
            grid_size=grid_size_b1,
            spline_order=spline_order,
            grid_range=grid_range_b1,
            **kwargs
        )
        self.trunk_net = KAN(
            layers_hidden=width_trunk,
            grid_size=grid_size_t,
            spline_order=spline_order,
            grid_range=grid_range_t,
            **kwargs
        )
        if width_branch2 is not None:
            self.branch2 = True
            self.branch_net2 = KAN(
                layers_hidden=width_branch2,
                grid_size=grid_size_b2,
                spline_order=spline_order,
                grid_range=grid_range_b2,
                **kwargs
            )
        if width_branch3 is not None:
            self.branch_net3 = True
            self.branch_net3 = KAN(
                layers_hidden=width_branch3,
                grid_size=grid_size_b3,
                spline_order=spline_order,
                grid_range=grid_range_b3,
                **kwargs
            )
    def forward(self, inputs):
        b1_out = self.branch_net1(inputs['branch1'])
        if self.branch2:
            b2_out = self.branch_net2(inputs['branch2'])
        if self.branch3:
            b3_out = self.branch_net3(inputs['branch3'])

        t_out = self.trunk_net(inputs['trunk'])    # [Ni, 64]
        
        step = int(self.hidden_out/4)
        if not self.branch2:
            y1 = paddle.sum(b1_out[:,0:step] * t_out[:,0:step], axis=1, keepdim=True) # [Ni, 1])
            y2 = paddle.sum(b1_out[:,step:2*step] * t_out[:,step:2*step], axis=1, keepdim=True) # [Ni, 1])
            y3 = paddle.sum(b1_out[:,2*step:3*step] * t_out[:,2*step:3*step], axis=1, keepdim=True) # [Ni, 1]
            y4 = paddle.sum(b1_out[:,3*step:] * t_out[:,3*step:], axis=1, keepdim=True) # [Ni, 1]
        elif not self.branch3:
            y1 = paddle.sum(b1_out[:,0:step] * b2_out[:,0:step] * t_out[:,0:step], axis=1, keepdim=True) # [Ni, 1])
            y2 = paddle.sum(b1_out[:,step:2*step] * b2_out[:,0:step] * t_out[:,step:2*step], axis=1, keepdim=True) # [Ni, 1])
            y3 = paddle.sum(b1_out[:,2*step:3*step] * b2_out[:,0:step] * t_out[:,2*step:3*step], axis=1, keepdim=True) # [Ni, 1]
            y4 = paddle.sum(b1_out[:,3*step:] * b2_out[:,0:step] * t_out[:,3*step:], axis=1, keepdim=True) # [Ni, 1]
        else:
            y1 = paddle.sum(b1_out[:,0:step] * b2_out[:,0:step] * b3_out[:,0:step] * t_out[:,0:step], axis=1, keepdim=True) # [Ni, 1])
            y2 = paddle.sum(b1_out[:,step:2*step] * b2_out[:,0:step] * b3_out[:,0:step] * t_out[:,step:2*step], axis=1, keepdim=True) # [Ni, 1])
            y3 = paddle.sum(b1_out[:,2*step:3*step] * b2_out[:,0:step] * b3_out[:,0:step] * t_out[:,2*step:3*step], axis=1, keepdim=True) # [Ni, 1]
            y4 = paddle.sum(b1_out[:,3*step:] * b2_out[:,0:step] * b3_out[:,0:step] * t_out[:,3*step:], axis=1, keepdim=True) # [Ni, 1]
        return paddle.concat([y1, y2, y3, y4], axis=1)  # [Ni, 4]

def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm