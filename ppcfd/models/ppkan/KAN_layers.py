import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np

class KAN_Bspline(nn.Layer):

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
        super(KAN_Bspline, self).__init__()
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
        #print("A shape: ", A.shape, "B shape: ", B.shape, "Solution shape: ", solution.shape)
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
            x=self.b_splines(x).reshape([x.shape[0],-1]).contiguous(), #.view(x.shape[0], -1), 
            weight=self.scaled_spline_weight.reshape([self.out_features, -1]).T.contiguous())#.view(self.out_features, -1).T)
        # 采用view似乎无法计算一阶微分
        # spline_output = paddle.nn.functional.linear(
        #     x=self.b_splines(x).view(x.shape[0], -1), 
        #     weight=self.scaled_spline_weight.view(self.out_features, -1).T)

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



class KAN_Legendre(nn.Layer):
    def __init__(self, input_features, output_features, max_degree):

        super(KAN_Legendre, self).__init__()
        self.max_degree = max_degree
        self.input_features = input_features
        self.output_features = output_features

        self.poly_weights = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(shape=[max_degree + 1, self.input_features, self.output_features])
        ) # Legendre polynomicals weight matrix, coefficients of each polynominal terms
        init_Orthogonal = nn.initializer.Orthogonal()
        init_Orthogonal(self.poly_weights) # Legendre polynominals satisfy Orthogonality and Completeness

        self.dropout = nn.Dropout(p=0.1)
        self.bias = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=self.output_features)
        )
    
    def forward(self, x):
        batch_size = tuple(x.shape)[0]
        # Legendre polynominals are calculated by the Rodrigues' formula, 
        # given P0 and P1, any Pm can be recurrently calculated according to the ortogonal and complete conditions.
        P_n_minus_2 = paddle.ones(shape=(batch_size, self.input_features)) # P_0
        P_n_minus_1 = x.clone()
        polys = [P_n_minus_2.unsqueeze(axis=-1), P_n_minus_1.unsqueeze(axis=-1)] # lists of polynominals 

        for n in range(2, self.max_degree + 1):
            P_n = ((2*n-1) * x * P_n_minus_1 - (n-1) * P_n_minus_2) / n
            polys.append(P_n.unsqueeze(axis=-1))
            P_n_minus_2 = P_n_minus_1
            P_n_minus_1 = P_n
        polys = paddle.concat(x=polys, axis=-1) # polynominal tensor, shape [batch_size, input_features, max_degree]
        polys = self.dropout(polys)
        output = paddle.einsum('bif, fio->bo', polys, self.poly_weights) + self.bias
        return output


class KAN_Laplace(nn.Layer):

    def __init__(self,
                 input_features,
                 output_features,
                 initial_grid_size,
                 addbias=True):
        super(KAN_Laplace, self).__init__()
        self.input_features = input_features
        self.output_features =output_features
        self.addbias = addbias
        self.grid_size_param = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.to_tensor(data=initial_grid_size, dtype='float32')
        )
        self.laplace_coeffs = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=[2, output_features, input_features, initial_grid_size])
        )
        init_XavierUniform = nn.initializer.XavierUniform()
        init_XavierUniform(self.laplace_coeffs)
        if self.addbias:
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.zeros(shape=[1, output_features])
            )
    
    def forward(self, x):
        grid_size = paddle.clip(x=self.grid_size_param, min=1).round().astype('int32')
        xshape = tuple(x.shape)
        outshape = xshape[:-1] + (self.output_features,)
        x = paddle.reshape(x=x, shape=(-1, self.input_features))
        lambdas = paddle.reshape(x=paddle.linspace(start=0.1, stop=1.0, num=grid_size), shape=(1, 1, 1, grid_size))
        x_rshape = paddle.reshape(x=x, shape=(tuple(x.shape)[0], 1, tuple(x.shape)[1], 1))
        exp_neg = paddle.exp(x=-lambdas * x_rshape)
        exp_pos = paddle.exp(x=lambdas * x_rshape)
        y = paddle.sum(x=exp_neg * self.laplace_coeffs[0:1, :, :, :grid_size], 
                        axis = (-2, -1))
        y += paddle.sum(x=exp_pos * self.laplace_coeffs[0:1, :, :, :grid_size], 
                        axis = (-2, -1)) 
        if self.addbias:
            y += self.bias

        y = paddle.reshape(x=y, shape=outshape)
        return y
        


class KAN_Fourier(nn.Layer):
    def __init__(self, 
                 input_features,
                 output_features,
                 initial_grid_size,
                 addbias=True
                 ):
        super(KAN_Fourier, self).__init__()
        self.addbias=addbias
        self.input_features = input_features
        self.output_features = output_features
        self.grid_size_param = paddle.base.framework.EagerParamBase.from_tensor(
            tensor = paddle.to_tensor(data=initial_grid_size, dtype='float32'))
        self.fourier_coeffs = paddle.base.framework.EagerParamBase.from_tensor(
            tensor = paddle.empty(shape=[2, output_features, input_features, initial_grid_size]))
        init_XavierUniform = nn.initializer.XavierUniform()
        init_XavierUniform(self.fourier_coeffs)
        if self.addbias:
            self.bias = paddle.base.framework.EagerParamBase.from_tensor(
                tensor = paddle.zeros(shape=[1, output_features])
            )
    
    def forward(self, x):
        grid_size = paddle.clip(x=self.grid_size_param, min=1).round().astype(dtype='int32')
        x_shape = tupe(x.shape)
        out_shape = x_shape[:-1] + (self.output_features)
        k = paddle.reshape(x=paddle.arange(start=1, end=grid_size + 1), 
        shape=(1, 1, 1, grid_size))
        x_rshape = paddle.reshape(x=x, shape=(tuple(x.shape)[0], 1, tupe(x.shape)[1], 1))
        c = paddle.cos(x=k * x_rshape)
        s = paddle.sin(x=k * x_rshape)
        y = paddle.sum(x=c * self.fourier_coeffs[0:1, :, :, :grid_size], axis=(-2, -1))
        y += paddle.sum(x=s * self.fourier_coeffs[1:2, :, :, :grid_size], axis=(-2, -1))

        if self.addbias:
            y += self.bias
        y = paddle.reshape(x=y, shape=out_shape)



