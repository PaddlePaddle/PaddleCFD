import paddle
import paddle.nn as nn
from problem import Stokes


def tile(x, y):
    X = x.tile(repeat_times=[tuple(y.shape)[0], 1])
    Y = paddle.concat(x=[y[i].tile(repeat_times=[tuple(x.shape)[0], 1]) for
        i in range(tuple(y.shape)[0])], axis=0)
    return paddle.concat(x=(X, Y), axis=1)

class BSNN(nn.Layer):
    def __init__(self, layers):
        
        super(BSNN, self).__init__()
        self.layers = layers
        self.num_layers = len(layers)
        self.act = paddle.sin
        self.width = [layers[0]] + [int(pow(2, i - 1) * layers[i]) for i in range(1, len(layers) - 1)] + [layers[-1]]
        self.masks = self.construct_mask()
        
        # Dynamically create layers
        self.layers_list = nn.LayerList()
        for i in range(len(layers) - 1):
            in_features = self.width[i]
            out_features = self.width[i + 1]
            self.layers_list.append(nn.Linear(in_features, out_features))

    def construct_mask(self):
        masks = []
        for l in range(2, self.num_layers - 2):
            num_blocks = int(pow(2, l - 1))
            blocksize1 = int(self.width[l] / num_blocks)
            blocksize2 = 2 * self.layers[l + 1]
            blocks = [paddle.ones(shape=(blocksize1, blocksize2)) for _ in range(num_blocks)]
            mask = paddle.block_diag(blocks)
            masks.append(mask)
        return masks

    def forward(self, x):
        # import pdb;pdb.set_trace()
        for i, layer in enumerate(self.layers_list):
            if 2 <= i < len(self.layers) - 2:
                W2 = layer.weight * self.masks[i - 2]
                x = paddle.add(paddle.matmul(x, W2), layer.bias)
                x = self.act(x)
            elif i == len(self.layers) - 2:
                x = layer(x)
            else:
                x = layer(x)
                x = self.act(x)
        return x
    


class Net_Integral(nn.Layer):
    def __init__(self, layers, shape, ngs_boundary, ngs_interior, problem, eval_mode=False):
        super().__init__()
        self.G = []
        if len(shape) > 1:
            for i in range(shape[0]):
                Row = []
                for j in range(shape[1]):
                    Row.append(BSNN(layers[i][j]))
                self.G.append(Row)
        else:
            raise NotImplementedError
            for i in range(shape[0]):
                self.G.append(BSNN(layers[i]))
        self.ngs_interior = ngs_interior
        self.ngs_boundary = ngs_boundary
        self.problem = problem
        if eval_mode is True:
            self.no_grad = paddle.no_grad
        else:
            from contextlib import contextmanager
            @contextmanager
            def empty_context_manager():
                yield
            self.no_grad = empty_context_manager

    def forward(self, x_in_coord_list, x_in_wts_list, x_bc_wts_list, x_bc_coord_list, z, f_interior_list, g_boundary_list, a_boundary_list, x_bc_normal):
        N_interior = tuple(x_in_coord_list[0].shape)[0]
        N_boundary = tuple(x_bc_coord_list[0].shape)[0]
        fG_quad = paddle.zeros_like(x=z)[:, 0]
        gGn_quad = paddle.zeros_like(x=fG_quad)
        
        def forward_layer(BSNN_interior, BSNN_boundary, fG_quad, gGn_quad):
            with self.no_grad(): # ban grad in eval mode for less GPU memory usage
                for k in range(self.ngs_interior):
                    INPUT_interior = tile(x_in_coord_list[k], z)
                    G_interior = BSNN_interior(INPUT_interior)
                    f_interior = f_interior_list[k]
                    f_interior = f_interior.tile(repeat_times=[z.shape[0], 1])
                    G_interior = G_interior.reshape([-1, N_interior]).transpose([1, 0])
                    f_interior = f_interior.reshape([-1, N_interior]).transpose([1, 0])
                    fG_interior = f_interior * G_interior
                    fG_quad += fG_interior.transpose([1, 0]) @ x_in_wts_list[k]
            for k in range(self.ngs_boundary):
                INPUT_boundary = tile(x_bc_coord_list[k], z)
                INPUT_boundary.stop_gradient = False
                G_boundary = BSNN_boundary(INPUT_boundary).reshape([-1, N_boundary]).transpose([1, 0])
                
                g_boundary = g_boundary_list[k]
                a_boundary = a_boundary_list[k]
                
                if not isinstance(self.problem, Stokes):
                    g_boundary = g_boundary.tile(repeat_times=[z.shape[0], 1])
                    g_boundary = g_boundary.reshape([-1, N_boundary]).transpose([1, 0])
                    a_boundary = a_boundary.tile(repeat_times=[z.shape[0], 1])
                    a_boundary = a_boundary.reshape([-1, N_boundary]).transpose([1, 0])
                else:
                    Ggrad_boundary = paddle.grad(G_boundary, INPUT_boundary, paddle.ones_like(G_boundary), allow_unused=True)[0][:, :3]
                    Gx_boundary = Ggrad_boundary[:, [0]].reshape([N_boundary, -1])
                    Gy_boundary = Ggrad_boundary[:, [1]].reshape([N_boundary, -1])
                    Gz_boundary = Ggrad_boundary[:, [2]].reshape([N_boundary, -1])
                    G_boundary = Gx_boundary * x_bc_normal[:, [0]] + Gy_boundary * x_bc_normal[:, [1]] + Gz_boundary * x_bc_normal[:, [2]]
                gGn_boundary = a_boundary * g_boundary * G_boundary
                gGn_quad += gGn_boundary.transpose([1, 0]) @ x_bc_wts_list[k]
            return fG_quad, gGn_quad
        
        if isinstance(self.problem, Stokes):
            for j in range(len(self.G[0])):
                # import pdb;pdb.set_trace()
                fG_quad, gGn_quad = forward_layer(self.G[0][j], self.G[0][j], fG_quad, gGn_quad)
        else:
            fG_quad, gGn_quad = forward_layer(self.G[0][0], self.G[0][1], fG_quad, gGn_quad)
        
        quad_res = (fG_quad - gGn_quad)[:, None]
        return quad_res
