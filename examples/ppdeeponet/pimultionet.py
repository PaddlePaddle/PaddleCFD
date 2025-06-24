import h5py
import numpy as np
import paddle
from Networks.EncoderNet import EncoderCNNet2d
from Solvers import PIMultiONet
from Utils.GenPoints import Point2D
from Utils.Grad import FDM_2d
from Utils.PlotFigure import Plot
from Utils.utils import np2tensor


def setup_seed(seed):
    paddle.seed(seed=seed)
    np.random.seed(seed)


def get_data(data, ndata, dtype, n0=0):
    a = np2tensor(np.array(data["coeff"][..., n0 : n0 + ndata]).T, dtype)
    u = np2tensor(np.array(data["sol"][..., n0 : n0 + ndata]).T, dtype)
    beta1 = np2tensor(np.array(data["beta1"][n0 : n0 + ndata]), dtype)
    beta2 = np2tensor(np.array(data["beta2"][n0 : n0 + ndata]), dtype)
    X, Y = np.array(data["X"]).T, np.array(data["Y"]).T
    mesh = np2tensor(np.vstack([X.flatten(), Y.flatten()]).T, dtype)
    gridx = mesh.reshape([-1, 2])
    x = gridx.tile(repeat_times=(ndata, 1, 1))
    a = a.reshape([ndata, -1, 1])
    u = u.reshape([ndata, -1, 1])
    return a, u, x, gridx, beta1, beta2


class mollifer(object):

    def __inint__(self):
        pass

    def __call__(self, u, x):
        u = u * paddle.sin(x=np.pi * x[..., 0:1]) * paddle.sin(x=np.pi * x[..., 1:2])
        return u


class LossClass(object):

    def __init__(self, solver):
        super(LossClass, self).__init__()
        self.solver = solver
        self.dtype = solver.dtype
        self.device = solver.device
        self.fun_a = fun_a
        self.model_enc = solver.model_dict["enc"]
        self.model_u = solver.model_dict["u"]
        self.mollifer = mollifer()
        self.a_train = a_train.to(self.device)
        self.u_train = u_train.to(self.device)
        self.x_train = x_train.to(self.device)
        self.deltax = 1 / (N_mesh - 1)
        self.deltay = 1 / (N_mesh - 1)

    def Loss_pde(self, index, w_pde):
        """Define the PDE loss"""
        if w_pde > 0.0:
            n_batch = tuple(index.shape)[0]
            x_mesh.tile(repeat_times=[n_batch, 1, 1]).to(self.device).stop_gradient = not True
            x = x_mesh.tile(repeat_times=[n_batch, 1, 1]).to(self.device)
            a = self.fun_a(x, self.a_train[index])
            a = a.reshape([-1, N_mesh, N_mesh, 1])
            u = self.model_u(x, self.model_enc(self.a_train[index]))
            u = self.mollifer(u, x).reshape([-1, N_mesh, N_mesh, 1])
            dudx, dudy = FDM_2d(u, self.deltax, self.deltay)
            adux = a[:, 1:-1, 1:-1, 0:1] * dudx
            aduy = a[:, 1:-1, 1:-1, 0:1] * dudy
            dauxdx, _ = FDM_2d(adux, self.deltax, self.deltay)
            _, dauydy = FDM_2d(aduy, self.deltax, self.deltay)
            left = (-(dauxdx + dauydy)).reshape([n_batch, -1])
            right = 10.0 * paddle.ones_like(x=left)
            return self.solver.getLoss(left, right)
        else:
            return paddle.to_tensor(data=0.0)

    def Loss_data(self, index, w_data):
        return paddle.to_tensor(data=0.0)

    def Error(self, x, a, u):
        try:
            u_pred = self.model_u(x, self.model_enc(a))
        except ValueError:
            u_pred = self.model_u(x)
        u_pred = self.mollifer(u_pred, x)
        return self.solver.getError(u_pred, u)


class Encoder(paddle.nn.Layer):

    def __init__(self, conv_arch: list, fc_arch: list, nx_size: int, ny_size: int, dtype=None):
        super(Encoder, self).__init__()
        self.conv = EncoderCNNet2d(
            conv_arch=conv_arch,
            fc_arch=fc_arch,
            activation_conv="SiLU",
            activation_fc="SiLU",
            nx_size=nx_size,
            ny_size=ny_size,
            kernel_size=(5, 5),
            stride=2,
            dtype=dtype,
        )

    def forward(self, x):
        x = self.conv(x)
        return x


random_seed = 1234
setup_seed(random_seed)
device = "gpu:0"
dtype = "float32"
problem_name = "DarcyFlow_cts2d"
tag = "fdm_TS"
mode = "eval"  # train/eval
data_train = h5py.File("./Problems/DarcyFlow_2d/smh_train.mat", "r")
data_test = h5py.File("./Problems/DarcyFlow_2d/smh_test_in.mat", "r")
res = 29

n_train, n_test = 1000, 200

a_train, u_train, x_train, gridx_train, beta1_train, beta2_train = get_data(data_train, n_train, dtype)
a_test, u_test, x_test, gridx_test, beta1_test, beta2_test = get_data(data_test, n_test, dtype)

pointGen = Point2D(x_lb=[0.0, 0.0], x_ub=[1.0, 1.0], dataType=dtype, random_seed=random_seed)
N_mesh = 29
x_mesh = pointGen.inner_point(N_mesh, method="mesh")

solver = PIMultiONet.Solver(device=device, dtype=dtype)
netType = "MultiONetBatch"
fun_a = solver.getModel_a(
    Exact_a=None,
    approximator="RBF",
    **{"x_mesh": gridx_train, "kernel": "gaussian", "eps": 25.0, "smoothing": 0.0, "degree": 6.0},
)


conv_arch = [1, 32, 64, 128]
fc_arch = [128 * 1 * 1, 128]
model_enc = Encoder(conv_arch, fc_arch, nx_size=res, ny_size=res, dtype=dtype).to(device)
hidden_list, act_x, act_a = [128] * 4, "Tanh_Sin", "Tanh_Sin"
model_u = solver.getModel(
    x_in_size=2, a_in_size=128, hidden_list=hidden_list, activation_x=act_x, activation_a=act_a, netType=netType
)

model_dict = {"u": model_u, "enc": model_enc}

if mode == "train":
    solver.train_setup(model_dict, lr=0.001, optimizer="AdamW", scheduler_type="StepLR", gamma=0.6, step_size=200)
    solver.train_index(
        LossClass,
        a_train,
        u_train,
        x_train,
        a_test,
        u_test,
        x_test,
        w_data=0.0,
        w_pde=1.0,
        batch_size=50,
        epochs=2000,
        epoch_show=50,
        **{"save_path": f"saved_models/PI{netType}_{tag}/"},
    )
else:
    model_u.load_dict(paddle.load("./saved_models/PIMultiONetBatch_fdm_TS/model_u.pdparams"))
    model_enc.load_dict(paddle.load("./saved_models/PIMultiONetBatch_fdm_TS/model_enc.pdparams"))

    x_var = paddle.to_tensor(x_test, stop_gradient=False)
    a_var = a_test.to(device)
    u_pred = model_u(x_var, model_enc(a_var))
    u_pred = mollifer()(u_pred, x_var).detach().cpu()
    #
    print("The shape of a_test:", a_test.shape)
    print("The shape of u_test:", u_test.shape, "u_pred shape", u_pred.shape)
    print("The test loss (avg):", solver.getLoss(u_pred, u_test))
    print("The test l2 error (avg):", solver.getError(u_pred, u_test))
    inx = 0
    # # #######################################
    Plot.show_2d_list(
        [gridx_train] + [gridx_test] * 3,
        [a_test[inx], u_test[inx], u_pred[inx], paddle.abs(u_test[inx] - u_pred[inx])],
        ["a_test", "u_test", "u_pred", "abs u"],
        lb=0.0,
        save_path="./saved_models/PIMultiONetBatch_fdm_TS/result",
    )
    #############################################
    # show loss
    loss_saved = solver.loadLoss(path=f"saved_models/PI{netType}_{tag}/", name="loss_pimultionet")
    Plot.show_loss(
        [loss_saved["loss_train"], loss_saved["loss_test"], loss_saved["loss_data"], loss_saved["loss_pde"]],
        ["loss_train", "loss_test", "loss_data", "loss_pde"],
        save_path="./saved_models/PIMultiONetBatch_fdm_TS/loss",
    )
    # show error
    Plot.show_error(
        [loss_saved["time"]] * 1,
        [loss_saved["error"]],
        ["l2_test"],
        save_path="./saved_models/PIMultiONetBatch_fdm_TS/error",
    )
