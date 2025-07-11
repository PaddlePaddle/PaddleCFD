import os
import time

import paddle
import scipy.io
from Solvers import Module
from tqdm import trange
from Utils.Losses import MyError
from Utils.Losses import MyLoss
from Utils.RBFInterpolatorMesh import RBFInterpolator

from ppcfd.models.ppdeeponet.DeepONets import DeepONetBatch
from ppcfd.models.ppdeeponet.FCNet import FCNet
from ppcfd.models.ppdeeponet.MultiONets import MultiONetBatch
from ppcfd.models.ppdeeponet.MultiONets import MultiONetBatch_X


class Solver(Module.Solver):

    def __init__(self, device="gpu:0", dtype="float32"):
        self.device = device
        self.dtype = dtype
        self.iter = 0
        self.time_list = []
        self.loss_train_list = []
        self.loss_test_list = []
        self.loss_data_list = []
        self.loss_pde_list = []
        self.error_list = []
        self.error_setup()
        self.loss_setup()

    def loadModel(self, path: str, name: str):
        """Load trained model"""
        return paddle.load(path=str(path + f"{name}.pdparams"))

    def saveModel(self, path: str, name: str, model_dict: dict):
        """Save trained model (the whole model)"""
        if not os.path.exists(path):
            os.makedirs(path)
        paddle.save(obj=model_dict["u"].state_dict(), path=path + "model_u.pdparams")
        paddle.save(obj=model_dict["a"].state_dict(), path=path + "model_a.pdparams")
        paddle.save(obj=model_dict["enc"].state_dict(), path=path + "model_enc.pdparams")

    def loadLoss(self, path: str, name: str):
        """Load saved losses"""
        loss_dict = scipy.io.loadmat(path + f"{name}.mat")
        return loss_dict

    def saveLoss(self, path: str, name: str):
        """Save losses"""
        dict_loss = {}
        dict_loss["loss_train"] = self.loss_train_list
        dict_loss["loss_test"] = self.loss_test_list
        dict_loss["loss_data"] = self.loss_data_list
        dict_loss["loss_pde"] = self.loss_pde_list
        dict_loss["error"] = self.error_list
        dict_loss["time"] = self.time_list
        scipy.io.savemat(path + f"{name}.mat", dict_loss)

    def callBack(self, loss_train, loss_data, loss_pde, loss_test, error_test, t_start):
        """call back"""
        self.loss_train_list.append(loss_train.item())
        self.loss_test_list.append(loss_test.item())
        self.loss_data_list.append(loss_data.item())
        self.loss_pde_list.append(loss_pde.item())
        self.time_list.append(time.time() - t_start)
        if isinstance(error_test, list):
            errs = [err.item() for err in error_test]
            self.error_list.append(errs)
        else:
            self.error_list.append(error_test.item())

    def error_setup(self, err_type: str = "lp_rel", d: int = 2, p: int = 2, size_average=True, reduction=True):
        """setups of error
        Input:
            err_type: from {'lp_rel', 'lp_abs'}
        """
        Error = MyError(d, p, size_average, reduction)
        if err_type == "lp_rel":
            self.getError = Error.Lp_rel
        elif err_type == "lp_abs":
            self.getError = Error.LP_abs
        else:
            raise NotImplementedError(f"{err_type} has not defined.")

    def loss_setup(self, loss_type: str = "mse_org", size_average=True, reduction=True):
        """setups of loss
        Input:
            loss_type: from {'mse_org', 'mse_rel'}
        """
        Loss = MyLoss(size_average, reduction)
        if loss_type == "mse_org":
            self.getLoss = Loss.mse_org
        elif loss_type == "mse_rel":
            self.getLoss = Loss.mse_rel
        else:
            raise NotImplementedError(f"{loss_type} has not defined.")

    def getModel_a(self, Exact_a: object = None, approximator: str = "RBF", **kwrds):
        """The model for coefficient a"""
        if Exact_a is not None:
            print("Using the exact definition of a.")
            model_a = Exact_a
        elif approximator == "RBF":
            print("self.device", self.device)
            x_mesh = kwrds["x_mesh"].to(self.device)
            model_a = RBFInterpolator(
                x_mesh=x_mesh,
                kernel=kwrds["kernel"],
                eps=kwrds["eps"],
                degree=kwrds["degree"],
                smoothing=kwrds["smoothing"],
                dtype=self.dtype,
            ).to(self.device)
        else:
            raise NotImplementedError(f"No such approximator: {approximator}.")
        return model_a

    def getModel(
        self,
        x_in_size: int,
        a_in_size: int,
        hidden_list: list,
        latent_size: int = None,
        out_size: int = 1,
        activation_x="ReLU",
        activation_a="Tanh",
        netType: str = "MultiONetBatch",
        **kwrds,
    ):
        """Get the neural network model"""
        if netType == "MultiONetBatch":
            model = MultiONetBatch(
                in_size_x=x_in_size,
                in_size_a=a_in_size,
                hidden_list=hidden_list,
                activation_x=activation_x,
                activation_a=activation_a,
                dtype=self.dtype,
                **kwrds,
            )
        elif netType == "MultiONetBatch_X":
            model = MultiONetBatch_X(
                in_size_x=x_in_size,
                in_size_a=a_in_size,
                latent_size=latent_size,
                out_size=out_size,
                hidden_list=hidden_list,
                activation_x=activation_x,
                activation_a=activation_a,
                dtype=self.dtype,
                **kwrds,
            )
        elif netType == "DeepONetBatch":
            model = DeepONetBatch(dtype=self.dtype, **kwrds)
        elif netType == "FCNet":
            model = FCNet(dtype=self.dtype, **kwrds)
        else:
            raise NotImplementedError
        return model.to(self.device)

    def train_setup(
        self,
        model_dict: dict,
        lr: float = 0.001,
        optimizer="Adam",
        scheduler_type: str = None,
        step_size=500,
        gamma=1 / 3,
        patience=20,
        factor=1 / 2,
    ):
        """Setups for training"""
        self.model_dict = model_dict
        param_list = []
        for model in model_dict.values():
            param_list += list(model.parameters())
        if optimizer == "Adam":
            self.optimizer = paddle.optimizer.Adam(parameters=param_list, learning_rate=lr, weight_decay=0.0001)
        elif optimizer == "AdamW":
            self.optimizer = paddle.optimizer.AdamW(parameters=param_list, learning_rate=lr, weight_decay=0.0001)
        elif optimizer == "RMSprop":
            self.optimizer = paddle.optimizer.RMSProp(
                parameters=param_list, learning_rate=lr, weight_decay=0.0001, epsilon=1e-08, rho=0.99
            )
        else:
            raise NotImplementedError
        if scheduler_type == "StepLR":
            tmp_lr = paddle.optimizer.lr.StepDecay(
                step_size=step_size, gamma=gamma, last_epoch=-1, learning_rate=self.optimizer.get_lr()
            )
            self.optimizer.set_lr_scheduler(tmp_lr)
            self.scheduler = tmp_lr
        elif scheduler_type == "Plateau":
            tmp_lr = paddle.optimizer.lr.ReduceOnPlateau(
                mode="min", factor=factor, patience=patience, learning_rate=self.optimizer.get_lr()
            )
            self.optimizer.set_lr_scheduler(tmp_lr)
            self.scheduler = tmp_lr
        self.scheduler_type = scheduler_type
        self.t_start = time.time()
        self.best_err_test = 10000000000.0

    def train(
        self,
        LossClass: Module.LossClass,
        a_train,
        u_train,
        x_train,
        a_test,
        u_test,
        x_test,
        w_data: float = 1.0,
        w_pde: float = 1.0,
        batch_size: int = 100,
        epochs: int = 1,
        epoch_show: int = 10,
        **kwrds,
    ):
        """Train the model"""
        train_loader = self.dataloader(a_train, u_train, x_train, batch_size=batch_size, shuffle=False)
        for epoch in trange(epochs):
            loss_train_sum, loss_data_sum, loss_pde_sum = 0.0, 0.0, 0.0
            for a, u, x in train_loader:
                lossClass = LossClass(self)
                a, u, x = a.to(self.device), u.to(self.device), x.to(self.device)
                loss_pde = lossClass.Loss_pde(a, w_pde)
                loss_data = lossClass.Loss_data(x, a, u, w_data)
                loss_train = w_data * loss_data + w_pde * loss_pde
                self.optimizer.clear_gradients(set_to_zero=False)
                loss_train.backward()
                self.optimizer.step()
                self.iter += 1
                loss_train_sum += loss_train
                loss_data_sum += loss_data
                loss_pde_sum += loss_pde
            a, u, x = a_test.to(self.device), u_test.to(self.device), x_test.to(self.device)
            lossClass = LossClass(self)
            try:
                with paddle.no_grad():
                    loss_test = lossClass.Loss_data(x, a, u, w_data=1.0)
                    error_test = lossClass.Error(x, a, u)
            except ValueError:
                loss_test = lossClass.Loss_data(x, a, u, w_data=1.0)
                error_test = lossClass.Error(x, a, u)
            self.callBack(
                loss_train_sum / len(train_loader),
                loss_data_sum / len(train_loader),
                loss_pde_sum / len(train_loader),
                loss_test,
                error_test,
                self.t_start,
            )
            if isinstance(error_test, list):
                error_test = sum(error_test) / len(error_test)
            if error_test.item() < self.best_err_test:
                self.best_err_test = error_test.item()
                self.saveModel(kwrds["save_path"], "model_dgno_besterror", self.model_dict)
            if self.scheduler_type is None:
                pass
            elif self.scheduler_type == "Plateau":
                self.scheduler.step(error_test.item())
            else:
                self.scheduler.step()
            if (epoch + 1) % epoch_show == 0:
                print(
                    f"Epoch:{epoch + 1} Time:{time.time() - self.t_start:.4f}, loss:{loss_train_sum.item() / len(train_loader):.4f}, loss_pde:{loss_pde_sum.item() / len(train_loader):.4f}, loss_data:{loss_data_sum.item() / len(train_loader):.4f}"
                )
                print(f"                l2_test:{error_test.item():.4f}, lr:{self.optimizer.get_lr()}")
        self.saveModel(kwrds["save_path"], name="model_dgno_final", model_dict=self.model_dict)
        self.saveLoss(kwrds["save_path"], name="loss_dgno")
        print(f"The total training time is {time.time() - self.t_start:.4f}")

    def train_finetune(
        self,
        LossClass: Module.LossClass,
        model_dict: dict,
        a: paddle.to_tensor,
        epochs: int = 1,
        lr=1.0,
        step_size=250,
        gamma=1 / 2,
        epoch_show: int = 100,
        **kwrds,
    ):
        """Fine Tunning"""
        self.model_dict = model_dict
        print("****************** The Fine-tune process ***********")
        param_list = []
        for name, param in model_dict["u"].named_parameters():
            if name.split(".")[0] == "net_a":
                param_list.append(param)
            elif name.split(".")[0] == "trunk":
                param_list.append(param)
        print("The length of param_list:", len(param_list))
        optimizer = paddle.optimizer.Adam(parameters=param_list, learning_rate=lr, weight_decay=0.0001)
        tmp_lr = paddle.optimizer.lr.StepDecay(
            step_size=step_size, gamma=gamma, last_epoch=-1, learning_rate=optimizer.get_lr()
        )
        optimizer.set_lr_scheduler(tmp_lr)
        scheduler = tmp_lr
        t_start = time.time()
        for epoch in trange(epochs):
            lossClass = LossClass(self)
            loss_train = lossClass.Loss_pde(a.to(self.device), w_pde=1.0)
            optimizer.clear_gradients(set_to_zero=False)
            loss_train.backward()
            optimizer.step()
            scheduler.step(loss_train.item())
            if (epoch + 1) % epoch_show == 0:
                print(f"The epoch:{epoch + 1}, time:{time.time() - t_start:.4f}, loss_pde:{loss_train.item()}")
        self.saveModel(kwrds["save_path"], name="model_dgno_finetune", model_dict=self.model_dict)

        print(f"The total time is {time.time() - t_start:.4f}")
