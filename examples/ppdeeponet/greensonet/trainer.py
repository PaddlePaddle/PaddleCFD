import os
import time

import paddle
from inference import VTK_Dataset
from model import Net_Integral
from utils import Mesh


class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.cuda_index = args.cuda_index
        self.device = args.device
        self.pde_case = args.pde_case
        self.resume = args.resume
        self.problem = args.problem
        self.tol = args.tol
        self.tol_change = args.tol_change
        self.domain = args.domain
        self.blocks_num = args.blocks_num
        self.ngs_boundary = args.ngs_boundary
        self.ngs_interior = args.ngs_interior
        self.train_samples = args.train_samples
        self.test_samples = args.test_samples
        meshfile, boundaryfile = self.get_mesh_path(self.args.mesh_path, args.boundary_mesh_path)
        self.mesh = Mesh(
            meshfile,
            boundaryfile,
            self.domain,
            self.blocks_num,
            ngs_boundary=self.ngs_boundary,
            ngs_interior=self.ngs_interior,
        )
        self.criterion = paddle.nn.MSELoss()
        self.epochs_Adam = self.args.epochs_Adam
        self.lam = self.args.lam
        self.lr = self.args.lr
        self.net_pde = Net_Integral(self.args.layers, args.shape, self.ngs_boundary, self.ngs_interior, self.problem)

    def get_mesh_path(self, mesh_path, boundary_mesh_path):
        return mesh_path, boundary_mesh_path

    def _model_path(self):
        """Create directory of saved model"""
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")
        pde = os.path.join("checkpoints", f"{self.pde_case}")
        if not os.path.exists(pde):
            os.mkdir(pde)
        geo = os.path.join("checkpoints", f"{self.pde_case}", f"{self.geo_type}")
        if not os.path.exists(geo):
            os.mkdir(geo)
        model_path = os.path.join("checkpoints", f"{self.pde_case}", f"{self.geo_type}")
        return model_path

    def train(self):
        if not isinstance(self.mesh.X_boundary["normal"], paddle.Tensor):
            self.mesh.X_boundary["normal"] = (
                paddle.to_tensor(data=self.mesh.X_boundary["normal"]).astype(dtype="float32").to(self.device)
            )
            self.mesh.X_boundary["boundary_type"] = (
                paddle.to_tensor(data=self.mesh.X_boundary["boundary_type"]).astype(dtype="float32").to(self.device)
            )
            for k in range(self.ngs_interior):
                self.mesh.X_interior["coord"][k] = (
                    paddle.to_tensor(data=self.mesh.X_interior["coord"][k]).astype(dtype="float32").to(self.device)
                )
                self.mesh.X_interior["wts"][k] = (
                    paddle.to_tensor(data=self.mesh.X_interior["wts"][k]).astype(dtype="float32").to(self.device)
                )
            for k in range(self.ngs_boundary):
                out_34 = (
                    paddle.to_tensor(data=self.mesh.X_boundary["coord"][k]).astype(dtype="float32").to(self.device)
                )
                out_34.stop_gradient = not True
                self.mesh.X_boundary["coord"][k] = out_34
                out_35 = paddle.to_tensor(data=self.mesh.X_boundary["wts"][k]).astype(dtype="float32").to(self.device)
                out_35.stop_gradient = not True
                self.mesh.X_boundary["wts"][k] = out_35
        for K in range(len(self.mesh.blocks)):
            if self.mesh.z_blocks[K] is not None and len(self.mesh.z_blocks[K]) > 0:
                if not isinstance(self.mesh.z_blocks[K]["coord"], paddle.Tensor):
                    self.mesh.z_blocks[K]["coord"] = (
                        paddle.to_tensor(data=self.mesh.z_blocks[K]["coord"]).astype(dtype="float32").to(self.device)
                    )
                self.train_block(K)
            else:
                print(f"Block #{K} is empty!!!")

    def train_block(self, k):
        if self.resume:
            resume_path = os.path.join("checkpoints", f"block{k}_0.pdparams")
            if os.path.isfile(resume_path):
                print(f"Resuming training, loading {resume_path} ...")
                for i in range(self.args.shape[0]):
                    for j in range(self.args.shape[1]):
                        checkpoint = paddle.load(path=str(self.model_path[j]))
                        self.net_pde.G[i][j].set_state_dict(checkpoint)
            else:
                raise
        for i in range(self.args.shape[0]):
            for j in range(self.args.shape[1]):
                self.net_pde.G[i][j].train()
                self.net_pde.G[i][j].to(self.device)
                self.net_pde.G[i][j].clear_gradients(set_to_zero=False)
        params = [
            param
            for i in range(self.args.shape[0])
            for j in range(self.args.shape[1])
            for param in self.net_pde.G[i][j].parameters()
            if not param.stop_gradient
        ]
        self.optimizer_Adam = paddle.optimizer.Adam(parameters=params, learning_rate=self.lr, weight_decay=0.0)
        tmp_lr = paddle.optimizer.lr.StepDecay(step_size=100, gamma=0.9, learning_rate=self.optimizer_Adam.get_lr())
        self.optimizer_Adam.set_lr_scheduler(tmp_lr)
        self.lr_scheduler = tmp_lr
        best_loss = 10000000000.0
        print(f"Start Trainning (blocks #{k})")
        tt = time.time()
        output_file = open(os.path.join(f"output_{k}.txt"), "w+")
        train_dataset = VTK_Dataset(self.args, self.mesh, self.problem, self.train_samples)

        for epoch in range(self.epochs_Adam):
            train_loss = 0
            total_loss = 0
            for data in train_dataset:
                (
                    x_in_coord_list,
                    x_in_wts_list,
                    x_bc_coord_list,
                    x_bc_wts_list,
                    f_interior_list,
                    g_boundary_list,
                    a_boundary_list,
                    z_blocks,
                    case_index,
                    x_bc_normal,
                ) = data
                coord = z_blocks[k]["coord"]
                # print("\nTrainning: the test cases ID: ", case_index)
                self.optimizer_Adam.clear_gradients(set_to_zero=False)
                u_pred = self.net_pde(
                    x_in_coord_list,
                    x_in_wts_list,
                    x_bc_wts_list,
                    x_bc_coord_list,
                    coord,
                    f_interior_list,
                    g_boundary_list,
                    a_boundary_list,
                    x_bc_normal,
                )
                u_exac = self.problem.u_exact(coord.numpy(), case_index)
                u_exac = paddle.to_tensor(u_exac, dtype="float32")
                loss = self.criterion(u_pred, u_exac)
                mean_loss = paddle.sum(x=(u_pred - u_exac) ** 2) / len(self.mesh.vertices)
                loss.backward()
                self.optimizer_Adam.step()
                train_loss += loss.item()
                total_loss += mean_loss
            self.lr_scheduler.step()
            train_loss = train_loss / self.train_samples
            total_loss = total_loss.item() / self.train_samples
            t2 = time.time()
            infos = (
                f"Epoch: {epoch:5d}/{self.epochs_Adam:5d} "
                + f"time: {t2 - tt:.2f} "
                + f"lr: {self.lr_scheduler.get_lr():.2e} "
                + f"loss: {train_loss:.5f}  "
                + f"total loss: {total_loss:.5f}  "
            )
            print(infos)
            output_file.write(
                f"""epoch:{epoch:5d}, time: {t2 - tt:.2f}, loss: {float(train_loss):.3f}, total loss: {float(total_loss):.3f}"""
            )
            tt = time.time()
            if (epoch + 1) % 5 == 0:
                is_best = train_loss < best_loss
                if is_best:
                    best_loss = train_loss
                    for i in range(self.args.shape[0]):
                        for j in range(self.args.shape[1]):
                            checkpoint_path = os.path.join(
                                "checkpoints", f"{self.pde_case}_net_pde.G[{i}][{j}].pdparams"
                            )
                            checkpoint = paddle.save(self.net_pde.G[i][j].state_dict(), path=checkpoint_path)
            if train_loss < self.tol:
                print(f"train_loss after Adam is {train_loss:.4e} ")
                is_best = train_loss < best_loss
                for i in range(self.args.shape[0]):
                    for j in range(self.args.shape[1]):
                        checkpoint_path = os.path.join("checkpoints", f"{self.pde_case}_net_pde.G[{i}][{j}].pdparams")
                        checkpoint = paddle.save(self.net_pde.G[i][j].state_dict(), path=checkpoint_path)
                break
        output_file.close()
