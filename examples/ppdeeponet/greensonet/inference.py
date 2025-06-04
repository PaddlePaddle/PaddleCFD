import os
import random
import re
import time

import meshio
import numpy as np
import paddle
from model import Net_Integral
from model import tile
from options import Options
from paddle.io import Dataset
from problem import Stokes
from utils import Mesh


def set_random_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_random_seed(0)


def extract_integers_from_string(input_string_list):
    pattern = re.compile(r"\d+")
    return [int(pattern.findall(s)[0]) for s in input_string_list]


class VTK_Dataset(Dataset):
    def __init__(self, args, mesh, problem, train_samples=None):
        self.args = args
        self.problem = problem
        self.file_list = os.listdir(problem.data_folder)
        self.case_index_list = extract_integers_from_string(self.file_list)
        if train_samples is not None:
            self.case_index_list.sort()
            self.case_index_list = self.case_index_list[:train_samples]
        print("Test/Train *.vtu file index = ", sorted(self.case_index_list))
        self.mesh = mesh
        self.z_blocks = self.mesh.z_blocks
        self.x_in_wts = paddle.to_tensor(self.mesh.X_interior["wts"], dtype="float32")
        self.x_in_coord = paddle.to_tensor(self.mesh.X_interior["coord"], dtype="float32")
        self.x_bc_wts = paddle.to_tensor(self.mesh.X_boundary["wts"], dtype="float32")
        self.x_bc_coord = paddle.to_tensor(self.mesh.X_boundary["coord"], dtype="float32")
        self.x_bc_normal = self.mesh.X_boundary["normal"]
        self.z_blocks_coord = self.mesh.z_blocks[0]["coord"]
        if "boundary_type" in self.mesh.X_boundary.keys():
            self.input_boundary_type = self.mesh.X_boundary["boundary_type"]
            self.input_boundary_type = paddle.to_tensor(self.input_boundary_type).astype("float32")

    def __getitem__(self, k):
        x_in_coord_list = []
        x_in_wts_list = []
        x_bc_coord_list = []
        x_bc_wts_list = []
        f_interior_list = []
        g_boundary_list = []
        a_boundary_list = []
        N_interior = self.x_in_coord[0].shape[0]
        N_boundary = self.x_bc_coord[0].shape[0]
        case_index = self.case_index_list[k]

        if isinstance(self.problem, Stokes):
            input_boundary_type = tile(self.input_boundary_type, self.z_blocks_coord)[:, 0]
            input_boundary_list = []
            for i in range(self.args.ngs_boundary):
                input_boundary_list.append(tile(self.x_bc_coord[i], self.z_blocks_coord)[:, :3])
        else:
            input_boundary_type = case_index
            input_boundary_list = self.x_bc_coord

        for i in range(self.args.ngs_interior):
            x_in_coord_list.append(self.x_in_coord[i])
            x_in_wts_list.append(self.x_in_wts[i])
            f_interior = self.problem.f(self.x_in_coord[i], case_index)
            if isinstance(self.problem, Stokes):
                f_interior = f_interior.reshape([N_interior, -1])
            f_interior_list.append(f_interior)

        for i in range(self.args.ngs_boundary):
            x_bc_wts_list.append(self.x_bc_wts[i])
            x_bc_coord_list.append(self.x_bc_coord[i])
            g_boundary = self.problem.g(input_boundary_list[i], input_boundary_type)
            a_boundary = self.problem.a(input_boundary_list[i])
            if isinstance(self.problem, Stokes):
                g_boundary = g_boundary.reshape([N_boundary, -1])
                a_boundary = a_boundary.reshape([N_boundary, -1])
            g_boundary_list.append(g_boundary)
            a_boundary_list.append(a_boundary)
        return (
            x_in_coord_list,
            x_in_wts_list,
            x_bc_coord_list,
            x_bc_wts_list,
            f_interior_list,
            g_boundary_list,
            a_boundary_list,
            self.z_blocks,
            case_index,
            self.x_bc_normal,
        )

    def __len__(self):
        return len(self.case_index_list)


class Tester:
    def __init__(self, args):
        self.args = args
        self.problem = args.problem
        self.mesh = Mesh(
            args.mesh_path,
            args.boundary_mesh_path,
            args.domain,
            args.blocks_num,
            ngs_boundary=self.args.ngs_boundary,
            ngs_interior=self.args.ngs_interior,
        )
        self.loss_function = paddle.nn.MSELoss()
        self.net_pde = Net_Integral(
            args.layers, args.shape, self.args.ngs_boundary, self.args.ngs_interior, args.problem, eval_mode=True
        )
        # load the pre-trained model from checkpoint
        for i in range(args.shape[0]):
            for j in range(args.shape[1]):
                print("\nLoading the pre-trained model from : ", self.args.checkpoint_path[j])
                checkpoint = paddle.load(path=str(self.args.checkpoint_path[j]))
                self.net_pde.G[i][j].set_state_dict(checkpoint)
                self.net_pde.G[i][j].eval()

    def calculate(self):
        print("Mesh blocks number :", len(self.mesh.blocks))
        for K in range(len(self.mesh.blocks)):
            if self.mesh.z_blocks[K] is not None and len(self.mesh.z_blocks[K]) > 0:
                self.mesh.z_blocks[K]["coord"] = paddle.to_tensor(self.mesh.z_blocks[K]["coord"], dtype="float32")
                self.calculate_block(K)
            else:
                raise ValueError(f"Block #{K} is empty!!!")

    def calculate_block(self, k):
        loss = 0
        loss_list = []
        l2_rel_error_list = []
        test_dataset = VTK_Dataset(self.args, self.mesh, self.problem)

        # for case_index in case_index_list:
        for data in test_dataset:
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

            print("\nInfer the test cases ID: ", case_index)
            tt = time.time()

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
            loss = self.loss_function(u_pred, u_exac)

            l2_rel_error = paddle.sum((u_pred - u_exac) ** 2) / len(self.mesh.vertices)

            print(
                f"Finish block {k}, spent time: {(time.time() - tt):.2f} s, mse loss: {loss.item():.4f}, l2 loss: {l2_rel_error.item():.6f}"
            )

            if self.args.save_vtk is True:
                self.save_to_vtk(case_index, coord.numpy(), u_pred.numpy(), u_exac.numpy())

            loss_list.append(loss.item())
            l2_rel_error_list.append(l2_rel_error.item())

        print(f"\nMSE loss over [{len(test_dataset)}] test cases {(sum(loss_list) / len(loss_list)):.3f}")
        print(f"L2 loss over [{len(test_dataset)}] test cases {(sum(l2_rel_error_list) / len(l2_rel_error_list)):.6f}")
        return loss_list, l2_rel_error_list

    def save_to_vtk(self, case_index, coords, u_pred, u_exac):
        error = np.abs(u_pred.numpy() - u_exac.numpy())
        sorted_data = []
        for node in self.mesh.vertices:
            if any(np.allclose(node, coord) for coord in coords):
                index = np.where(np.all(np.isclose(coords, node), axis=1))[0][0]
                sorted_data.append([node[0], node[1], node[2], u_pred[index][0], u_exac[index][0], error[index][0]])
            else:
                res = self.problem.u_exact(node[np.newaxis, ...], case_index)
                boundary_point_value = res
                sorted_data.append(
                    [node[0], node[1], node[2], boundary_point_value[0][0], boundary_point_value[0][0], 0]
                )
        sorted_data = np.array(sorted_data)
        mesh = meshio.Mesh(
            points=self.mesh.vertices,
            cells=[("tetra", self.mesh.elements)],
            point_data={"u_pred": sorted_data[:, 3], "u_exac": sorted_data[:, 4], "u_error": sorted_data[:, 5]},
        )
        mesh.write(f"./output/vtk_in_paraview/test_case{case_index + 1}_Greenonet.vtu")


if __name__ == "__main__":
    args = Options().parse()

    # Stokes
    print("/----------------------- [Case 3: Stokes equations] [name : 3D lid-driven cavity] -----------------------/")
    args.mesh_path = "./mesh/stokes/domain.mphtxt"
    args.boundary_mesh_path = "./mesh/stokes/boundary.mphtxt"
    args.checkpoint_path = [
        "./checkpoints/z_block0_0.pdparams",
        "./checkpoints/z_block0_1.pdparams",
        "./checkpoints/z_block0_2.pdparams",
    ]
    args.velocity_component_name = "x"
    args.problem = Stokes("./data/stokes/test_data", args.velocity_component_name)

    args.save_vtk = False
    args.domain = [0, 1, 0, 1, 0, 1]
    args.blocks_num = [1, 1, 1]
    args.shape = [1, 3]
    args.ngs_boundary = 3
    args.ngs_interior = 4
    args.test_samples = 1
    args.layers = [[[6, 12, 24, 12, 1], [6, 12, 24, 12, 1], [6, 12, 24, 12, 1]]]
    tester = Tester(args)
    tester.calculate()
