#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
"""
Authors: chenkai26(chenkai26@baidu.com)
Date:    2025/5/20
"""

import json
import os

import hydra
import numpy as np
import open3d as o3d
import paddle
import pandas as pd
from omegaconf import DictConfig


class CSVConvert:

    def __init__(self, mesh_path, save_path, csvID, index, info):
        self.csv_data = pd.read_csv(os.path.join(mesh_path, csvID)).to_numpy()
        self.centroid = self.csv_data[:, -3:]
        self.cell_area_ijk = self.csv_data[:, :3]

        self.inward_surface_normal = None
        self.cell_area = None

        self.mesh_path = mesh_path
        self.save_path = save_path
        self.csvID = csvID
        self.info = info
        self.index = index

    @property
    def area(self):
        try:
            self.cell_area = np.sqrt(np.sum(self.cell_area_ijk**2, axis=1))
        except TypeError:
            print(f"{self.csvID} skipped.")
        return self.cell_area

    @property
    def normal(self):
        self.inward_surface_normal = -1 * self.cell_area_ijk / self.cell_area[:, np.newaxis]
        return self.inward_surface_normal

    def save_volume_mesh(self):
        print("csv cell number:", len(self.centroid))

        # 保存为numpy文件
        print(f"area, centroids, normal are saving to : {self.save_path}")
        os.makedirs(self.save_path, exist_ok=True)
        np.save(
            f"{self.save_path}/area_{str(self.index).zfill(4)}.npy",
            self.area.astype(np.float32),
        )  # 保存面积
        np.save(
            f"{self.save_path}/centroid_{str(self.index).zfill(4)}.npy",
            self.centroid.astype(np.float32),
        )  # 保存中心点‌
        np.save(
            f"{self.save_path}/normal_{str(self.index).zfill(4)}.npy",
            self.inward_surface_normal.astype(np.float32),
        )  # 保存法向量

        print("Volume mesh information has been saved as NumPy arrays.")
        return None

    def save_info(self):
        # info_dict = {
        #     "length": 0,
        #     "width": 0,
        #     "height": 0,
        #     "clearance": 0,
        #     "slant": 0,
        #     "radius": 0,
        #     "velocity": self.info["velocity"],
        #     "re": 0,
        #     "reference_area": self.info["reference_area"],
        #     "density": self.info["density"],
        #     "compute_normal": False,
        # }
        paddle.save(
            obj=self.info,
            path=f"{self.save_path}/info_{str(self.index).zfill(4)}.pdparams",
        )
        print(f"info has been saved to : {os.path.join(self.save_path, f'info_{str(self.index).zfill(4)}.pdparams')}")
        return None


class Compute_df_stl:
    def __init__(self, mesh_path, save_path, stlID, index, bounds_dir):
        self.mesh_path = mesh_path
        self.save_path = save_path
        self.stlID = stlID
        self.bounds_dir = bounds_dir
        self.query_points = self.compute_query_points()
        self.index = index

    def compute_query_points(self, eps=1e-6):
        with open(os.path.join(self.bounds_dir, "global_bounds.txt"), "r") as fp:
            min_bounds = fp.readline().split(" ")
            max_bounds = fp.readline().split(" ")
            min_bounds = [(float(a) - eps) for a in min_bounds]
            max_bounds = [(float(a) + eps) for a in max_bounds]
        sdf_spatial_resolution = [64, 64, 64]
        tx = np.linspace(min_bounds[0], max_bounds[0], sdf_spatial_resolution[0])
        ty = np.linspace(min_bounds[1], max_bounds[1], sdf_spatial_resolution[1])
        tz = np.linspace(min_bounds[2], max_bounds[2], sdf_spatial_resolution[2])
        query_points = np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1).astype(np.float32)
        return query_points

    def compute_df_from_mesh(self):
        # 读取CATIA导出的二进制STL文件
        # stl_mesh = mesh.Mesh.from_file(os.path.join(self.mesh_path, self.stlID))
        # print('stl mesh loaded.')
        # vertices = stl_mesh.vectors.reshape(-1, 3) * 1e-3
        # # print("vertices:", vertices)
        # faces = np.arange(vertices.shape[0]).reshape(-1, 3)

        # # 构建Open3D网格
        # o3d_mesh = o3d.geometry.TriangleMesh()
        # o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        # o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

        # 读取starccm导出的stl文件
        stl_mesh = o3d.io.read_triangle_mesh(os.path.join(self.mesh_path, self.stlID))
        num_triangles = len(stl_mesh.triangles)
        print(f"Mesh num in stl: {num_triangles}")
        o3d_mesh = o3d.t.geometry.TriangleMesh.from_legacy(stl_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(o3d_mesh)
        df = scene.compute_distance(o3d.core.Tensor(self.query_points)).numpy()
        # closest_point = scene.compute_closest_points(
        #     o3d.core.Tensor(self.query_points)
        # )["points"].numpy()
        df_dict = {
            "df": df,
        }
        np.save(f"{self.save_path}/df_{str(self.index).zfill(4)}.npy", df_dict["df"])
        print(f"df has been saved to : {os.path.join(self.save_path, f'df_{str(self.index).zfill(4)}.npy')}")
        return None


@hydra.main(version_base=None, config_path="./configs", config_name="train")
def main(cfg: DictConfig):
    if cfg.process_mode == "infer":
        mesh_path = cfg.pre_input_path  # '/home/chenkai26/Paddle-AeroSimOpt/data/'
        save_path = cfg.pre_output_path  # '/home/chenkai26/Paddle-AeroSimOpt/data/extracted_info/'
        bounds_dir = cfg.bounds_dir

        # rextract & save elements from csv & stl

        csvIDs = [d for d in os.listdir(mesh_path) if d[-4:] == ".csv"]
        print("All csvID:", csvIDs)
        # stpIDs = stpIDs[2:]
        print("Chosen csvID:", csvIDs)

        # info = {"velocity": 65.0, "reference_area": 0.176, "density": 1.05}

        index = 200
        for csvID in csvIDs:
            print("csvID:", csvID)

            json_file_path = os.path.join(mesh_path, csvID[:-4] + ".json")
            with open(json_file_path, "r", encoding="utf-8") as file:
                info = json.load(file)

            csv_trans = CSVConvert(mesh_path, save_path, csvID, index, info)
            area = csv_trans.area
            normal = csv_trans.normal
            csv_trans.save_volume_mesh()
            csv_trans.save_info()

            stlID = csvID[:-4] + ".stl"
            compute_df = Compute_df_stl(mesh_path, save_path, stlID, index, bounds_dir)
            compute_df.compute_df_from_mesh()
            index += 1

    else:
        raise


"""
python inferenceDataPreprocess.py -cn train.yaml process_mode=infer \
    pre_input_path=/home/chenkai26/Paddle-AeroSimOpt/refine_data/2014-f-6103 \
    pre_output_path=/home/chenkai26/Paddle-AeroSimOpt/pre_process/2014-f-6103
"""
if __name__ == "__main__":
    main()
