#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#

import json
import os
import re
import sys
from typing import List
from typing import Tuple

import hydra
import meshio
import numpy as np
import open3d as o3d
import paddle
import pandas as pd
from omegaconf import DictConfig
from stl import mesh


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
        self.inward_surface_normal = (
            -1 * self.cell_area_ijk / self.cell_area[:, np.newaxis]
        )
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

        paddle.save(
            obj=self.info,
            path=f"{self.save_path}/info_{str(self.index).zfill(4)}.pdparams",
        )
        print(
            f"info has been saved to : {os.path.join(self.save_path, f'info_{str(self.index).zfill(4)}.pdparams')}"
        )
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
        query_points = np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1).astype(
            np.float32
        )
        return query_points

    def compute_df_from_mesh(self):

        stl_mesh = o3d.io.read_triangle_mesh(os.path.join(self.mesh_path, self.stlID))
        num_triangles = len(stl_mesh.triangles)
        print(f"Mesh num in stl: {num_triangles}")

        json_file_path = os.path.join(
            self.save_path, self.stlID[:-4] + "_mesh_num.json"
        )
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        if os.path.isfile(json_file_path):
            os.remove(json_file_path)
        with open(json_file_path, "w", encoding="utf-8") as file:
            json.dump({"surface_mesh_num": num_triangles}, file)

        o3d_mesh = o3d.t.geometry.TriangleMesh.from_legacy(stl_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(o3d_mesh)
        df = scene.compute_distance(o3d.core.Tensor(self.query_points)).numpy()

        df_dict = {
            "df": df,
        }
        np.save(f"{self.save_path}/df_{str(self.index).zfill(4)}.npy", df_dict["df"])
        print(
            f"df has been saved to : {os.path.join(self.save_path, f'df_{str(self.index).zfill(4)}.npy')}"
        )
        return None


@hydra.main(version_base=None, config_path="./configs", config_name="train")
def main(cfg: DictConfig):
    if cfg.process_mode == "infer":
        mesh_path = cfg.pre_input_path  
        save_path = (
            cfg.pre_output_path
        )  
        bounds_dir = cfg.bounds_dir

        csvIDs = [d for d in os.listdir(mesh_path) if d[-4:] == ".csv"]
        print("All csvID:", csvIDs)
        print("Chosen csvID:", csvIDs)

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


if __name__ == "__main__":
    main()
