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

import gmsh
import hydra
import meshio
import numpy as np
import open3d as o3d
import paddle
import pandas as pd
from omegaconf import DictConfig
from stl import mesh


class STLConvert:
    def __init__(self, geo_path, save_path, stpID, index, info):
        self.geo_path = geo_path
        self.save_path = save_path
        self.stpID = stpID
        self.info = info
        self.index = index

    def extract_values_from_arrays(self):

        # 读取STL文件
        stlID = f"{self.stpID[:-4]}.stl"
        stl_mesh = mesh.Mesh.from_file(os.path.join(self.geo_path, stlID))

        # 提取法向量数组 [每个面含1个法向量]
        normals = stl_mesh.normals  # 形状为(n,3)的数组‌
        # print("normals_n3:", normals)
        unit_normals = (
            -1 * normals / np.linalg.norm(normals, axis=1, keepdims=True)
        )  # 单位法向量‌

        # 计算网格中心点
        vertices = stl_mesh.points.reshape(-1, 3, 3)  # 将顶点数据重组为(n,3,3)结构
        centroid = vertices.mean(axis=1)  # 计算每个三角面三个顶点的几何中心‌

        # 计算网格单元面积
        v0, v1, v2 = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        vec1 = v1 - v0  # 生成第一条边向量
        vec2 = v2 - v0  # 生成第二条边向量
        cross_product = np.cross(vec1, vec2)  # 计算叉乘
        areas = np.linalg.norm(cross_product, axis=1) / 2  # 面积公式‌

        print("stl cell number:", len(centroid))
        print(
            "centroid:", centroid, "\n", "areas:", areas, "\n", "normals:", unit_normals
        )

        # 保存为numpy文件
        print(f"area, centrods, normal has been saved to : {self.save_path}")
        os.makedirs(self.save_path, exist_ok=True)
        np.save(
            f"{self.save_path}/area_{str(self.index).zfill(4)}.npy",
            areas.astype(np.float32),
        )  # 保存面积
        np.save(
            f"{self.save_path}/centroid_{str(self.index).zfill(4)}.npy",
            centroid.astype(np.float32),
        )  # 保存中心点‌
        np.save(
            f"{self.save_path}/normal_{str(self.index).zfill(4)}.npy",
            unit_normals.astype(np.float32),
        )  # 保存法向量

        print("Mesh information extracted and saved as NumPy arrays.")

    def save_info(self):

        paddle.save(
            obj=self.info,
            path=f"{self.save_path}/info_{str(self.index).zfill(4)}.pdparams",
        )
        print(
            f"info has been saved to : {f'{self.save_path}/info_{str(self.index).zfill(4)}.pdparams'}"
        )
        return None


class Compute_df_stl:
    def __init__(self, geo_path, save_path, stlID, index, bounds_dir):
        self.geo_path = geo_path
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
        # 读取starccm导出的stl文件
        stl_mesh = o3d.io.read_triangle_mesh(os.path.join(self.geo_path, self.stlID))
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
        geo_path = cfg.pre_input_path  
        save_path = (
            cfg.pre_output_path
        )  
        bounds_dir = cfg.bounds_dir

        stlIDs = [d for d in os.listdir(geo_path) if d[-4:] == ".stl"]
        print("All stlID:", stlIDs)
        print("Chosen stlID:", stlIDs)

        index = 200
        for stlID in stlIDs:
            print("StlID:", stlID)

            json_file_path = os.path.join(geo_path, stlID[:-4] + ".json")
            with open(json_file_path, "r", encoding="utf-8") as file:
                info = json.load(file)

            stl_trans = STLConvert(geo_path, save_path, stlID, index, info)

            stl_trans.extract_values_from_arrays()
            stl_trans.save_info()

            compute_df = Compute_df_stl(geo_path, save_path, stlID, index, bounds_dir)
            compute_df.compute_df_from_mesh()
            index += 1

    else:
        raise


if __name__ == "__main__":
    main()
