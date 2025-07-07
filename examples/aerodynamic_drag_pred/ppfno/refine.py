#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#

import logging
import os
import re
import shutil
import sys
from os import path as osp
from typing import List, Tuple

import gmsh
import hydra
import meshio
import numpy as np
import open3d as o3d
import paddle
import pandas as pd
from omegaconf import DictConfig
from stl import mesh


def copy_json_file(src_file_path, dest_dir_path):
    # 检查源文件是否存在
    if not os.path.isfile(src_file_path):
        logging.info(f"源文件 {src_file_path} 不存在。")
        return

    # 检查目标目录是否存在，如果不存在则创建
    if not os.path.exists(dest_dir_path):
        os.makedirs(dest_dir_path)

    # 获取文件名
    file_name = os.path.basename(src_file_path)

    # 构建目标文件路径
    dest_file_path = os.path.join(dest_dir_path, file_name)

    # 复制文件
    try:
        shutil.copy(src_file_path, dest_file_path)
        logging.info(f"文件 {file_name} 已成功复制到 {dest_dir_path}")
    except Exception as e:
        logging.info(f"复制文件时发生错误: {e}")


class STPRefine:
    def __init__(self, geo_path, save_path, stpID, index, compute_closest_point=False):
        self.geo_path = geo_path
        self.save_path = save_path
        self.stpID = stpID
        self.index = index
        self.compute_closest_point = compute_closest_point

    def refine_stl(self, boxes, maxsize, minsize):
        src_jsons_path = os.listdir(self.geo_path)
        src_jsons_path = [d for d in os.listdir(self.geo_path) if d.endswith(".json")]
        src_jsons_path = [os.path.join(self.geo_path, d) for d in src_jsons_path]
        for src_path in src_jsons_path:
            copy_json_file(src_path, self.save_path)
        # 初始化Gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # 控制台信息输出 0:不输出 1:输出
        gmsh.clear()
        gmsh.model.add("step_mesh")

        step_path = osp.join(self.geo_path, self.stpID)  # 替换为你的 STEP 文件路径
        gmsh.model.occ.importShapes(step_path)

        gmsh.model.occ.synchronize()

        logging.info("stp loaded.")

        for i in range(len(boxes)):
            # 定义加密区域（长方体参数）
            refine_box = {
                "xmin": boxes[i][0][0],
                "xmax": boxes[i][1][0],  # X方向范围
                "ymin": boxes[i][0][1],
                "ymax": boxes[i][1][1],  # Y方向范围
                "zmin": boxes[i][0][2],
                "zmax": boxes[i][1][2],  # Z方向范围
                "size_min": minsize[i],  # 加密区域最小网格尺寸
                "size_max": maxsize[i],  # 加密区域最大网格尺寸
            }
            logging.info(refine_box)

            global_size = 5.0

            gmsh.model.mesh.field.add("Box", i)
            gmsh.model.mesh.field.setNumber(i, "XMin", refine_box["xmin"])
            gmsh.model.mesh.field.setNumber(i, "XMax", refine_box["xmax"])
            gmsh.model.mesh.field.setNumber(i, "YMin", refine_box["ymin"])
            gmsh.model.mesh.field.setNumber(i, "YMax", refine_box["ymax"])
            gmsh.model.mesh.field.setNumber(i, "ZMin", refine_box["zmin"])
            gmsh.model.mesh.field.setNumber(i, "ZMax", refine_box["zmax"])
            gmsh.model.mesh.field.setNumber(i, "VIn", refine_box["size_min"])
            gmsh.model.mesh.field.setNumber(i, "VOut", refine_box["size_max"])

            gmsh.model.mesh.field.setAsBackgroundMesh(i)
            gmsh.option.setNumber("Mesh.MeshSizeMax", global_size)
            gmsh.option.setNumber("Mesh.Algorithm", 2)
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.optimize("Laplace2D")
            logging.info(f"geom at index {i} re-meshing has been completed.")

        os.makedirs(self.save_path, exist_ok=True)
        gmsh.write(osp.join(self.save_path, f"{self.stpID[:-4]}.stl"))

        logging.info(
            f"The new stl file is saved to {osp.join(self.save_path, f'{self.stpID[:-4]}.stl')}"
        )
        gmsh.finalize()


def get_refine_params(stp_path):
    excel = pd.ExcelFile(stp_path)
    sheet_names = excel.sheet_names
    table = pd.read_excel(excel, sheet_names[0])
    boxes, maxsizes, minsizes = [], [], []

    for i in range(int(len(table["顶点1/mm"]) / 3)):
        box_min = []
        box_max = []
        boxes.append([])
        for j in range(3):
            box_min.append(table["顶点1/mm"][3 * i + j])
            box_max.append(table["顶点2/mm"][3 * i + j])
        boxes[i].append(box_min)
        boxes[i].append(box_max)
        maxsizes.append(table["普遍网格尺寸/mm"][3 * i])
        minsizes.append(table["最小网格尺度/mm"][3 * i])
    boxes = [boxes[-1]]
    maxsizes = [maxsizes[-1]]
    minsizes = [minsizes[-1]]

    return boxes, maxsizes, minsizes


@hydra.main(version_base=None, config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    geo_path = cfg.refine_input_path  
    save_path = (
        cfg.refine_output_path
    )  
    os.makedirs(os.path.join(save_path, "log"), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(save_path, "log", "encrypt.txt"),
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s: %(message)s",
        force=True,
    )

    stpIDs = [d for d in os.listdir(geo_path) if d[-4:] == ".stp"]
    logging.info(f"All stpID: {stpIDs}")
    logging.info(f"Chosen stpID: {stpIDs}")
    boxes = [[[-5500.0, -300.0, -20.0], [5500.0, 300.0, 4800.0]]]
    maxsizes = [5.0]
    minsizes = [0.625]
    logging.info(boxes)
    logging.info(maxsizes)
    logging.info(minsizes)

    index = 200
    for stpID in stpIDs:
        stl_trans = STPRefine(geo_path, save_path, stpID, index)
        stl_trans.refine_stl(boxes, maxsizes, minsizes)
        index += 1


if __name__ == "__main__":
    main()
