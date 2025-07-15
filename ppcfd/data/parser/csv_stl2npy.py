import sys
import os
import paddle
"""
Authors: chenkai26(chenkai26@baidu.com)
Date:    2024/8/27
"""
import re
import numpy as np
import pandas as pd
import meshio
import pymeshlab
from stl import mesh
from typing import List
from typing import Tuple
import open3d as o3d


class CFDDataTransiton:

    def __init__(self, case_path, save_path, caseID):
        self.inward_surface_normal = None
        self.cell_area = None
        self.csv_data = pd.read_csv(os.path.join(case_path, f'{caseID}.csv')
            ).to_numpy()
        self.centroid = self.csv_data[:, -3:]
        self.press = self.csv_data[:, 0]
        self.wallshearstress = self.csv_data[:, 1:4] * -1
        self.cell_area_ijk = self.csv_data[:, 4:7]
        self.flow_direction = np.array([-1, 0, 0])
        self.lift_direction = np.array([0, 0, 1])
        self.flow_speed = 65.0
        self.reference_area = 0.1866
        self.mass_density = 1.225
        self.const = 2.0 / (self.mass_density * self.flow_speed ** 2 * self
            .reference_area)
        self.save_path = save_path
        self.case_path = case_path
        self.caseID = caseID

    @property
    def area(self):
        self.cell_area = np.sqrt(np.sum(self.cell_area_ijk ** 2, axis=1))
        return self.cell_area

    @property
    def normal(self):
        self.inward_surface_normal = -1 * self.cell_area_ijk / self.cell_area[
            :, np.newaxis]
        return self.inward_surface_normal

    def drag_c(self):

        def pressure_drag_c():
            cell_fp = self.press * self.cell_area * np.sum(self.
                inward_surface_normal * self.flow_direction, axis=1)
            cd_p = np.sum(cell_fp, axis=0) * self.const
            return cd_p

        def friction_drag_c():
            cell_ff = self.cell_area * np.sum(self.wallshearstress * self.
                flow_direction, axis=1)
            cd_f = np.sum(cell_ff, axis=0) * self.const
            return cd_f
        cd_p = pressure_drag_c()
        cd_f = friction_drag_c()
        return cd_p, cd_f

    def lift_c(self):

        def pressure_lift_c():
            cell_fp = self.press * self.cell_area * np.sum(self.
                inward_surface_normal * self.lift_direction, axis=1)
            cl_p = np.sum(cell_fp, axis=0) * self.const
            return cl_p

        def friction_lift_c():
            cell_ff = self.cell_area * np.sum(self.wallshearstress * self.
                lift_direction, axis=1)
            cl_f = np.sum(cell_ff, axis=0) * self.const
            return cl_f
        cl_p = pressure_lift_c()
        cl_f = friction_lift_c()
        return cl_p, cl_f

    def generate_visual_vtk(self):
        cells = [('vertex', np.arange(tuple(self.centroid.shape)[0]).
            reshape(-1, 1))]
        mesh = meshio.Mesh(points=self.centroid, cells=cells)
        mesh.point_data.update({'pressure': self.press})
        mesh.point_data.update({'wallshearstress': self.wallshearstress})
        mesh.point_data.update({'area': self.area})
        mesh.point_data.update({'normal': self.normal})
        meshio.write(os.path.join(self.case_path,
            f'mesh_{self.caseID[4:].zfill(4)}.vtk'), mesh)
        return None

    def save_values(self):
        np.save(os.path.join(self.save_path,
            f'pressure_{self.caseID[4:].zfill(4)}.npy'), self.press)
        np.save(os.path.join(self.save_path,
            f'wallshearstress_{self.caseID[4:].zfill(4)}.npy'), self.
            wallshearstress)
        np.save(os.path.join(self.save_path,
            f'centroid_{self.caseID[4:].zfill(4)}.npy'), self.centroid)
        np.save(os.path.join(self.save_path,
            f'area_{self.caseID[4:].zfill(4)}.npy'), self.area)
        np.save(os.path.join(self.save_path,
            f'normal_{self.caseID[4:].zfill(4)}.npy'), self.normal)
        info_dict = {'length': 0, 'width': 0, 'height': 0, 'clearance': 0,
            'slant': 0, 'radius': 0, 'velocity': 65.0, 're': 0,
            'reference_area': 0.1866, 'density': 1.225, 'compute_normal': False
            }
        paddle.save(obj=info_dict, path=os.path.join(self.save_path,
            f'info_{self.caseID[4:].zfill(4)}.pdparams'))
        return None


class MeshTransition:

    def __init__(self, stl_path, save_path, caseID, compute_closest_point=
        False, geo='mesh'):
        self.stl_path = stl_path
        self.save_path = save_path
        self.caseID = caseID
        self.query_points = self.compute_query_points()
        self.compute_closest_point = compute_closest_point
        self.geo = geo

    def combine_stl(self):
        stl_files = [f for f in os.listdir(self.stl_path + '/surface mesh') if
            os.path.isfile(os.path.join(self.stl_path + '/surface mesh', f))]
        combined_meshes = None
        for stl_file in stl_files:
            current_mesh = mesh.Mesh.from_file(self.stl_path +
                '/surface mesh/' + stl_file)
            if combined_meshes is None:
                combined_meshes = current_mesh.data.copy()
            else:
                combined_meshes = np.concatenate([combined_meshes,
                    current_mesh.data])
        combined_mesh = mesh.Mesh(combined_meshes)
        combined_mesh.save(self.stl_path + f'/{self.caseID}.stl')

    def refine_stl(self):
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(self.stl_path + f'/{self.caseID}.stl')
        mesh.apply_filter('meshing_isotropic_explicit_remeshing',
            iterations=5, targetlen=pymeshlab.PercentageValue(0.03))
        print('Refined TriangleMesh with %s points and %s triangles.' % (
            mesh.current_mesh().vertex_number(), mesh.current_mesh().
            face_number()))
        mesh.save_current_mesh(self.stl_path + f'/{self.caseID}_refined.stl')

    def save_mesh(self, combine_stl=False, doRefine=False):
        if combine_stl:
            self.combine_stl()
        if doRefine:
            self.refine_stl()
        stl_mesh = o3d.io.read_triangle_mesh(self.stl_path +
            f'/{self.caseID}.stl')
        print(stl_mesh)
        stl_mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(self.save_path +
            f'/mesh_{self.caseID[4:].zfill(4)}.ply', stl_mesh)

    def load_bound(self, filename='watertight_global_bounds.txt', eps=1e-06
        ) ->Tuple[List[float], List[float]]:
        with open(os.path.join(self.save_path, filename), 'r') as fp:
            min_bounds = fp.readline().split(' ')
            max_bounds = fp.readline().split(' ')
            min_bounds = [(float(a) - eps) for a in min_bounds]
            max_bounds = [(float(a) + eps) for a in max_bounds]
        return min_bounds, max_bounds

    def compute_query_points(self):
        min_bounds, max_bounds = self.load_bound(filename=
            'global_bounds.txt', eps=0.01)
        sdf_spatial_resolution = [64, 64, 64]
        tx = np.linspace(min_bounds[0], max_bounds[0],
            sdf_spatial_resolution[0])
        ty = np.linspace(min_bounds[1], max_bounds[1],
            sdf_spatial_resolution[1])
        tz = np.linspace(min_bounds[2], max_bounds[2],
            sdf_spatial_resolution[2])
        query_points = np.stack(np.meshgrid(tx, ty, tz, indexing='ij'), axis=-1
            ).astype(np.float32)
        return query_points

    def compute_df_closest_point_from_mesh(self):
        stl_mesh = o3d.io.read_triangle_mesh(self.save_path +
            f'/mesh_{self.caseID[4:].zfill(4)}.ply')
        stl_mesh = o3d.t.geometry.TriangleMesh.from_legacy(stl_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(stl_mesh)
        df = scene.compute_distance(o3d.core.Tensor(self.query_points)).numpy()
        if self.compute_closest_point:
            closest_point = scene.compute_closest_points(o3d.core.Tensor(
                self.query_points))['points'].numpy()
        else:
            closest_point = None
        df_closest_point_dict = {'df': df, 'closest': closest_point}
        return df_closest_point_dict

    def compute_df_closest_point_from_pcd(self):
        query_points = self.query_points.reshape(-1, 3)
        query_points = o3d.utility.Vector3dVector(query_points)
        pcd_query_points = o3d.geometry.PointCloud()
        pcd_query_points.points = query_points
        train_point = np.load(os.path.join(self.stl_path,
            f'centroid_{self.caseID[4:].zfill(4)}.npy'))
        train_point = o3d.utility.Vector3dVector(train_point)
        pcd_train = o3d.geometry.PointCloud()
        pcd_train.points = train_point
        df = pcd_query_points.compute_point_cloud_distance(pcd_train)
        df = np.asarray(df).reshape(64, 64, 64)
        closest_point = None
        df_closest_point_dict = {'df': df, 'closest': closest_point}
        return df_closest_point_dict

    def save_df_closest_point(self):
        if self.geo == 'mesh':
            df_closest_point_dict = self.compute_df_closest_point_from_mesh()
        elif self.geo == 'pcd':
            df_closest_point_dict = self.compute_df_closest_point_from_pcd()
        else:
            raise 'Not supported geometry source. Only Mesh or PCD supported.'
        np.save(os.path.join(self.save_path,
            f'df_{self.caseID[4:].zfill(4)}.npy'), df_closest_point_dict['df'])
        if df_closest_point_dict['closest'] is not None:
            np.save(os.path.join(self.save_path,
                f'closest_{self.caseID[4:].zfill(4)}.npy'),
                df_closest_point_dict['closest'])


def print_floats(*args):
    format_str = '{:.' + str(4) + 'f}'
    for num in args:
        print(format_str.format(num), end='\t')
    print()


def compute_save_bounds_all(path, save_path):
    print('Computing bounds...')
    caseIDs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(
        path, d))]
    area_bounds_all = []
    global_bounds_all = []
    p_all = []
    wss_all = []
    for caseID in caseIDs:
        print(caseID)
        case_path = os.path.join(path, caseID)
        data_trans = CFDDataTransiton(case_path, save_path, caseID)
        area = data_trans.area
        area_bounds_all.append([np.expand_dims(np.min(data_trans.cell_area),
            axis=0), np.expand_dims(np.max(data_trans.cell_area), axis=0)])
        global_bounds_all.append([np.expand_dims(np.min(data_trans.centroid,
            axis=0), axis=0), np.expand_dims(np.max(data_trans.centroid,
            axis=0), axis=0)])
        p_all.append(data_trans.press)
        wss_all.append(data_trans.wallshearstress)
    area_bounds_all = [np.concatenate(column) for column in zip(*
        area_bounds_all)]
    global_bounds_all = [np.concatenate(column, axis=0) for column in zip(*
        global_bounds_all)]
    p_all = np.concatenate(p_all, axis=0)
    wss_all = np.concatenate(wss_all, axis=0)
    area_bounds = [np.min(area_bounds_all[0]), np.max(area_bounds_all[1])]
    global_bounds = [np.min(global_bounds_all[0], axis=0), np.max(
        global_bounds_all[1], axis=0)]
    train_pressure_mean_std = [np.mean(p_all), np.std(p_all)]
    train_wallshearstress_mean_std = [np.mean(wss_all, axis=0), np.std(
        wss_all, axis=0)]
    bounds_dict = {'area_bounds': area_bounds, 'global_bounds':
        global_bounds, 'train_pressure_mean_std': train_pressure_mean_std,
        'train_wallshearstress_mean_std': train_wallshearstress_mean_std}
    print('bounds_dict:', bounds_dict)
    for k, v in bounds_dict.items():
        with open(save_path + '/' + k + '.txt', 'w') as f:
            if k == 'global_bounds' or k == 'train_wallshearstress_mean_std':
                for i in range(len(v)):
                    f.write(' '.join(str(number) for number in v[i].tolist(
                        )) + '\n')
            else:
                for i in range(len(v)):
                    f.write('%s\n' % v[i].tolist())
    print('Bounds computed.')
    return None


def auto_trans(case_path, stl_path, save_path, caseID):
    data_trans = CFDDataTransiton(case_path, save_path, caseID)
    press = data_trans.press
    wallshearstress = data_trans.wallshearstress
    centroid = data_trans.centroid
    area = data_trans.area
    normal = data_trans.normal
    cd_p, cd_f = data_trans.drag_c()
    cl_p, cl_f = data_trans.lift_c()
    np.set_printoptions(precision=4)
    print(len(normal))
    print('cd_p', '\t', 'cd_f', '\t', 'cd')
    print_floats(cd_p, cd_f, cd_p + cd_f)
    print('cl_p', '\t', 'cl_f', '\t', 'cl')
    print_floats(cl_p, cl_f, cl_p + cl_f)
    data_trans.generate_visual_vtk()
    data_trans.save_values()
    mesh_trans = MeshTransition(stl_path, save_path, caseID,
        compute_closest_point=False)
    mesh_trans.save_mesh()
    mesh_trans.save_df_closest_point()


def extract_number(s):
    s = s[4:]
    ids = re.findall('\\d+', s)
    return int(ids[0])


if __name__ == '__main__':
    path = 'D:/Dataset/zhongche-SimExpData/simExpCases'
    save_path = 'D:/Dataset/zhongche-SimExpData/transformedSimExpDataPaddle'
    stl_path = 'D:/Dataset/zhongche-SimExpData/stlFiles'
    caseIDs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(
        path, d))]
    caseIDs.sort(key=extract_number)
    # caseIDs = caseIDs[-1:]
    for caseID in caseIDs:
        print(caseID)
        case_path = os.path.join(path, caseID)
        auto_trans(case_path, stl_path, save_path, caseID)
