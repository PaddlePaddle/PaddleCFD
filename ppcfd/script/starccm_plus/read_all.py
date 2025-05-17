import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import paddle
import vtk
from ppcfd.script.starccm_plus.pyfrontal import calculate_frontal_area
from ppcfd.script.starccm_plus.read_case import read_case
from vtk.util.numpy_support import numpy_to_vtk
from vtk.util.numpy_support import vtk_to_numpy

sys.path.append("./PaddleScience/")
sys.path.append("/home/aistudio/3rd_lib")


def normalization(locations, min_bounds, max_bounds):
    locations = (locations - min_bounds) / (max_bounds - min_bounds)
    locations = 2 * locations - 1
    return locations


def load_bound(data_dir, filename):
    with open(data_dir / filename, "r") as fp:
        min_bounds = fp.readline().split(" ")
        max_bounds = fp.readline().split(" ")
        min_bounds = [float(a) - 1e-6 for a in min_bounds]
        max_bounds = [float(a) + 1e-6 for a in max_bounds]
    return min_bounds, max_bounds


def normalizer(data_dir):
    p_norm = UnitGaussianNormalizer(
        paddle.zeros(5), eps=1e-06, reduce_dim=0, verbose=False
    )
    mean, std = load_bound(data_dir, filename="train_pressure_mean_std.txt")
    p_norm.mean, p_norm.std = paddle.to_tensor(mean), paddle.to_tensor(std)
    wss_norm = UnitGaussianNormalizer(
        paddle.zeros(5), eps=1e-06, reduce_dim=0, verbose=False
    )
    mean, std = load_bound(data_dir, filename="train_wss_mean_std.txt")
    wss_norm.mean, wss_norm.std = paddle.to_tensor(mean), paddle.to_tensor(std)
    return [p_norm.decode, wss_norm.decode], [p_norm.encode, wss_norm.encode]


def location_normalization(locations, min_bounds, max_bounds):
    min_bounds = paddle.to_tensor(min_bounds)
    max_bounds = paddle.to_tensor(max_bounds)
    locations = (locations - min_bounds) / (max_bounds - min_bounds)
    locations = 2 * locations - 1
    return locations


def read_ply(file_path):
    reader = vtk.vtkPLYReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    return reader, polydata


def read_stl(file_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    return reader, polydata


def read_obj(file_path):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    return reader, polydata


def read_vtk(file_path):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    point_data_keys = [
        polydata.GetPointData().GetArrayName(i)
        for i in range(polydata.GetPointData().GetNumberOfArrays())
    ]
    cell_data_keys = [
        polydata.GetCellData().GetArrayName(i)
        for i in range(polydata.GetCellData().GetNumberOfArrays())
    ]
    print("Point Data Keys:", point_data_keys)
    print("Cell Data Keys:", cell_data_keys)
    return reader, polydata


def get_normals(polydata):
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(polydata)
    normals_filter.ComputeCellNormalsOn()
    normals_filter.ConsistencyOn()
    normals_filter.FlipNormalsOn()
    normals_filter.AutoOrientNormalsOn()
    normals_filter.Update()
    numpy_cell_normals = vtk_to_numpy(
        normals_filter.GetOutput().GetCellData().GetNormals()
    ).astype(np.float32)
    return numpy_cell_normals


def get_areas(polydata):
    cell_size_filter = vtk.vtkCellSizeFilter()
    cell_size_filter.SetInputData(polydata)
    cell_size_filter.ComputeAreaOn()
    cell_size_filter.Update()
    numpy_cell_areas = vtk_to_numpy(
        cell_size_filter.GetOutput().GetCellData().GetArray("Area")
    ).astype(np.float32)
    return numpy_cell_areas


def get_centroids(polydata):
    cell_centers = vtk.vtkCellCenters()
    cell_centers.SetInputData(polydata)
    cell_centers.Update()
    numpy_cell_centers = vtk_to_numpy(
        cell_centers.GetOutput().GetPoints().GetData()
    ).astype(np.float32)
    return numpy_cell_centers


def get_nodes(polydata):
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float32)
    return points


def velocity(polydata):
    vel = vtk_to_numpy(polydata.GetPointData().GetArray("point_vectors")).astype(
        np.float32
    )
    return vel


def load_sdf_queries(min_bounds, max_bounds):
    tx = np.linspace(min_bounds[0], max_bounds[0], 64)
    ty = np.linspace(min_bounds[1], max_bounds[1], 64)
    tz = np.linspace(min_bounds[2], max_bounds[2], 64)
    sdf_query_points = np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1).astype(
        np.float32
    )
    return sdf_query_points


def load_sdf(file_path, sdf_query_points):
    mesh = open3d.io.read_triangle_mesh(str(file_path))
    mesh = open3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = open3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    sdf = scene.compute_distance(sdf_query_points).numpy()
    sdf = sdf[np.newaxis, :]
    return sdf


def load_sdf_from_vtk(polydata_triangle, sdf_query_points):
    points = vtk_to_numpy(polydata_triangle.GetPoints().GetData()).astype(np.float32)
    triangles = vtk_to_numpy(polydata_triangle.GetPolys().GetData()).astype(np.float32)
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(points)
    mesh.triangles = open3d.utility.Vector3iVector(triangles.reshape(-1, 4)[:, 1:])
    mesh = open3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = open3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    sdf = scene.compute_distance(sdf_query_points).numpy()
    sdf = sdf
    return sdf


def read(file_path):
    def decode_fn(data, i):
        return decode[i](data)

    if file_path.suffix == ".ply":
        _, polydata = read_ply(file_path)
        decode, encode = normalizer(file_path.parent / "txt/")

        min_bounds, max_bounds = load_bound(
            file_path.parent / "txt/", filename="global_bounds.txt"
        )
        sdf_query_points = load_sdf_queries(min_bounds, max_bounds)
        sdf = load_sdf(file_path, sdf_query_points)
        sdf_query_points = (
            location_normalization(sdf_query_points, min_bounds, max_bounds)
            .unsqueeze(axis=0)
            .transpose([0, 4, 1, 2, 3])
        )

        normal = get_normals(polydata)
        areas = get_areas(polydata)
        flow_normals = paddle.zeros_like(x=normal)
        flow_normals[:, 0] = -1

        direction = paddle.sum(x=normal * flow_normals, axis=1, keepdim=False)
        centroids = location_normalization(
            get_centroids(polydata), min_bounds, max_bounds
        )
        centroids = paddle.to_tensor(centroids)

        data_dict = {
            "centroids": [centroids],
            "normal": [normal],
            "areas": [areas],
            "sdf_query_points": paddle.to_tensor(sdf_query_points),
            "df": paddle.to_tensor(sdf),
            "info": [{"velocity": np.float32(33.33)}],
            "polydata": polydata,
            "c_p": (direction * areas).reshape((-1, 1)),
            "c_wss": (-const * areas).reshape((-1, 1)),
            "decode_fn": decode_fn,
        }
    elif file_path.suffix == ".vtk":
        _, polydata = read_vtk(file_path)
        decode, encode = normalizer(file_path.parent / "txt/")
        data_dir = Path("/home/aistudio/txt")
        min_bounds, max_bounds = load_bound(data_dir, filename="global_bounds.txt")
        sdf_query_points = load_sdf_queries()
        sdf = load_sdf(file_path, sdf_query_points)
        sdf_query_points = (
            location_normalization(sdf_query_points, min_bounds, max_bounds)
            .unsqueeze(axis=0)
            .transpose([0, 4, 1, 2, 3])
        )
        data_dict = {
            "vertices": get_nodes(polydata),
            "velocity": velocity(polydata),
            "sdf": sdf,
            "sdf_query_points": sdf_query_points,
            "info": [{"velocity": 33.33}],
        }
    elif file_path.suffix == ".case":
        paddledict, multi_blockdata, _ = read_case(file_path)
        reference_area = calculate_frontal_area(
            file_name=file_path.as_posix(),
            vtk_data=multi_blockdata,
            proj_axis="X",
            debug=False,
        )
        data_dict = {
            # "vtk_data": vtk_data,
            "centroids": paddledict["centroids"],
            "normal": paddledict["normal"],
            "areas": paddledict["areas"],
            "wss": paddledict["wss"],
            "pressure": paddledict["pressure"],
            "file_path": file_path.as_posix(),
            "test_name": file_path.name,
            "reference_area": reference_area,
            "Cd": 0.0,
        }
    else:
        raise NotImplementedError

    return data_dict


def write(polydata, p, var_name, vtk_name):
    np_array = p.numpy()
    vtk_array = numpy_to_vtk(np_array)
    vtk_array.SetName(var_name)  
    polydata.GetCellData().AddArray(vtk_array)
    appendFilter = vtk.vtkAppendFilter()
    appendFilter.AddInputData(polydata)
    appendFilter.Update()
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(vtk_name)  
    writer.SetInputData(appendFilter.GetOutput())
    writer.Write()


def write_ply(polydata, filename):
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(filename)  
    writer.SetInputData(polydata)
    writer.Write()


if __name__ == "__main__":
    paddle.set_device("cpu")
    folder_1 = Path("./data_fake_starccm+/case_partial/")
    folder_2 = Path("./data_fake_starccm+/case/")
    save_dir = Path("./temp_case_to_paddledict/")

    os.makedirs(save_dir, exist_ok=True)

    file_list_1 = [folder_1 / f for f in os.listdir(folder_1) if f.endswith(".case")]
    file_list_2 = [folder_2 / f for f in os.listdir(folder_2) if f.endswith(".case")]
    file_list = file_list_2

    file_list = list(set(file_list))

    train_test_split = ["train" if random.random() < 0.9 else "test" for _ in file_list]

    centroid_x_mean = []
    centroid_y_mean = []
    centroid_z_mean = []
    centroid_x_std = []
    centroid_y_std = []
    centroid_z_std = []
    p_mean = []
    p_std = []
    wss_x_mean = []
    wss_y_mean = []
    wss_z_mean = []
    wss_x_std = []
    wss_y_std = []
    wss_z_std = []
    assert len(file_list) > 0, f"No files found, len(file_list) = {len(file_list)}"
    for i, f in enumerate(file_list):
        std = time.time()
        output_dir = save_dir / train_test_split[i] / (Path(f).name + ".paddledict")
        print(f"\nConverting [{f}] to [{output_dir}]")
        data_dict = read(f)
        print(f"number of cells: {data_dict['centroids'].shape[0]}")
        print(f"Totoal read time: {float(time.time() - std):.3f} seconds")
        paddle.save(data_dict, output_dir.as_posix())
        data_dict = paddle.load(output_dir.as_posix())

        centroid_x_mean.append(data_dict["centroids"][:, 0].mean())
        centroid_y_mean.append(data_dict["centroids"][:, 1].mean())
        centroid_z_mean.append(data_dict["centroids"][:, 2].mean())
        centroid_x_std.append(data_dict["centroids"][:, 0].std())
        centroid_y_std.append(data_dict["centroids"][:, 1].std())
        centroid_z_std.append(data_dict["centroids"][:, 2].std())

        print(
            "p_mean p_std: ",
            float(data_dict["pressure"].mean()),
            float(data_dict["pressure"].std()),
        )
        p_mean.append(data_dict["pressure"].mean())
        p_std.append(data_dict["pressure"].std())

        wss_x_mean.append(data_dict["wss"][:, 0].mean())
        wss_y_mean.append(data_dict["wss"][:, 1].mean())
        wss_z_mean.append(data_dict["wss"][:, 2].mean())
        wss_x_std.append(data_dict["wss"][:, 0].std())
        wss_y_std.append(data_dict["wss"][:, 1].std())
        wss_z_std.append(data_dict["wss"][:, 2].std())

    data_to_save = {
        "centroid_x_mean": paddle.to_tensor(centroid_x_mean).mean(),
        "centroid_y_mean": paddle.to_tensor(centroid_y_mean).mean(),
        "centroid_z_mean": paddle.to_tensor(centroid_z_mean).mean(),
        "centroid_x_std": paddle.to_tensor(centroid_x_std).mean(),
        "centroid_y_std": paddle.to_tensor(centroid_y_std).mean(),
        "centroid_z_std": paddle.to_tensor(centroid_z_std).mean(),
        "p_mean": paddle.to_tensor(p_mean).mean(),
        "p_std": paddle.to_tensor(p_std).mean(),
        "wss_x_mean": paddle.to_tensor(wss_x_mean).mean(),
        "wss_y_mean": paddle.to_tensor(wss_y_mean).mean(),
        "wss_z_mean": paddle.to_tensor(wss_z_mean).mean(),
        "wss_x_std": paddle.to_tensor(wss_x_std).mean(),
        "wss_y_std": paddle.to_tensor(wss_y_std).mean(),
        "wss_z_std": paddle.to_tensor(wss_z_std).mean(),
    }

    mean_std_dir = save_dir / "mean_std.paddledict"
    paddle.save(data_to_save, mean_std_dir.as_posix())
