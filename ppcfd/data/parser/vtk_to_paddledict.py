import numpy as np
import vtk
import paddle
from vtk.util.numpy_support import vtk_to_numpy


def read_vtk_file_and_compute_all_data(input_file, mesh_name=None, tri_filter=True):
    print(f"reading vtk file : {input_file}")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(input_file)
    reader.Update()
    polydata = reader.GetOutput()


    if "p" in point_data_keys:
        # 获取点数据中的压力值
        pressure_data = polydata.GetPointData().GetArray("p")

        # 遍历所有单元并计算平均压力
        cell_pressures = []
        for i in range(num_cells):
            cell = polydata.GetCell(i)
            cell_points = cell.GetPointIds()
            cell_pressure_sum = 0
            for j in range(cell_points.GetNumberOfIds()):
                point_id = cell_points.GetId(j)
                cell_pressure_sum += pressure_data.GetValue(point_id)
            cell_avg_pressure = cell_pressure_sum / cell_points.GetNumberOfIds()
            cell_pressures.append(cell_avg_pressure)
        cell_pressures = np.array(cell_pressures)
    else:
        cell_pressures = None

    # # 转换为NumPy数组
    numpy_cell_areas = vtk_to_numpy(
        cell_size_filter.GetOutput().GetCellData().GetArray("Area")
    )
    numpy_cell_normals = vtk_to_numpy(
        normals_filter.GetOutput().GetCellData().GetNormals()
    )

    # 计算边界框
    bounds = polydata.GetBounds()
    # bounds是一个包含六个元素的元组，分别对应(xmin, xmax, ymin, ymax, zmin, zmax)
    length = bounds[1] - bounds[0]  # X轴上的距离
    width = bounds[3] - bounds[2]  # Y轴上的距离
    height = bounds[5] - bounds[4]  # Z轴上的距离
    numpy_cell_centers = (numpy_cell_centers - np.array([1.5760515, -0.019655414, 0.5975443])) / np.array([1.3543817, 0.62339926, 0.397299])

    # _组织数据到字典
    data_dict = {
        "Point_Coordinates": vtk_to_numpy(polydata.GetPoints().GetData()),
        "centroids": numpy_cell_centers.astype('float32').reshape([1, -1, 3]),
    }


    return data_dict


if __name__ == "__main__":
    data_dict = read_vtk_file_and_compute_all_data("./DrivAer_F_D_WM_WW_0014.vtk")
    paddle.save(data_dict, "test_0014.paddledict")
