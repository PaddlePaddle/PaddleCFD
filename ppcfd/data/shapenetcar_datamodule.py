import itertools
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import vtk
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from vtk.util.numpy_support import vtk_to_numpy


def radius(
    x: paddle.Tensor,
    y: paddle.Tensor,
    r: float,
    batch_x: Optional[paddle.Tensor] = None,
    batch_y: Optional[paddle.Tensor] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 32,
    batch_size: Optional[int] = None,
) -> paddle.Tensor:
    if x.numel() == 0 or y.numel() == 0:
        return paddle.empty([2, 0], dtype="int64", place=x.place)

    x = x.reshape([-1, 1]) if x.ndim == 1 else x
    y = y.reshape([-1, 1]) if y.ndim == 1 else y

    if batch_size is None:
        batch_size = 1
        if batch_x is not None:
            assert x.shape[0] == batch_x.numel()
            batch_size = int(batch_x.max()) + 1
        if batch_y is not None:
            assert y.shape[0] == batch_y.numel()
            batch_size = max(batch_size, int(batch_y.max()) + 1)
    assert batch_size > 0

    x = paddle.concat([x, 2 * r * batch_x.reshape([-1, 1])], axis=-1) if batch_x is not None else x
    y = paddle.concat([y, 2 * r * batch_y.reshape([-1, 1])], axis=-1) if batch_y is not None else y

    tree = cKDTree(x.numpy())

    def query_neighbors(idx):
        _, indices = tree.query(y[idx].numpy(), k=max_num_neighbors, distance_upper_bound=r + 1e-8)
        row = [idx] * len(indices)
        return row, indices

    rows, cols = [], []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(query_neighbors, range(y.shape[0]))
        for row, col in results:
            rows.extend(row)
            cols.extend(col)

    row_tensor = paddle.to_tensor(rows, dtype="int64")
    col_tensor = paddle.to_tensor(cols, dtype="int64")
    mask = col_tensor < tree.n

    return paddle.stack([row_tensor[mask], col_tensor[mask]], axis=0)


def radius_graph(
    x: paddle.Tensor,
    r: float,
    batch: Optional[paddle.Tensor] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
    num_workers: int = 32,
    batch_size: Optional[int] = None,
) -> paddle.Tensor:
    assert flow in ["source_to_target", "target_to_source"]

    edge_index = radius(
        x,
        x,
        r,
        batch,
        batch,
        max_num_neighbors if loop else max_num_neighbors + 1,
        num_workers,
        batch_size,
    )

    if flow == "source_to_target":
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return paddle.stack([row, col], axis=0)


class Data:
    def __init__(self, pos=None, x=None, y=None, edge_index=None, surf=None, sample_name=None):
        self.pos = pos
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.surf = surf
        self.sample_name = sample_name

    def __repr__(self):
        return (
            f"\nData(x={self._format_attr(self.x)}, "
            f"edge_index={self._format_attr(self.edge_index)}, "
            f"y={self._format_attr(self.y)}, "
            f"pos={self._format_attr(self.pos)}, "
            f"surf={self._format_attr(self.surf)}, "
            f"sample_name={self._format_attr(self.sample_name)})"
        )

    def _format_attr(self, attr):
        if attr is None:
            return "None"
        elif hasattr(attr, "shape"):
            return f"[{', '.join(map(str, attr.shape))}]"
        else:
            return str(attr)


def load_unstructured_grid_data(file_name):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    output = reader.GetOutput()
    return output


def unstructured_grid_data_to_poly_data(unstructured_grid_data):
    filter = vtk.vtkDataSetSurfaceFilter()
    filter.SetInputData(unstructured_grid_data)
    filter.Update()
    poly_data = filter.GetOutput()
    return poly_data, filter


def get_sdf(target, boundary):
    nbrs = NearestNeighbors(n_neighbors=1).fit(boundary)
    dists, indices = nbrs.kneighbors(target)
    neis = np.array([boundary[i[0]] for i in indices])
    dirs = (target - neis) / (dists + 1e-08)
    return dists.reshape(-1), dirs


def get_normal(unstructured_grid_data):
    poly_data, surface_filter = unstructured_grid_data_to_poly_data(unstructured_grid_data)
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(poly_data)
    normal_filter.SetAutoOrientNormals(1)
    normal_filter.SetConsistency(1)
    normal_filter.SetComputeCellNormals(1)
    normal_filter.SetComputePointNormals(0)
    normal_filter.Update()
    unstructured_grid_data.GetCellData().SetNormals(normal_filter.GetOutput().GetCellData().GetNormals())
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(unstructured_grid_data)
    c2p.Update()
    unstructured_grid_data = c2p.GetOutput()
    normal = vtk_to_numpy(c2p.GetOutput().GetPointData().GetNormals()).astype(np.double)
    normal /= np.max(np.abs(normal), axis=1, keepdims=True) + 1e-08
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-08
    if np.isnan(normal).sum() > 0:
        print(np.isnan(normal).sum())
        print("recalculate")
        return get_normal(unstructured_grid_data)
    return normal


def visualize_poly_data(poly_data, surface_filter, normal_filter=None):
    if normal_filter is not None:
        mask = vtk.vtkMaskPoints()
        mask.SetInputData(normal_filter.GetOutput())
        mask.Update()
        arrow = vtk.vtkArrowSource()
        arrow.Update()
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(mask.GetOutput())
        glyph.SetSourceData(arrow.GetOutput())
        glyph.SetVectorModeToUseNormal()
        glyph.SetScaleFactor(0.1)
        glyph.Update()
        norm_mapper = vtk.vtkPolyDataMapper()
        norm_mapper.SetInputData(normal_filter.GetOutput())
        glyph_mapper = vtk.vtkPolyDataMapper()
        glyph_mapper.SetInputData(glyph.GetOutput())
        norm_actor = vtk.vtkActor()
        norm_actor.SetMapper(norm_mapper)
        glyph_actor = vtk.vtkActor()
        glyph_actor.SetMapper(glyph_mapper)
        glyph_actor.GetProperty().SetColor(1, 0, 0)
        norm_render = vtk.vtkRenderer()
        norm_render.AddActor(norm_actor)
        norm_render.SetBackground(0, 1, 0)
        glyph_render = vtk.vtkRenderer()
        glyph_render.AddActor(glyph_actor)
        glyph_render.AddActor(norm_actor)
        glyph_render.SetBackground(0, 0, 1)
    scalar_range = poly_data.GetScalarRange()
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(surface_filter.GetOutputPort())
    mapper.SetScalarRange(scalar_range)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)
    renderer_window = vtk.vtkRenderWindow()
    renderer_window.AddRenderer(renderer)
    if normal_filter is not None:
        renderer_window.AddRenderer(norm_render)
        renderer_window.AddRenderer(glyph_render)
    renderer_window.Render()
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderer_window)
    interactor.Initialize()
    interactor.Start()


def get_datalist(
    samples, norm=False, coef_norm=None, save_dir="../data/shapenetcar/preprocessed_data", preprocessed=True
):
    dataset = []
    mean_in, mean_out = 0, 0
    std_in, std_out = 0, 0

    for k, s in tqdm(enumerate(samples), total=len(samples), desc="Processing Samples"):
        if preprocessed and save_dir is not None:
            save_path = os.path.join(save_dir, s)
            if not os.path.exists(save_path):
                continue
            init = np.load(os.path.join(save_path, "x.npy"))
            target = np.load(os.path.join(save_path, "y.npy"))
            pos = np.load(os.path.join(save_path, "pos.npy"))
            surf = np.load(os.path.join(save_path, "surf.npy"))
            edge_index = np.load(os.path.join(save_path, "edge_index.npy"))
        else:
            file_name_press = os.path.join(root, os.path.join(s, "quadpress_smpl.vtk"))
            file_name_velo = os.path.join(root, os.path.join(s, "hexvelo_smpl.vtk"))
            if not os.path.exists(file_name_press) or not os.path.exists(file_name_velo):
                continue
            unstructured_grid_data_press = load_unstructured_grid_data(file_name_press)
            unstructured_grid_data_velo = load_unstructured_grid_data(file_name_velo)
            velo = vtk_to_numpy(unstructured_grid_data_velo.GetPointData().GetVectors())
            press = vtk_to_numpy(unstructured_grid_data_press.GetPointData().GetScalars())
            points_velo = vtk_to_numpy(unstructured_grid_data_velo.GetPoints().GetData())
            points_press = vtk_to_numpy(unstructured_grid_data_press.GetPoints().GetData())
            edges_press = get_edges(unstructured_grid_data_press, points_press, cell_size=4)
            edges_velo = get_edges(unstructured_grid_data_velo, points_velo, cell_size=8)
            sdf_velo, normal_velo = get_sdf(points_velo, points_press)
            sdf_press = np.zeros(tuple(points_press.shape)[0])
            normal_press = get_normal(unstructured_grid_data_press)
            surface = {tuple(p) for p in points_press}
            exterior_indices = [i for i, p in enumerate(points_velo) if tuple(p) not in surface]
            velo_dict = {tuple(p): velo[i] for i, p in enumerate(points_velo)}
            pos_ext = points_velo[exterior_indices]
            pos_surf = points_press
            sdf_ext = sdf_velo[exterior_indices]
            sdf_surf = sdf_press
            normal_ext = normal_velo[exterior_indices]
            normal_surf = normal_press
            velo_ext = velo[exterior_indices]
            velo_surf = np.array([(velo_dict[tuple(p)] if tuple(p) in velo_dict else np.zeros(3)) for p in pos_surf])
            press_ext = np.zeros([len(exterior_indices), 1])
            press_surf = press
            init_ext = np.c_[pos_ext, sdf_ext, normal_ext]
            init_surf = np.c_[pos_surf, sdf_surf, normal_surf]
            target_ext = np.c_[velo_ext, press_ext]
            target_surf = np.c_[velo_surf, press_surf]
            surf = np.concatenate([np.zeros(len(pos_ext)), np.ones(len(pos_surf))])
            pos = np.concatenate([pos_ext, pos_surf])
            init = np.concatenate([init_ext, init_surf])
            target = np.concatenate([target_ext, target_surf])
            edge_index = get_edge_index(pos, edges_press, edges_velo)
            if save_dir is not None:
                save_path = os.path.join(save_dir, s)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.save(os.path.join(save_path, "x.npy"), init)
                np.save(os.path.join(save_path, "y.npy"), target)
                np.save(os.path.join(save_path, "pos.npy"), pos)
                np.save(os.path.join(save_path, "surf.npy"), surf)
                np.save(os.path.join(save_path, "edge_index.npy"), edge_index)
        surf = paddle.to_tensor(data=surf)
        pos = paddle.to_tensor(data=pos)
        x = paddle.to_tensor(data=init)
        y = paddle.to_tensor(data=target)
        edge_index = paddle.to_tensor(data=edge_index)
        if norm and coef_norm is None:
            if k == 0:
                old_length = tuple(init.shape)[0]
                mean_in = init.mean(axis=0)
                mean_out = target.mean(axis=0)
            else:
                new_length = old_length + tuple(init.shape)[0]
                mean_in += (init.sum(axis=0) - tuple(init.shape)[0] * mean_in) / new_length
                mean_out += (target.sum(axis=0) - tuple(init.shape)[0] * mean_out) / new_length
                old_length = new_length
        data = Data(pos=pos, x=x, y=y, surf=surf.astype(dtype="bool"), edge_index=edge_index, sample_name=s)
        dataset.append(data)
    if norm and coef_norm is None:
        for k, data in enumerate(dataset):
            if k == 0:
                old_length = tuple(data.x.numpy().shape)[0]
                std_in = ((data.x.numpy() - mean_in) ** 2).sum(axis=0) / old_length
                std_out = ((data.y.numpy() - mean_out) ** 2).sum(axis=0) / old_length
            else:
                new_length = old_length + tuple(data.x.numpy().shape)[0]
                std_in += (
                    ((data.x.numpy() - mean_in) ** 2).sum(axis=0) - tuple(data.x.numpy().shape)[0] * std_in
                ) / new_length
                std_out += (
                    ((data.y.numpy() - mean_out) ** 2).sum(axis=0) - tuple(data.x.numpy().shape)[0] * std_out
                ) / new_length
                old_length = new_length
        std_in = np.sqrt(std_in)
        std_out = np.sqrt(std_out)
        for data in dataset:
            data.x = ((data.x - mean_in) / (std_in + 1e-08)).astype(dtype="float32")
            data.y = ((data.y - mean_out) / (std_out + 1e-08)).astype(dtype="float32")
        coef_norm = mean_in, std_in, mean_out, std_out
        dataset = dataset, coef_norm
    elif coef_norm is not None:
        for data in dataset:
            data.x = ((data.x - coef_norm[0]) / (coef_norm[1] + 1e-08)).astype(dtype="float32")
            data.y = ((data.y - coef_norm[2]) / (coef_norm[3] + 1e-08)).astype(dtype="float32")
    return dataset


def get_edges(unstructured_grid_data, points, cell_size=4):
    edge_indeces = set()
    cells = vtk_to_numpy(unstructured_grid_data.GetCells().GetData()).reshape(-1, cell_size + 1)
    for i in range(len(cells)):
        for j, k in itertools.product(range(1, cell_size + 1), repeat=2):
            edge_indeces.add((cells[i][j], cells[i][k]))
            edge_indeces.add((cells[i][k], cells[i][j]))
    edges = [[], []]
    for u, v in edge_indeces:
        edges[0].append(tuple(points[u]))
        edges[1].append(tuple(points[v]))
    return edges


def get_edge_index(pos, edges_press, edges_velo):
    indices = {tuple(pos[i]): i for i in range(len(pos))}
    edges = set()
    for i in range(len(edges_press[0])):
        edges.add((indices[edges_press[0][i]], indices[edges_press[1][i]]))
    for i in range(len(edges_velo[0])):
        edges.add((indices[edges_velo[0][i]], indices[edges_velo[1][i]]))
    edge_index = np.array(list(edges)).T
    return edge_index


def get_samples(root):
    folds = [f"param{i}" for i in range(9)]
    samples = []
    for fold in folds:
        fold_samples = []
        files = os.listdir(os.path.join(root, fold))
        for file in files:
            path = os.path.join(root, os.path.join(fold, file))
            if os.path.isdir(path):
                fold_samples.append(os.path.join(fold, file))
        samples.append(fold_samples)
    return samples


def get_induced_graph(data, idx, num_hops):
    subset = set([idx])
    current_layer_nodes = set([idx])

    for _ in range(num_hops):
        neighbors = set()
        for node in current_layer_nodes:
            neighbors.update(data.edge_index[1][data.edge_index[0] == node].numpy())
            neighbors.update(data.edge_index[0][data.edge_index[1] == node].numpy())
        current_layer_nodes = neighbors - subset
        subset.update(current_layer_nodes)

    subset = paddle.to_tensor(list(subset), dtype="int64")
    mask = paddle.to_tensor(
        [(i in subset) and (j in subset) for i, j in zip(data.edge_index[0], data.edge_index[1])],
        dtype="bool",
    )
    sub_edge_index = data.edge_index[:, mask]
    return Data(x=data.x[subset], y=data.y[idx], edge_index=sub_edge_index)


def load_train_val_fold(args, preprocessed=True):
    samples = get_samples(args.data_module.data_dir)
    trainlst = []
    for i in range(len(samples)):
        if i == 0:
            continue
        trainlst += samples[i]
    vallst = samples[0] if 0 <= 0 < len(samples) else None
    trainlst = sorted(trainlst)[: args.data_module.n_train_num]
    vallst = sorted(vallst)[: args.data_module.n_val_num]
    print("n_train_num", len(trainlst))
    print("n_valid", len(vallst))
    train_dataset, coef_norm = get_datalist(
        trainlst, save_dir=args.data_module.data_dir, norm=True, preprocessed=preprocessed
    )
    val_dataset = get_datalist(
        vallst, save_dir=args.data_module.data_dir, coef_norm=coef_norm, preprocessed=preprocessed
    )
    print("train_dataset[0]", train_dataset[0])
    print("val_dataset[0]", val_dataset[0])
    print("load data finish")
    return train_dataset, val_dataset, coef_norm


def pc_normalize(pc):
    centroid = paddle.mean(pc, axis=0)
    pc = pc - centroid
    m = paddle.max(paddle.sqrt(paddle.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def get_shape(data, max_n_point=8192, normalize=True, use_height=False):
    surf_indices = paddle.nonzero(data.surf).squeeze().numpy().tolist()
    if len(surf_indices) > max_n_point:
        surf_indices = np.array(random.sample(surf_indices, max_n_point))
    shape_pc = paddle.gather(data.pos, paddle.to_tensor(surf_indices, dtype="int64"))
    if normalize:
        shape_pc = pc_normalize(shape_pc)
    if use_height:
        gravity_dim = 1
        height_array = shape_pc[:, gravity_dim : gravity_dim + 1] - paddle.min(
            shape_pc[:, gravity_dim : gravity_dim + 1]
        )
        shape_pc = paddle.concat((shape_pc, height_array), axis=1)

    return shape_pc


def create_edge_index_radius(data, r, max_neighbors=32):
    data.edge_index = radius_graph(x=data.pos, r=r, loop=True, max_num_neighbors=max_neighbors)
    return data


class GraphDataset(paddle.io.Dataset):
    def __init__(self, datalist, use_height=False, use_cfd_mesh=True, r=None, transform=None):
        super().__init__()
        self.datalist = datalist
        self.transform = transform
        self.use_height = use_height
        self._indices: Optional[Sequence] = None
        self.fake_data = paddle.ones([3682, 3])
        for i in tqdm(range(len(self.datalist)), desc="Caching Samples"):
            data = self.datalist[i]
            data, _ = self.get(self.indices()[i])
            data = data if self.transform is None else self.transform(data)
            self.datalist[i] = [data.x, data.y, data.surf, data.sample_name]

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx: Union[int, np.integer, paddle.Tensor, np.ndarray]) -> Tuple["Data", paddle.Tensor]:
        return self.datalist[idx]

    def get(self, idx):
        data = self.datalist[idx]
        shape = get_shape(data, use_height=self.use_height)
        return data, shape

    def indices(self) -> Sequence:
        return range(len(self.datalist)) if self._indices is None else self._indices
