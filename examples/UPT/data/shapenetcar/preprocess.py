import argparse
from pathlib import Path

import numpy as np
import paddle
import vtk
from tqdm import tqdm
from vtk.util.numpy_support import vtk_to_numpy


DOMAIN_MIN = (-2.0, -1.0, -4.5)
DOMAIN_MAX = (2.0, 4.5, 6.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess ShapeNetCar MLCFD data for the UPT example."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to mlcfd_data/training_data containing param0 ... param8.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Output path. UPT expects this directory to be named preprocessed.",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[32, 40, 48, 64, 80],
        help="SDF grid resolutions to generate.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=889,
        help="Expected number of processed samples. Set to 0 to disable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files even if they already exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process the first N samples. Useful for smoke tests.",
    )
    return parser.parse_args()


def read_unstructured_grid(vtk_path):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(vtk_path))
    reader.Update()
    grid = reader.GetOutput()
    if grid.GetNumberOfPoints() == 0:
        raise RuntimeError(f"failed to read points from {vtk_path}")
    return grid


def get_referenced_point_indices(grid):
    cell_array = grid.GetCells()
    if hasattr(cell_array, "GetConnectivityArray"):
        connectivity = vtk_to_numpy(cell_array.GetConnectivityArray())
        offsets = vtk_to_numpy(cell_array.GetOffsetsArray())
        if not np.all(np.diff(offsets) == 4):
            raise RuntimeError("quadpress_smpl.vtk is expected to contain quad cells")
        cells = connectivity.reshape(-1, 4)
    else:
        raw_cells = vtk_to_numpy(cell_array.GetData())
        legacy_cells = raw_cells.reshape(-1, 5)
        if not np.all(legacy_cells[:, 0] == 4):
            raise RuntimeError("quadpress_smpl.vtk is expected to contain quad cells")
        cells = legacy_cells[:, 1:]
    if cells.shape[1] != 4:
        raise RuntimeError("quadpress_smpl.vtk is expected to contain quad cells")
    return np.unique(cells.reshape(-1))


def get_pressure(grid, sample_dir):
    scalars = grid.GetPointData().GetScalars()
    if scalars is not None:
        return vtk_to_numpy(scalars).reshape(-1)
    press_path = sample_dir / "press.npy"
    if not press_path.exists():
        raise FileNotFoundError(f"missing pressure data: {press_path}")
    return np.load(press_path).reshape(-1)


def build_implicit_distance(grid):
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(grid)
    surface_filter.Update()

    implicit = vtk.vtkImplicitPolyDataDistance()
    implicit.SetInput(surface_filter.GetOutput())
    return implicit


def compute_sdf(implicit, resolution):
    sampler = vtk.vtkSampleFunction()
    sampler.SetImplicitFunction(implicit)
    sampler.SetModelBounds(
        DOMAIN_MIN[0],
        DOMAIN_MAX[0],
        DOMAIN_MIN[1],
        DOMAIN_MAX[1],
        DOMAIN_MIN[2],
        DOMAIN_MAX[2],
    )
    sampler.SetSampleDimensions(resolution, resolution, resolution)
    sampler.ComputeNormalsOff()
    sampler.Update()

    sdf = vtk_to_numpy(sampler.GetOutput().GetPointData().GetScalars())
    return sdf.astype(np.float32).reshape(
        (resolution, resolution, resolution), order="F"
    )


def save_tensor(array, path):
    paddle.save(paddle.to_tensor(array.astype(np.float32)), str(path))


def process_sample(sample_dir, out_dir, resolutions, overwrite):
    vtk_path = sample_dir / "quadpress_smpl.vtk"
    if not vtk_path.exists():
        return False

    required_outputs = [
        out_dir / "mesh_points.th",
        out_dir / "pressure.th",
        *[out_dir / f"sdf_res{res}.th" for res in resolutions],
    ]
    if not overwrite and all(path.exists() for path in required_outputs):
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    grid = read_unstructured_grid(vtk_path)
    point_indices = get_referenced_point_indices(grid)
    points = vtk_to_numpy(grid.GetPoints().GetData()).astype(np.float32)[point_indices]
    pressure = get_pressure(grid, sample_dir).astype(np.float32)[point_indices]

    save_tensor(points, out_dir / "mesh_points.th")
    save_tensor(pressure, out_dir / "pressure.th")

    implicit = build_implicit_distance(grid)
    for resolution in resolutions:
        sdf = compute_sdf(implicit, resolution)
        save_tensor(sdf, out_dir / f"sdf_res{resolution}.th")
    return True


def iter_samples(src):
    for param_idx in range(9):
        param_dir = src / f"param{param_idx}"
        if not param_dir.exists():
            raise FileNotFoundError(f"missing directory: {param_dir}")
        for sample_dir in sorted(path for path in param_dir.iterdir() if path.is_dir()):
            yield param_idx, sample_dir


def main():
    args = parse_args()
    args.src = args.src.expanduser().resolve()
    args.dst = args.dst.expanduser().resolve()

    processed = 0
    samples = list(iter_samples(args.src))
    if args.limit is not None:
        samples = samples[: args.limit]

    for param_idx, sample_dir in tqdm(samples, desc="preprocessing ShapeNetCar"):
        out_dir = args.dst / f"param{param_idx}" / sample_dir.name
        processed += int(
            process_sample(
                sample_dir=sample_dir,
                out_dir=out_dir,
                resolutions=args.resolutions,
                overwrite=args.overwrite,
            )
        )

    print(f"processed {processed} samples into {args.dst}")
    if args.expected_count and args.limit is None and processed != args.expected_count:
        raise RuntimeError(
            f"expected {args.expected_count} processed samples, got {processed}"
        )


if __name__ == "__main__":
    main()
