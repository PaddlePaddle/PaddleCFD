import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


matplotlib.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "serif"


def interpolate_to_grid_with_mask(grid_x, grid_y, field, NX, NY, grid_xa=None, grid_ya=None, get_mask=False):
    """
    Interpolate unstructured data (grid_x, grid_y, field) onto a structured grid of size (NX, NY)
    and create a binary mask that is 0 inside the airfoil and 1 outside.

    Parameters:
        grid_x (np.array): x-coordinates of the unstructured data points, shape (n_nodes,)
        grid_y (np.array): y-coordinates of the unstructured data points, shape (n_nodes,)
        field (np.array): field values at the unstructured data points, shape (n_nodes,)
        NX (int): Number of points in the structured grid along the x-axis
        NY (int): Number of points in the structured grid along the y-axis
        grid_xa (np.array): x-coordinates of the airfoil geometry, shape (n_nodesa,)
        grid_ya (np.array): y-coordinates of the airfoil geometry, shape (n_nodesa,)

    Returns:
        np.array: Interpolated field values on the structured grid, shape (NX, NY)
        np.array: Binary mask, 0 inside the airfoil and 1 outside, shape (NX, NY)
    """
    target_x = np.linspace(np.min(grid_x), np.max(grid_x), NX)
    target_y = np.linspace(np.min(grid_y), np.max(grid_y), NY)
    grid_X, grid_Y = np.meshgrid(target_x, target_y)
    grid_Z = griddata((grid_x, grid_y), field, (grid_X, grid_Y), method="linear")
    if get_mask:
        airfoil_path = Path(np.column_stack((grid_xa, grid_ya)))
        points = np.vstack((grid_X.ravel(), grid_Y.ravel())).T
        inside_airfoil = airfoil_path.contains_points(points).reshape(NX, NY)
        mask = np.where(inside_airfoil, 0, 1)
    else:
        mask = None
    return grid_Z, mask


def fill_nan_with_nearest(data):
    """
    Replace NaN values in the 3D dataset with the nearest non-NaN neighbor.

    Parameters:
        data (np.array): Input 3D array with shape (nt, nx, ny) containing NaNs.

    Returns:
        np.array: Modified array with NaNs replaced by nearest non-NaN neighbors.
    """
    nan_mask = np.isnan(data)
    distances, indices = distance_transform_edt(nan_mask, return_indices=True)
    filled_data = data[tuple(indices)]
    data[nan_mask] = filled_data[nan_mask]
    return data


NX = 1024
NY = 1024
delta_y = 128
skip_x = skip_y = 2
script_dir = os.path.dirname(os.path.abspath(__file__))
print("Current directory:", script_dir)
script_dir = "/workspace/workspace/NO_DM-develop/Airfoil-LES_paddle/data-prepare"

file = h5py.File(os.path.join(script_dir, "airfoilLES_grid.h5"), "r")
grid_x = np.array(file["x"])
grid_y = np.array(file["y"])
grid_xa = np.array(file["xa"])
grid_ya = np.array(file["ya"])
plt.scatter(grid_x, grid_y, s=0.01, c="red")
plt.scatter(grid_xa, grid_ya, s=1, c="blue")
plt.savefig(os.path.join(script_dir, "grid.png"), dpi=600, bbox_inches="tight")

ux_ls = []
ux_outliers = []
uy_outliers = []
uz_outliers = []
for i in tqdm(range(1, 3901, 5)):
    t_idx = str(100000 + i)[1:]
    file = h5py.File(
        os.path.join(script_dir, "airfoilLES_midspan", f"airfoilLES_t{t_idx}.h5"),
        "r",
    )
    print(f"reading: {file}")
    ux = np.array(file["ux"])
    ux_min, ux_max = ux.min(), ux.max()
    if ux_min < -2 or ux_max > 2:
        ux_outliers.append((t_idx, ux_min, ux_max))
    uy = np.array(file["uy"])
    uy_min, uy_max = uy.min(), uy.max()
    if uy_min < -2 or uy_max > 2:
        uy_outliers.append((t_idx, uy_min, uy_max))
    uz = np.array(file["uz"])
    uz_min, uz_max = uz.min(), uz.max()
    if uz_min < -2 or uz_max > 2:
        uz_outliers.append((t_idx, uz_min, uz_max))
    field = ux
    if i == 1:
        grid_field, grid_mask = interpolate_to_grid_with_mask(
            grid_x, grid_y, field, NX, NY, grid_xa, grid_ya, get_mask=True
        )
        grid_mask = grid_mask[512 - delta_y : 512 + delta_y][::skip_x, ::skip_y]
    else:
        grid_field, _ = interpolate_to_grid_with_mask(grid_x, grid_y, field, NX, NY)
    temp_grid_field = grid_field[512 - delta_y : 512 + delta_y][::skip_x, ::skip_y]
    ux_ls.append(temp_grid_field)
    file.close()

UX = np.array(ux_ls)
np.save(os.path.join(script_dir, "UX.npy"), UX)
np.save(os.path.join(script_dir, "mask.npy"), grid_mask)

UX_nan_filtered = fill_nan_with_nearest(UX)
np.save(os.path.join(script_dir, "UX_nan_filtered.npy"), UX_nan_filtered)
