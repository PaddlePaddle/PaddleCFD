import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def normalize_physics_props(case_params: Dict[str, float]):
    """
    Normalize the physics properties in-place.
    """
    density_mean = 5
    density_std = 4
    viscosity_mean = 0.00238
    viscosity_std = 0.005
    case_params["density"] = (case_params["density"] - density_mean) / density_std
    case_params["viscosity"] = (
        case_params["viscosity"] - viscosity_mean
    ) / viscosity_std


def normalize_bc(case_params: Dict[str, float], key: str):
    """
    Normalize the boundary conditions in-place.
    """
    case_params[key] = case_params[key] / 50 - 0.5


def plot_contour(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    plt.tricontourf(x, y, z)
    plt.colorbar()
    plt.show()


def dump_json(data, path):
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    """Load a JSON object from a file"""
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def plot(inputs, outputs, labels, output_file: Path):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 5))
    axs = axs.flatten()
    last_im = None

    def sub_plot(idx, data, title):
        nonlocal last_im
        ax = axs[idx - 1]
        im = ax.imshow(data.cpu().detach().numpy())
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        last_im = im

    sub_plot(1, inputs[0], "input u")
    sub_plot(4, inputs[1], "input v")
    sub_plot(2, labels[0], "label u")
    sub_plot(5, labels[1], "label v")
    sub_plot(3, outputs[0], "output u")
    sub_plot(6, outputs[1], "output v")
    plt.savefig(output_file)
    plt.clf()
    plt.close()


def plot_loss(losses, out: Path):
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(out)
    plt.clf()
    plt.close()
