import os

import numpy as np
from matplotlib import pyplot as plt

colors = ["blue", "orange", "green", "purple", "olive", "red", "magenta", "black"]


def plot_ax(ax, array, title=None, fontsize=10, cmap="bwr", sym=False):
    if sym:
        vmax = np.max(np.abs(array))
        im = ax.imshow(array, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(array, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.ax.tick_params(labelsize=fontsize - 4)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)


def plot_2d_pde(
    output: np.ndarray,
    data_all,
    times,
    input_len,
    plot_title,
    filename,
    folder="",
    input_plot_len=-1,
    output_plot_len=-1,
    dim=-1,
):
    """
    output:     (output_len, x_num, x_num, data_dim)
    data_all:   (input_len + output_len, x_num, x_num, data_dim)
    times:      (input_len + output_len)
    """
    if dim < 0:
        dim = output.shape[-1]
    output_len = output.shape[0]
    if input_plot_len < 0:
        input_plot_len = input_len
    if output_plot_len < 0:
        output_plot_len = output_len
    total_col = input_plot_len + output_plot_len * 3
    fig, axs = plt.subplots(
        dim, total_col, squeeze=False, figsize=(4.3 * total_col, 4 * dim)
    )
    for input_step in range(input_plot_len):
        for j in range(dim):
            plot_ax(
                axs[j, input_step],
                data_all[input_step, ..., j],
                f"input, step {input_step}, t={times[input_step]:.2f}",
            )
    for idx, output_step in enumerate(range(output_len - output_plot_len, output_len)):
        step = input_len + output_step
        for j in range(dim):
            cur_idx = input_plot_len + idx * 3
            cur_target = data_all[step, ..., j]
            plot_ax(
                axs[j, cur_idx], cur_target, f"target, step {step}, t={times[step]:.2f}"
            )
            cur_output = output[output_step, ..., j]
            plot_ax(axs[j, cur_idx + 1], cur_output, "output")
            diff = cur_target - cur_output
            plot_ax(axs[j, cur_idx + 2], diff, "diff", sym=True)
    for ax in axs.flat:
        ax.label_outer()
        ax.tick_params(axis="both", labelsize=8)
    plt.suptitle(plot_title, fontsize=20)
    plt.tight_layout()
    path = os.path.join(folder, filename + ".png")
    plt.savefig(path)
    plt.close(fig)
    return path


def plot_ax_formal(ax, array, title=None, fontsize=15, cmap="RdBu_r", sym=False):
    if sym:
        vmax = np.max(np.abs(array))
        im = ax.imshow(array, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(array, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.ax.tick_params(labelsize=fontsize)
    ax.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )
    if title is not None:
        ax.set_title(title, fontsize=fontsize)


def plot_2d_pde_formal(
    output: np.ndarray,
    data_all,
    times,
    input_len,
    plot_title,
    filename,
    folder="",
    input_plot_len=-1,
    output_plot_len=-1,
    dim=-1,
):
    """
    output:     (output_len, x_num, x_num, data_dim)
    data_all:   (input_len + output_len, x_num, x_num, data_dim)
    times:      (input_len + output_len)
    """
    if dim < 0:
        dim = output.shape[-1]
    output_len = output.shape[0]
    if output_plot_len < 0:
        output_plot_len = output_len
    total_col = output_plot_len
    total_row = dim * 3
    fig, axs = plt.subplots(
        total_row, total_col, squeeze=False, figsize=(4.3 * total_col, 4 * total_row)
    )
    for idx, output_step in enumerate(range(output_len - output_plot_len, output_len)):
        step = input_len + output_step
        for j in range(dim):
            cur_target = data_all[step, ..., j]
            plot_ax_formal(axs[j * 3, idx], cur_target, f"Target")
            cur_output = output[output_step, ..., j]
            plot_ax_formal(axs[j * 3 + 1, idx], cur_output, "Prediction")
            diff = cur_target - cur_output
            plot_ax_formal(axs[j * 3 + 2, idx], diff, "Difference", sym=True)
    for ax in axs.flat:
        ax.label_outer()
        ax.tick_params(axis="both", labelsize=8)
    plt.suptitle(plot_title + "\n\n", fontsize=20)
    plt.tight_layout()
    path = os.path.join(folder, filename + ".png")
    plt.savefig(path)
    plt.close(fig)
    return path
