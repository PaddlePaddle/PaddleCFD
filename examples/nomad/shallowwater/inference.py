import argparse
import os
import numpy as np
import paddle

from ppcfd.models.nomad.shallowwater.nomad_model import OperatorModel


def main(n, decoder, checkpoint=None):

    num_test = 1000

    Nx = 32
    Ny = 32
    Nt = 5

    ds = 3
    du = 3
    dy = 3

    d = np.load("./test_SW.npz")

    u_test = d["u_test"]
    S_test = d["S_test"]

    T = d["T"]
    CX = d["CX"]
    CY = d["CY"]

    U_test = u_test.reshape(num_test, Nx * Ny, du).astype(np.float32)

    T_grid, X_grid, Y_grid = np.meshgrid(T, CX, CY, indexing="ij")

    Y_full = np.concatenate(
        (
            T_grid.flatten()[:, None],
            X_grid.flatten()[:, None],
            Y_grid.flatten()[:, None],
        ),
        axis=-1,
    )

    Y_full = np.tile(Y_full[None, :, :], (num_test, 1, 1)).astype(np.float32)

    U_test = paddle.to_tensor(U_test)

    Y_full = paddle.to_tensor(Y_full)

    m = Nx * Ny

    if decoder == "nonlinear":

        branch_layers = [m * du, 100, 100, 100, 100, 100, ds * n]

        trunk_layers = [ds * n * 2, 100, 100, 100, 100, 100, ds]

    else:

        branch_layers = [m * du, 100, 100, 100, 100, 100, ds * n]

        trunk_layers = [dy, 100, 100, 100, 100, 100, ds * n]

    model = OperatorModel(branch_layers, trunk_layers, n=n, decoder=decoder, ds=ds)

    if checkpoint is None:
        checkpoint = f"checkpoints/nomad_sw_{decoder}_n{n}.pdparams"

    model.load_dict(paddle.load(checkpoint))
    print("Model loaded:", checkpoint)

    model.eval()

    S_test = S_test.reshape(num_test, Nt * Nx * Ny, ds)

    pred = np.zeros_like(S_test)

    with paddle.no_grad():

        for i in range(num_test):

            u = U_test[i : i + 1]

            y = Y_full[i : i + 1]

            out = model.predict((u, y))

            pred[i] = out.numpy()

    os.makedirs("results", exist_ok=True)

    np.save("results/pred_test.npy", pred)

    print("Inference finished.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("n", type=int, nargs="+")
    parser.add_argument("decoder", type=str, nargs="+")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model parameters.",
    )

    args = parser.parse_args()

    main(args.n[0], args.decoder[0], args.checkpoint)
