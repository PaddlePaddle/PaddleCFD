import argparse
import os
import numpy as np
import paddle

from data_utils import generate_data
from ppcfd.models.nomad.antiderivative.nomad_antiderivative import OperatorModel


def main(n, decoder):

    P = 500
    m = 500
    num_test = 1000

    rng = np.random.default_rng(123)

    freqs = rng.uniform(0, 10, size=(num_test,))

    U, y, s = generate_data(freqs, m, P)

    U = paddle.to_tensor(U).unsqueeze(-1)
    y = paddle.to_tensor(y).unsqueeze(-1)

    ds = 1

    if decoder == "nonlinear":

        branch_layers = [m, 100, 100, 100, 100, 100, ds * n]

        trunk_layers = [ds * n * 2, 100, 100, 100, 100, 100, ds]

    else:

        branch_layers = [m, 100, 100, 100, 100, 100, ds * n]

        trunk_layers = [1, 100, 100, 100, 100, 100, ds * n]

    model = OperatorModel(
        branch_layers,
        trunk_layers,
        m,
        P,
        n,
        decoder,
        ds,
    )

    model.set_state_dict(
        paddle.load("checkpoints/nomad_antiderivative.pdparams")
    )

    model.eval()

    pred = model.predict((U, y)).numpy()

    os.makedirs("results", exist_ok=True)

    np.save("results/pred.npy", pred)

    print("Prediction saved.")
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("n", type=int)
    parser.add_argument("decoder", type=str)

    args = parser.parse_args()

    main(args.n, args.decoder)