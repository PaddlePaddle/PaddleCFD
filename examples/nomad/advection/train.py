import argparse
import os
import timeit

import numpy as np
import paddle
from tqdm import trange

from dataset import DataGenerator
from utils import save_model

from ppcfd.models.nomad.advection.operator_model import OperatorModel


def train_model(model, train_dataset, test_dataset, optimizer, nIter):

    pbar = trange(nIter)

    for it in pbar:

        train_batch = next(train_dataset)
        test_batch = next(test_dataset)

        loss_train = model.loss(train_batch)

        loss_train.backward()

        optimizer.step()
        optimizer.clear_grad()

        if it % 100 == 0:

            model.eval()

            with paddle.no_grad():

                loss_test = model.loss(test_batch)

                errorTest = model.L2error(test_batch)

            pbar.set_postfix(
                {
                    "train_loss": float(loss_train.numpy()),
                    "test_loss": float(loss_test.numpy()),
                    "test_error": float(errorTest.numpy()),
                }
            )

            model.train()


def main(n, decoder):

    TRAINING_ITERATIONS = 20000

    P = 25600
    m = 256

    num_train = 1000
    num_test = 1000

    batch_size = 4

    du = 1
    dy = 2
    ds = 1

    Nx = 256
    Nt = 100

    data = np.load("./pure_advection_traintest.npz")

    U_train = data["ic"][:num_train]
    s_train = data["solution"][:num_train]

    U_test = data["ic"][-num_test:]
    s_test = data["solution"][-num_test:]

    x = np.linspace(0, 2, num=Nx)
    t = np.linspace(0, 1, num=Nt)

    TT, XX = np.meshgrid(t, x, indexing="ij")

    y = np.concatenate(
        (TT.flatten()[:, None], XX.flatten()[:, None]), axis=-1
    )

    y_train = np.tile(y[None, ...], (num_train, 1, 1))
    y_test = np.tile(y[None, ...], (num_test, 1, 1))

    U_train = paddle.to_tensor(U_train, dtype="float32")
    y_train = paddle.to_tensor(y_train, dtype="float32")
    s_train = paddle.to_tensor(s_train, dtype="float32")

    U_test = paddle.to_tensor(U_test, dtype="float32")
    y_test = paddle.to_tensor(y_test, dtype="float32")
    s_test = paddle.to_tensor(s_test, dtype="float32")

    U_train = paddle.reshape(U_train, [num_train, m, du])
    y_train = paddle.reshape(y_train, [num_train, P, dy])
    s_train = paddle.reshape(s_train, [num_train, P, ds])

    U_test = paddle.reshape(U_test, [num_test, m, du])
    y_test = paddle.reshape(y_test, [num_test, P, dy])
    s_test = paddle.reshape(s_test, [num_test, P, ds])

    train_dataset = DataGenerator(U_train, y_train, s_train, batch_size)
    test_dataset = DataGenerator(U_test, y_test, s_test, batch_size)

    if decoder == "nonlinear":

        branch_layers = [m, 100, 100, 100, 100, 100, ds * n]

        trunk_layers = [ds * n * 2, 100, 100, 100, 100, 100, ds]

    else:

        branch_layers = [m, 100, 100, 100, 100, 100, ds * n]

        trunk_layers = [dy, 100, 100, 100, 100, 100, ds * n]

    model = OperatorModel(
        branch_layers,
        trunk_layers,
        m,
        P,
        n,
        decoder,
        ds,
    )

    model.count_params()

    optimizer = paddle.optimizer.Adam(
        learning_rate=0.001,
        parameters=model.parameters(),
    )

    start = timeit.default_timer()

    train_model(
        model,
        iter(train_dataset),
        iter(test_dataset),
        optimizer,
        TRAINING_ITERATIONS,
    )

    print("Training time:", timeit.default_timer() - start)

    save_model(
        model,
        "checkpoints/nomad_advection_model.pdparams",
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("n", type=int)

    parser.add_argument("decoder", type=str)

    args = parser.parse_args()

    main(args.n, args.decoder)