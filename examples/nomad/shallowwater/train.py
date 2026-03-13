import argparse
import os
import numpy as np
import paddle
from tqdm import trange

from dataset import DataGenerator, output_construction
from ppcfd.models.nomad.shallowwater.nomad_model import OperatorModel


def train(model, train_dataset, test_dataset, optimizer, scheduler, nIter):

    model.train()

    pbar = trange(nIter)

    for it in pbar:

        train_batch = next(train_dataset)

        loss = model.loss(train_batch)

        loss.backward()

        optimizer.step()

        optimizer.clear_grad()

        scheduler.step()

        if it % 100 == 0:

            test_batch = next(test_dataset)

            model.eval()

            with paddle.no_grad():

                loss_test = model.loss(test_batch)

                error = model.L2error(test_batch)

            pbar.set_postfix(
                train_loss=float(loss.numpy()),
                test_loss=float(loss_test.numpy()),
                test_error=float(error.numpy()),
            )

            model.train()


def main(n, decoder):

    TRAINING_ITERATIONS = 100000

    P = 128
    m = 1024

    num_train = 1000
    num_test = 1000

    batch_size = 100

    du = 3
    dy = 3
    ds = 3

    Nx = 32
    Ny = 32
    Nt = 5

    d = np.load("./train_SW.npz")

    u_train = d["u_train"]
    S_train = d["S_train"]
    T = d["T"]
    CX = d["CX"]
    CY = d["CY"]

    d = np.load("./test_SW.npz")

    u_test = d["u_test"]
    S_test = d["S_test"]

    U_train = u_train.reshape(num_train, Nx * Ny, du).astype(np.float32)
    U_test = u_test.reshape(num_test, Nx * Ny, du).astype(np.float32)

    s_train = np.zeros((num_train, P, ds), dtype=np.float32)
    y_train = np.zeros((num_train, P, dy), dtype=np.float32)

    for i in range(num_train):

        s_train[i], y_train[i] = output_construction(
            S_train[i], T, CX, CY, P=P, ds=ds, Nx=Nx, Ny=Ny, Nt=Nt
        )

    U_train = paddle.to_tensor(U_train)
    y_train = paddle.to_tensor(y_train)
    s_train = paddle.to_tensor(s_train)

    train_dataset = DataGenerator(U_train, y_train, s_train, batch_size)

    if decoder == "nonlinear":

        branch_layers = [m * du, 100, 100, 100, 100, 100, ds * n]

        trunk_layers = [ds * n * 2, 100, 100, 100, 100, 100, ds]

    else:

        branch_layers = [m * du, 100, 100, 100, 100, 100, ds * n]

        trunk_layers = [dy, 100, 100, 100, 100, 100, ds * n]

    model = OperatorModel(branch_layers, trunk_layers, n=n, decoder=decoder, ds=ds)

    optimizer = paddle.optimizer.Adam(
        learning_rate=0.001,
        parameters=model.parameters(),
    )

    scheduler = paddle.optimizer.lr.StepDecay(0.001, step_size=100, gamma=0.99)

    train(
        model,
        iter(train_dataset),
        iter(train_dataset),
        optimizer,
        scheduler,
        TRAINING_ITERATIONS,
    )

    os.makedirs("checkpoints", exist_ok=True)

    paddle.save(model.state_dict(), f"checkpoints/nomad_sw_{decoder}_n{n}.pdparams")

    print("Model saved.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("n", type=int, nargs="+")
    parser.add_argument("decoder", type=str, nargs="+")

    args = parser.parse_args()

    main(args.n[0], args.decoder[0])