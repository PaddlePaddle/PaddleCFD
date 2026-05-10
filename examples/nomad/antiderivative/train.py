import argparse
import os
import timeit
import numpy as np
import paddle
from tqdm import trange

from data_utils import DataGenerator, generate_data
from ppcfd.models.nomad.antiderivative.nomad_antiderivative import OperatorModel


def train_model(model, train_dataset, test_dataset, optimizer, scheduler, nIter):

    pbar = trange(nIter)

    model.train()

    for it in pbar:

        train_batch = next(train_dataset)
        test_batch = next(test_dataset)

        loss = model.loss(train_batch)

        loss.backward()

        optimizer.step()

        optimizer.clear_grad()

        scheduler.step()

        if it % 100 == 0:

            model.eval()

            with paddle.no_grad():

                test_loss = model.loss(test_batch)

                error = model.L2error(test_batch)

            pbar.set_postfix(
                train_loss=float(loss.numpy()),
                test_loss=float(test_loss.numpy()),
                test_error=float(error.numpy()),
            )

            model.train()


def main(n, decoder, iterations=20000, train_seed=0, test_seed=12345):

    P = 500
    m = 500

    num_train = 1000
    num_test = 1000

    batch_size = 100

    train_rng = np.random.default_rng(train_seed)
    test_rng = np.random.default_rng(test_seed)

    train_freqs = train_rng.uniform(0, 10, size=(num_train,))
    test_freqs = test_rng.uniform(0, 10, size=(num_test,))

    U_train, y_train, s_train = generate_data(train_freqs, m, P)
    U_test, y_test, s_test = generate_data(test_freqs, m, P)

    U_train = paddle.to_tensor(U_train).unsqueeze(-1)
    y_train = paddle.to_tensor(y_train).unsqueeze(-1)
    s_train = paddle.to_tensor(s_train).unsqueeze(-1)

    U_test = paddle.to_tensor(U_test).unsqueeze(-1)
    y_test = paddle.to_tensor(y_test).unsqueeze(-1)
    s_test = paddle.to_tensor(s_test).unsqueeze(-1)

    train_dataset = DataGenerator(U_train, y_train, s_train, batch_size)
    test_dataset = DataGenerator(U_test, y_test, s_test, batch_size)

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

    model.count_params()

    scheduler = paddle.optimizer.lr.StepDecay(
        learning_rate=0.001,
        step_size=100,
        gamma=0.99,
    )

    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler,
        parameters=model.parameters(),
    )

    start = timeit.default_timer()

    train_model(
        model,
        iter(train_dataset),
        iter(test_dataset),
        optimizer,
        scheduler,
        iterations,
    )

    elapsed = timeit.default_timer() - start

    print("Training time:", elapsed)

    os.makedirs("checkpoints", exist_ok=True)

    paddle.save(model.state_dict(), "checkpoints/nomad_antiderivative.pdparams")

    print("Model saved.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("n", type=int)
    parser.add_argument("decoder", type=str)
    parser.add_argument(
        "--iterations",
        type=int,
        default=20000,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=0,
        help="Random seed for synthetic training frequencies.",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=12345,
        help="Random seed for synthetic test frequencies.",
    )

    args = parser.parse_args()

    main(args.n, args.decoder, args.iterations, args.train_seed, args.test_seed)
