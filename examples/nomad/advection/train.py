import argparse
import os
import timeit

import numpy as np
import paddle
from tqdm import trange

from dataset import DataGenerator
from utils import load_model, save_model

from ppcfd.models.nomad.advection.operator_model import OperatorModel


def save_training_state(
    path,
    model,
    optimizer,
    scheduler,
    train_dataset,
    test_dataset,
    global_step,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    paddle.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_dataset": train_dataset.state_dict(),
            "test_dataset": test_dataset.state_dict(),
            "global_step": global_step,
        },
        path,
    )
    print("Training state saved to:", path)


def load_training_state(
    path,
    model,
    optimizer,
    scheduler,
    train_dataset,
    test_dataset,
):
    state = paddle.load(path)
    model.set_state_dict(state["model"])
    optimizer.set_state_dict(state["optimizer"])
    scheduler.set_state_dict(state["scheduler"])
    train_dataset.set_state_dict(state["train_dataset"])
    test_dataset.set_state_dict(state["test_dataset"])
    print("Training state loaded:", path)
    return int(state.get("global_step", 0))


def train_model(
    model,
    train_dataset,
    test_dataset,
    optimizer,
    scheduler,
    nIter,
    state_path=None,
    state_interval=0,
    global_step=0,
):

    pbar = trange(nIter)

    for it in pbar:

        train_batch = next(train_dataset)
        test_batch = next(test_dataset)

        loss_train = model.loss(train_batch)

        loss_train.backward()

        optimizer.step()
        optimizer.clear_grad()
        scheduler.step()
        global_step += 1

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

        if state_path and state_interval > 0 and global_step % state_interval == 0:
            save_training_state(
                state_path,
                model,
                optimizer,
                scheduler,
                train_dataset,
                test_dataset,
                global_step,
            )

    return global_step


def main(
    n,
    decoder,
    iterations=20000,
    batch_size=100,
    learning_rate=0.001,
    decay_step=100,
    decay_rate=0.99,
    checkpoint="checkpoints/nomad_advection_model.pdparams",
    resume=False,
    seed=None,
    train_seed=1234,
    test_seed=1234,
    state_path=None,
    resume_state=False,
    state_interval=0,
):

    TRAINING_ITERATIONS = iterations

    P = 25600
    m = 256

    num_train = 1000
    num_test = 1000

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

    train_dataset = DataGenerator(U_train, y_train, s_train, batch_size, train_seed)
    test_dataset = DataGenerator(U_test, y_test, s_test, batch_size, test_seed)

    if decoder == "nonlinear":

        branch_layers = [m, 100, 100, 100, 100, 100, ds * n]

        trunk_layers = [ds * n * 2, 100, 100, 100, 100, 100, ds]

    else:

        branch_layers = [m, 100, 100, 100, 100, 100, ds * n]

        trunk_layers = [dy, 100, 100, 100, 100, 100, ds * n]

    if seed is not None:
        np.random.seed(seed)
        paddle.seed(seed)

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
        learning_rate=learning_rate,
        step_size=decay_step,
        gamma=decay_rate,
    )

    optimizer = paddle.optimizer.Adam(
        learning_rate=scheduler,
        parameters=model.parameters(),
    )

    global_step = 0
    if resume_state:
        if state_path is None:
            raise ValueError("--resume-state requires --state-path.")
        global_step = load_training_state(
            state_path,
            model,
            optimizer,
            scheduler,
            train_dataset,
            test_dataset,
        )
    elif resume:
        load_model(model, checkpoint)

    start = timeit.default_timer()

    global_step = train_model(
        model,
        iter(train_dataset),
        iter(test_dataset),
        optimizer,
        scheduler,
        TRAINING_ITERATIONS,
        state_path,
        state_interval,
        global_step,
    )

    print("Training time:", timeit.default_timer() - start)
    print("Global step:", global_step)

    save_model(
        model,
        checkpoint,
    )
    if state_path:
        save_training_state(
            state_path,
            model,
            optimizer,
            scheduler,
            train_dataset,
            test_dataset,
            global_step,
        )


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
        "--batch-size",
        type=int,
        default=100,
        help="Number of training samples per optimization step.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--decay-step",
        type=int,
        default=100,
        help="Step interval for exponential learning-rate decay.",
    )
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.99,
        help="Learning-rate decay factor.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/nomad_advection_model.pdparams",
        help="Path to save or load model parameters.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load the checkpoint before training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for Paddle and NumPy model initialization.",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=1234,
        help="Seed for training batch sampling.",
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=1234,
        help="Seed for test batch sampling during training.",
    )
    parser.add_argument(
        "--state-path",
        type=str,
        default=None,
        help="Optional path for full training state checkpoint.",
    )
    parser.add_argument(
        "--resume-state",
        action="store_true",
        help="Resume model, optimizer, scheduler, RNGs, and global step.",
    )
    parser.add_argument(
        "--state-interval",
        type=int,
        default=0,
        help="Save full training state every N global steps; 0 disables periodic saves.",
    )

    args = parser.parse_args()

    main(
        args.n,
        args.decoder,
        args.iterations,
        args.batch_size,
        args.learning_rate,
        args.decay_step,
        args.decay_rate,
        args.checkpoint,
        args.resume,
        args.seed,
        args.train_seed,
        args.test_seed,
        args.state_path,
        args.resume_state,
        args.state_interval,
    )
