import argparse
import os
import numpy as np
import paddle
from tqdm import trange

from dataset import DataGenerator, output_construction
from ppcfd.models.nomad.shallowwater.nomad_model import OperatorModel


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


def train(
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

    model.train()

    pbar = trange(nIter)

    for it in pbar:

        train_batch = next(train_dataset)

        loss = model.loss(train_batch)

        loss.backward()

        optimizer.step()

        optimizer.clear_grad()

        scheduler.step()
        global_step += 1

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
    iterations=100000,
    checkpoint=None,
    seed=None,
    train_seed=1234,
    test_seed=1234,
    state_path=None,
    resume_state=False,
    state_interval=0,
):

    TRAINING_ITERATIONS = iterations

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
    s_test = np.zeros((num_test, P, ds), dtype=np.float32)
    y_test = np.zeros((num_test, P, dy), dtype=np.float32)

    for i in range(num_train):

        s_train[i], y_train[i] = output_construction(
            S_train[i], T, CX, CY, P=P, ds=ds, Nx=Nx, Ny=Ny, Nt=Nt
        )

    for i in range(num_test):

        s_test[i], y_test[i] = output_construction(
            S_test[i], T, CX, CY, P=P, ds=ds, Nx=Nx, Ny=Ny, Nt=Nt
        )

    U_train = paddle.to_tensor(U_train)
    y_train = paddle.to_tensor(y_train)
    s_train = paddle.to_tensor(s_train)
    U_test = paddle.to_tensor(U_test)
    y_test = paddle.to_tensor(y_test)
    s_test = paddle.to_tensor(s_test)

    train_dataset = DataGenerator(U_train, y_train, s_train, batch_size, train_seed)
    test_dataset = DataGenerator(U_test, y_test, s_test, batch_size, test_seed)

    if decoder == "nonlinear":

        branch_layers = [m * du, 100, 100, 100, 100, 100, ds * n]

        trunk_layers = [ds * n * 2, 100, 100, 100, 100, 100, ds]

    else:

        branch_layers = [m * du, 100, 100, 100, 100, 100, ds * n]

        trunk_layers = [dy, 100, 100, 100, 100, 100, ds * n]

    if seed is not None:
        np.random.seed(seed)
        paddle.seed(seed)

    model = OperatorModel(branch_layers, trunk_layers, n=n, decoder=decoder, ds=ds)

    scheduler = paddle.optimizer.lr.StepDecay(0.001, step_size=100, gamma=0.99)

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

    global_step = train(
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

    os.makedirs("checkpoints", exist_ok=True)

    if checkpoint is None:
        checkpoint = f"checkpoints/nomad_sw_{decoder}_n{n}.pdparams"

    paddle.save(model.state_dict(), checkpoint)
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

    print("Model saved:", checkpoint)
    print("Global step:", global_step)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("n", type=int, nargs="+")
    parser.add_argument("decoder", type=str, nargs="+")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100000,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to save model parameters.",
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
        args.n[0],
        args.decoder[0],
        args.iterations,
        args.checkpoint,
        args.seed,
        args.train_seed,
        args.test_seed,
        args.state_path,
        args.resume_state,
        args.state_interval,
    )
