import numpy as np
import paddle


def output_construction(Ux, t_his, cx, cy, P=1000, ds=3, Nx=32, Ny=32, Nt=100):

    U_all = np.zeros((P, ds), dtype=np.float32)
    Y_all = np.zeros((P, ds), dtype=np.float32)

    it = np.random.randint(Nt, size=P)
    x = np.random.randint(Nx, size=P)
    y = np.random.randint(Ny, size=P)

    T, X, Y = np.meshgrid(t_his, cx, cy, indexing="ij")

    Y_all[:, :] = np.concatenate(
        (
            T[it, x][np.arange(P), y][:, None],
            X[it, x][np.arange(P), y][:, None],
            Y[it, x][np.arange(P), y][:, None],
        ),
        axis=-1,
    )

    U_all[:, :] = Ux[it, x][np.arange(P), y]

    return U_all, Y_all


class DataGenerator:
    def __init__(self, u, y, s, batch_size=100, seed=1234):

        self.u = u
        self.y = y
        self.s = s

        self.N = u.shape[0]

        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def state_dict(self):
        return {"rng_state": self.rng.bit_generator.state}

    def set_state_dict(self, state_dict):
        self.rng.bit_generator.state = state_dict["rng_state"]

    def __next__(self):

        idx = self.rng.choice(self.N, size=(self.batch_size,), replace=False)

        idx = paddle.to_tensor(idx, dtype   ="int64")

        s = paddle.index_select(self.s, idx, axis=0)

        u = paddle.index_select(self.u, idx, axis=0)

        y = paddle.index_select(self.y, idx, axis=0)

        return (u, y), s
