import numpy as np
import paddle


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

    def __next__(self):

        idx = self.rng.choice(self.N, size=(self.batch_size,), replace=False)

        idx = paddle.to_tensor(idx, dtype="int64")

        s = paddle.index_select(self.s, idx, axis=0)
        u = paddle.index_select(self.u, idx, axis=0)
        y = paddle.index_select(self.y, idx, axis=0)

        return (u, y), s


def interpolate_1d(x_query, x_grid, values):

    return np.interp(x_query, x_grid, values)


def generate_one_datum(freq, m=500, P=500):

    X = np.linspace(0, 1, m).astype(np.float32)

    u = np.cos(2 * np.pi * freq * X).astype(np.float32)

    y_train = np.linspace(0, 1, P).astype(np.float32)

    u_on_y = interpolate_1d(y_train, X, u)

    s_train = np.zeros_like(y_train)

    for i in range(1, P):

        dt = y_train[i] - y_train[i - 1]

        s_train[i] = s_train[i - 1] + 0.5 * (u_on_y[i] + u_on_y[i - 1]) * dt

    return u, y_train, s_train


def generate_data(freqs, m, P):

    u_list = []
    y_list = []
    s_list = []

    for f in freqs:

        u, y, s = generate_one_datum(f, m, P)

        u_list.append(u)
        y_list.append(y)
        s_list.append(s)

    return (
        np.stack(u_list).astype(np.float32),
        np.stack(y_list).astype(np.float32),
        np.stack(s_list).astype(np.float32),
    )