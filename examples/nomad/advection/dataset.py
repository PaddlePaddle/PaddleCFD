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

        idx = self.rng.choice(
            self.N,
            size=(self.batch_size,),
            replace=False,
        )

        idx = paddle.to_tensor(idx, dtype="int64")

        s = paddle.index_select(self.s, idx, axis=0)
        u = paddle.index_select(self.u, idx, axis=0)
        y = paddle.index_select(self.y, idx, axis=0)

        return (u, y), s