import numpy as np


class RandomSampler:
    def __init__(self, data_source, replacement=False, num_samples=None, generator=None, num_repeats=1):
        assert num_repeats >= 1
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples or len(data_source)
        self.generator = generator
        self.num_repeats = num_repeats

    @property
    def effective_length(self):
        return self.num_samples

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = self.generator or np.random.default_rng()
        n = len(self.data_source)
        if self.replacement:
            idxs = rng.integers(0, n, size=self.num_samples)
        else:
            idxs = rng.permutation(n)
        if self.num_repeats > 1:
            idxs = np.repeat(idxs, self.num_repeats)[:self.num_samples]
        yield from idxs.tolist()
