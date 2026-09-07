from __future__ import annotations

import random
from collections.abc import Iterable

import paddle


class IterDataPipe(paddle.io.IterableDataset):
    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise TypeError(f"{type(self).__name__} is an iterable dataset")

    def shuffle(self, buffer_size: int, seed: int | None = None):
        return ShuffledIterDataPipe(self, buffer_size=buffer_size, seed=seed)

    def cycle(self):
        return CycledIterDataPipe(self)


class ShuffledIterDataPipe(IterDataPipe):
    def __init__(self, datapipe: Iterable, buffer_size: int, seed: int | None = None):
        super().__init__()
        self.datapipe = datapipe
        self.buffer_size = buffer_size
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        iterator = iter(self.datapipe)
        buffer = []
        try:
            for _ in range(self.buffer_size):
                buffer.append(next(iterator))
        except StopIteration:
            pass
        while buffer:
            index = rng.randrange(len(buffer))
            yield buffer.pop(index)
            try:
                buffer.append(next(iterator))
            except StopIteration:
                continue


class CycledIterDataPipe(IterDataPipe):
    def __init__(self, datapipe: Iterable):
        super().__init__()
        self.datapipe = datapipe

    def __iter__(self):
        while True:
            yielded = False
            for item in self.datapipe:
                yielded = True
                yield item
            if not yielded:
                return


class Multiplexer(IterDataPipe):
    def __init__(self, *datapipes: Iterable):
        super().__init__()
        self.datapipes = datapipes

    def __iter__(self):
        iterators = [iter(datapipe) for datapipe in self.datapipes]
        while iterators:
            next_iterators = []
            for iterator in iterators:
                try:
                    yield next(iterator)
                    next_iterators.append(iterator)
                except StopIteration:
                    continue
            iterators = next_iterators


class SampleMultiplexer(IterDataPipe):
    def __init__(self, datapipes_to_weights: dict[Iterable, float], seed: int | None = None):
        super().__init__()
        self.datapipes_to_weights = datapipes_to_weights
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        active = [
            [iter(datapipe), float(weight)]
            for datapipe, weight in self.datapipes_to_weights.items()
        ]
        while active:
            positive = [item for item in active if item[1] > 0]
            if not positive:
                return
            population = list(range(len(positive)))
            weights = [item[1] for item in positive]
            selected = positive[rng.choices(population, weights=weights, k=1)[0]][0]
            try:
                yield next(selected)
            except StopIteration:
                active = [item for item in active if item[0] is not selected]
