import einops
import numpy as np
import paddle

from kappadata.datasets.kd_wrapper import KDWrapper


class KDRandomClassWrapper(KDWrapper):
    def __init__(self, dataset, mode='random', mode_kwargs=None, num_classes=None, seed=0, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        if num_classes is None:
            class_shape = self.dataset.getshape('class')
            assert len(class_shape) == 1
            self._num_classes = class_shape[0]
        else:
            self._num_classes = num_classes
        self._seed = seed
        self._mode = mode
        self._mode_kwargs = mode_kwargs
        self._generate_classes()

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value
        self._generate_classes()

    @property
    def num_classes(self):
        return self._num_classes

    @num_classes.setter
    def num_classes(self, value):
        self._num_classes = value
        self._generate_classes()

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self._generate_classes()

    @property
    def class_names(self):
        raise NotImplementedError

    def _generate_classes(self):
        size = len(self.dataset)
        rng = np.random.default_rng(self._seed)
        if self._mode == 'random':
            gen_fn = self._random
        elif self._mode == 'randperm':
            gen_fn = self._randperm
        elif self._mode == 'gatherbug':
            gen_fn = self._gatherbug
        else:
            raise NotImplementedError
        kwargs = dict(num_classes=self._num_classes, size=size, rng=rng)
        self._classes = gen_fn(**kwargs, **(self._mode_kwargs or {})).tolist()

    @staticmethod
    def _random(num_classes, size, rng):
        return paddle.to_tensor(rng.integers(0, num_classes, size=size), dtype=paddle.int64)

    @staticmethod
    def _randperm(num_classes, size, rng):
        return KDRandomClassWrapper.__repeat(paddle.to_tensor(rng.permutation(num_classes), dtype=paddle.int64), size, num_classes)

    @staticmethod
    def _gatherbug(num_classes, size, world_size, **_):
        samples_per_class = (size + num_classes - 1) // num_classes
        classes = paddle.arange(num_classes).repeat_interleave(samples_per_class)[:size]
        num_padded_samples = (world_size - size % world_size) % world_size
        if num_padded_samples > 0:
            classes = paddle.concat([classes, classes[:num_padded_samples]])
        classes = einops.rearrange(classes, '(classes world_size) -> (world_size classes)', world_size=world_size)
        if num_padded_samples > 0:
            classes = classes[:size]
        return classes

    @staticmethod
    def __repeat(tensor, size, num_classes):
        repeats = (size + num_classes - 1) // num_classes
        return paddle.tile(tensor, repeat_times=[repeats])[:size]

    def getitem_class(self, idx, ctx=None):
        return self._classes[idx]

    def getall_class(self):
        return self._classes

    def getshape_class(self):
        return self._num_classes,
