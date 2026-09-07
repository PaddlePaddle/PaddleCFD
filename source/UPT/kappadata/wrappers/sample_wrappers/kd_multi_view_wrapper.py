from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.factory import object_to_transform
from kappadata.transforms import KDTransform, KDComposeTransform, KDStochasticTransform, KDIdentityTransform
from .x_transform_wrapper import XTransformWrapper


@dataclass
class KDMultiViewConfig:
    n_views: int
    transform: callable


class KDMultiViewWrapper(KDWrapper):
    def __init__(self, dataset, configs, seed=None):
        super().__init__(dataset=dataset)
        # if dataset is XTransformWrapper -> check that transform is deterministic (can only check KDTransforms)
        if isinstance(dataset, XTransformWrapper):
            assert dataset.transform.is_deterministic

        assert isinstance(configs, list)
        # copy to not alter the original list
        configs = deepcopy(configs)
        # parse configs
        for i in range(len(configs)):
            if not isinstance(configs[i], KDMultiViewConfig):
                if isinstance(configs[i], (list, tuple)):
                    # parse tuple/list/callable
                    # - KDMultiViewWrapper(configs=[(2, ToTensor())])
                    # - KDMultiViewWrapper(configs=[2])
                    # - KDMultiViewWrapper(configs=[ToTensor()])
                    if len(configs[i]) == 2 and isinstance(configs[i][0], int):
                        # KDMultiViewWrapper(configs=[(2, ToTensor())])
                        n_views, transform = configs[i]
                    elif len(configs[i]) == 1:
                        if isinstance(configs[i], int):
                            # KDMultiViewWrapper(configs=[2])
                            n_views = configs[i]
                            transform = None
                        else:
                            # KDMultiViewWrapper(configs=[ToTensor()])
                            n_views = 1
                            transform = configs[i]
                    else:
                        # KDMultiViewWrapper(configs=[[Resize, Norm], [Resize, Norm]])
                        n_views = 1
                        transform = configs[i]
                elif isinstance(configs[i], dict):
                    # parse from dict
                    # - KDMultiViewWrapper(configs=[dict(n_views=2, transform=ToTensor())])
                    # - KDMultiViewWrapper(configs=[dict(transform=ToTensor())])
                    # - KDMultiViewWrapper(configs=[dict(n_views=2)])
                    assert "n_views" in configs[i] or "transform" in configs[i]
                    n_views = configs[i].get("n_views", 1)
                    transform = configs[i].get("transform", None)
                else:
                    # parse transform/n_views only
                    # - KDMultiViewWrapper(configs=[ToTensor()])
                    # - KDMultiViewWrapper(configs=[1])
                    if isinstance(configs[i], int):
                        n_views = configs[i]
                        transform = None
                    else:
                        n_views = 1
                        transform = configs[i]
                # allow transform to be passed as dict
                if isinstance(transform, (list, tuple, dict)):
                    transform = object_to_transform(transform)
                if transform is None:
                    transform = KDIdentityTransform()
                assert isinstance(n_views, int)
                assert callable(transform)
                configs[i] = KDMultiViewConfig(n_views=n_views, transform=transform)

        self.transform_configs = configs
        self.n_views = sum(config.n_views for config in configs)
        self.seed = seed

    def getitem_x(self, idx, ctx=None):
        sample = self.dataset.getitem_x(idx)
        x = []
        i = 0
        for config in self.transform_configs:
            # set rng of transforms
            if self.seed is not None:
                rng = np.random.default_rng(seed=self.seed + idx)
                if isinstance(config.transform, (KDComposeTransform, KDStochasticTransform)):
                    config.transform.set_rng(rng)
            # sample views
            for _ in range(config.n_views):
                view_ctx = {}
                if isinstance(config.transform, KDTransform):
                    view = config.transform(sample, ctx=view_ctx)
                else:
                    view = config.transform(sample)
                x.append(view)
                if ctx is not None:
                    ctx[f"view{i}"] = view_ctx
                    i += 1
        return x

    def _worker_init_fn(self, rank, **kwargs):
        for config in self.transform_configs:
            if isinstance(config.transform, KDTransform):
                config.transform.worker_init_fn(rank, **kwargs)
