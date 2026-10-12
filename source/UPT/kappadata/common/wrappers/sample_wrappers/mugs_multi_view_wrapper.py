from kappadata.common.transforms.byol_transforms import BYOLTransform0, BYOLTransform1
from kappadata.common.transforms.mugs_transforms import MUGSStrongGlobalTransform, MUGSStrongLocalTransform
from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.transforms.base.kd_compose_transform import KDComposeTransform
from kappadata.transforms.base.kd_stochastic_transform import KDStochasticTransform
from kappadata.transforms.base.kd_transform import KDTransform
from kappadata.utils.global_rng import GlobalRng
from kappadata.wrappers.sample_wrappers.x_transform_wrapper import XTransformWrapper


class MUGSMultiViewWrapper(KDWrapper):
    def __init__(
            self,
            dataset,
            global_size=224,
            local_size=96,
            min_scale=0.05,
            mid_scale=0.25,
            num_local_crops=10,
            seed=None,
            **kwargs,
    ):
        super().__init__(dataset=dataset, **kwargs)
        # if dataset is XTransformWrapper -> check that transform is deterministic (can only check KDTransforms)
        if isinstance(dataset, XTransformWrapper):
            assert dataset.transform.is_deterministic
        self.seed = seed
        self.num_local_crops = num_local_crops

        # teacher views (same as byol)
        self.teacher_transform0 = BYOLTransform0(min_scale=mid_scale)
        self.teacher_transform1 = BYOLTransform1(min_scale=mid_scale)
        # student strong augmentations (weak augmentation is the same as teacher)
        self.student_strong_transform = MUGSStrongGlobalTransform(size=global_size, min_scale=mid_scale)
        # local weak transform (same as byol)
        self.local_weak_transform = BYOLTransform0(size=local_size, min_scale=min_scale, max_scale=mid_scale)
        # local strong transform (same as global strong but smaller)
        self.local_strong_transform = MUGSStrongLocalTransform(
            size=local_size,
            min_scale=min_scale,
            max_scale=mid_scale,
        )

        # compose to list for easy rng setting
        self.transforms = [
            self.teacher_transform0,
            self.teacher_transform1,
            self.student_strong_transform,
            self.local_weak_transform,
            self.local_strong_transform,
        ]

    def getitem_x(self, idx, ctx=None):
        # get base sample
        sample = self.dataset.getitem_x(idx)

        # set rng of transforms
        if self.seed is not None:
            rng = np.random.default_rng(seed=self.seed + idx)
            for transform in self.transforms:
                if isinstance(transform, (KDComposeTransform, KDStochasticTransform)):
                    transform.set_rng(rng)
        else:
            rng = GlobalRng()

        # teacher global views
        x = [
            self.teacher_transform0(sample),
            self.teacher_transform1(sample),
        ]
        # student global views
        is_weak_global_aug = rng.random() < 0.5
        if ctx is not None:
            ctx["is_weak_global_aug"] = is_weak_global_aug
        if is_weak_global_aug:
            x += [
                self.teacher_transform0(sample),
                self.teacher_transform1(sample),
            ]
        else:
            x += [
                self.student_strong_transform(sample),
                self.student_strong_transform(sample),
            ]

        # local
        for _ in range(self.num_local_crops):
            cur_weak_local_aug = rng.random() < 0.5
            if cur_weak_local_aug:
                x.append(self.local_weak_transform(sample))
            else:
                x.append(self.local_strong_transform(sample))

        return x

    def _worker_init_fn(self, rank, **kwargs):
        for transform in self.transforms:
            if isinstance(transform, KDTransform):
                transform.worker_init_fn(rank, **kwargs)
