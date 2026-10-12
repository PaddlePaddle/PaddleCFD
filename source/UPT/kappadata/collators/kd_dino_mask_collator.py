import math

import paddle

from kappadata.utils.param_checking import to_2tuple
from kappadata.wrappers import ModeWrapper
from .base import KDSingleCollator


class KDDinoMaskCollator(KDSingleCollator):
    def __init__(
            self,
            mask_ratio,
            mask_prob,
            mask_size,
            num_views=2,
            min_num_patches=4,
            min_aspect=0.3,
            max_aspect=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask_ratio = to_2tuple(mask_ratio)
        self.mask_prob = mask_prob
        self.num_views = num_views
        self.height, self.width = to_2tuple(mask_size)
        self.num_patches = self.height * self.width
        self.min_num_patches = min_num_patches
        self.log_aspect_max = math.log(max_aspect or 1 / min_aspect)
        self.log_aspect_min = math.log(min_aspect)

    @property
    def default_collate_mode(self):
        return "before"

    def collate(self, batch, dataset_mode, ctx=None):
        if ctx is None:
            return batch
        x = ModeWrapper.get_item(mode=dataset_mode, item="x", batch=batch)
        if isinstance(x, list):
            batch_size = len(x[0])
        else:
            batch_size = len(x)
        # apply mask to only a subset of the full batch
        num_masked_samples = int(batch_size * self.num_views * self.mask_prob)
        # actual mask ratios are sampled within "bin-ranges"
        # i think this was done to have an approximately equal number of masked patches per batch
        # example:
        # mask_ratio=(0.1, 0.5) batch_size=8 mask_prob=0.5 -> num_masked_samples=4
        # probs = [uniform(0.1, 0.2), uniform(0.2, 0.3), uniform(0.3, 0.4), uniform(0.4, 0.5)]
        mask_ratio_min, mask_ratio_max = self.mask_ratio
        probs = paddle.linspace(mask_ratio_min, mask_ratio_max, num_masked_samples + 1)

        masks = [paddle.zeros(self.height, self.width, dtype=paddle.bool) for _ in range(batch_size * self.num_views)]
        for i in range(num_masked_samples):
            num_masked_patches_total = int(self.rng.uniform(probs[i], probs[i + 1]) * self.num_patches)
            self._generate_mask(masks[i], num_masked_patches_total)
        self.rng.shuffle(masks)
        mask = paddle.stack(masks)
        ctx["mask"] = mask
        return batch

    def _generate_mask(self, mask, num_masked_patches_total):
        num_masked_patches = 0
        while num_masked_patches < num_masked_patches_total:
            num_masked_patches_remaining = num_masked_patches_total - num_masked_patches
            delta = self._mask_block(mask, num_masked_patches_remaining)
            if delta == 0:
                break
            else:
                num_masked_patches += delta

    def _mask_block(self, mask, num_remaining_patches_to_mask):
        delta = 0
        for _ in range(10):
            # sample height and width of block
            # NOTE: DINOv2 uses random.uniform(self.min_num_patches, num_remaining_patches_to_mask) which works
            # when num_remaining_patches_to_mask > self.min_num_patches -> not sure if this is intended
            area_min = min(self.min_num_patches, num_remaining_patches_to_mask)
            area_max = max(self.min_num_patches, num_remaining_patches_to_mask)
            target_area = self.rng.uniform(area_min, area_max)
            aspect_ratio = math.exp(self.rng.uniform(self.log_aspect_min, self.log_aspect_max))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            # resample if block is out of bounds
            if w >= self.width or h >= self.height:
                continue

            # sample location of block
            top = self.rng.integers(0, self.height - h + 1)
            left = self.rng.integers(0, self.width - w + 1)
            bot = top + h
            right = left + w

            # resample if block is already fully masked out
            num_unmasked_patches_in_block = h * w - mask[top:bot, left:right].sum()
            if num_unmasked_patches_in_block == 0:
                continue

            # resample if masking out the block would result in more masked patches than defined
            if num_unmasked_patches_in_block > num_remaining_patches_to_mask:
                continue

            # update mask
            for i in range(top, bot):
                for j in range(left, right):
                    if mask[i, j] == 0:
                        mask[i, j] = 1
                        delta += 1

            if delta > 0:
                break
        return delta
