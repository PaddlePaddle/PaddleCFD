import math
from multiprocessing import Value

import numpy as np
import paddle

from kappadata.compat.data import default_collate
from kappadata.utils.param_checking import to_2tuple
from kappadata.wrappers import ModeWrapper
from .base import KDSingleCollator


class KDIjepaMaskCollator(KDSingleCollator):
    """KappaData adaption of the I-JEPA masking collator."""

    def __init__(self, input_size=(224, 224), patch_size=16, encoder_mask_scale=(0.85, 1.0), predictor_mask_scale=(0.15, 0.2), predictor_aspect_ratio=(0.75, 1.5), num_enc_masks=1, num_pred_masks=4, min_keep=10, tries=20, **kwargs):
        super().__init__(**kwargs)
        self.input_size = to_2tuple(input_size)
        self.patch_size = to_2tuple(patch_size)
        self.seqlen_h = self.input_size[0] // self.patch_size[0]
        self.seqlen_w = self.input_size[1] // self.patch_size[1]
        self.encoder_mask_scale = to_2tuple(encoder_mask_scale)
        self.predictor_mask_scale = to_2tuple(predictor_mask_scale)
        self.predictor_aspect_ratio = to_2tuple(predictor_aspect_ratio)
        self.num_enc_masks = num_enc_masks
        self.num_pred_masks = num_pred_masks
        self.min_keep = min_keep
        self.tries = tries
        self._itr_counter = Value('i', -1)

    @property
    def default_collate_mode(self):
        return 'before'

    def collate(self, batch, dataset_mode, ctx=None):
        if ctx is None:
            return batch
        x = ModeWrapper.get_item(mode=dataset_mode, item='x', batch=batch)
        batch_size = len(x)

        seed = self.step()
        rng = np.random.default_rng(seed)
        predictor_size = self._sample_block_size(rng=rng, scale=self.predictor_mask_scale, aspect_ratio_range=self.predictor_aspect_ratio)
        encoder_size = self._sample_block_size(rng=rng, scale=self.encoder_mask_scale, aspect_ratio_range=(1., 1.))

        predictor_masks, encoder_masks = [], []
        min_keep_pred = min_keep_enc = self.seqlen_h * self.seqlen_w
        for _ in range(batch_size):
            pred_masks, pred_masks_complement = [], []
            for _ in range(self.num_pred_masks):
                mask, mask_complement = self._sample_block_mask(predictor_size)
                pred_masks.append(mask)
                pred_masks_complement.append(mask_complement)
                min_keep_pred = min(min_keep_pred, len(mask))
            predictor_masks.append(pred_masks)

            enc_masks = []
            for _ in range(self.num_enc_masks):
                mask = self._sample_block_mask_constrained(encoder_size, acceptable_regions=pred_masks_complement)
                enc_masks.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            encoder_masks.append(enc_masks)

        predictor_masks = [[mask[:min_keep_pred] for mask in masks] for masks in predictor_masks]
        predictor_masks = default_collate(predictor_masks)
        encoder_masks = [[mask[:min_keep_enc] for mask in masks] for masks in encoder_masks]
        encoder_masks = default_collate(encoder_masks)

        ctx['encoder_masks'] = paddle.concat(encoder_masks)
        ctx['predictor_masks'] = paddle.concat(predictor_masks)
        return batch

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            value = i.value
        return value

    def _sample_block_size(self, rng, scale, aspect_ratio_range):
        rand = float(rng.random())
        min_scale, max_scale = scale
        mask_scale = min_scale + rand * (max_scale - min_scale)
        max_keep = int(self.seqlen_h * self.seqlen_w * mask_scale)
        min_ar, max_ar = aspect_ratio_range
        aspect_ratio = min_ar + rand * (max_ar - min_ar)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        h = min(h, self.seqlen_h - 1)
        w = min(w, self.seqlen_w - 1)
        return h, w

    def _sample_block_mask(self, block_size):
        block_h, block_w = block_size
        top = self.rng.integers(0, self.seqlen_h - block_h)
        left = self.rng.integers(0, self.seqlen_w - block_w)
        bot = top + block_h
        right = left + block_w
        mask = paddle.zeros((self.seqlen_h, self.seqlen_w), dtype=paddle.int32)
        mask[top:bot, left:right] = 1
        mask = mask.flatten().nonzero().squeeze()
        mask_complement = paddle.ones((self.seqlen_h, self.seqlen_w), dtype=paddle.int32)
        mask_complement[top:bot, left:right] = 0
        return mask, mask_complement

    def _sample_block_mask_constrained(self, block_size, acceptable_regions):
        block_h, block_w = block_size
        tries = 0
        while True:
            top = self.rng.integers(0, self.seqlen_h - block_h)
            left = self.rng.integers(0, self.seqlen_w - block_w)
            bot = top + block_h
            right = left + block_w
            mask = paddle.zeros((self.seqlen_h, self.seqlen_w), dtype=paddle.int32)
            mask[top:bot, left:right] = 1
            for k in range(max(int(len(acceptable_regions) - tries // self.tries), 0)):
                mask *= acceptable_regions[k]
            mask = mask.flatten().nonzero()
            if len(mask) > self.min_keep:
                break
            tries += 1
        return mask.squeeze()
