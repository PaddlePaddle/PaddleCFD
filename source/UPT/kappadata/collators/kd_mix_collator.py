import paddle

from kappadata.collators.base.kd_single_collator import KDSingleCollator
from kappadata.error_messages import REQUIRES_MIXUP_P_OR_CUTMIX_P
from kappadata.wrappers.mode_wrapper import ModeWrapper


class KDMixCollator(KDSingleCollator):
    def __init__(self, mixup_alpha=None, cutmix_alpha=None, mixup_p=None, cutmix_p=None, apply_mode='batch', lamb_mode='batch', shuffle_mode='flip', **kwargs):
        super().__init__(**kwargs)
        assert (mixup_p is not None) or (cutmix_p is not None), REQUIRES_MIXUP_P_OR_CUTMIX_P
        mixup_p = mixup_p or 0.
        cutmix_p = cutmix_p or 0.
        assert isinstance(mixup_p, (int, float)) and 0. <= mixup_p <= 1.
        assert isinstance(cutmix_p, (int, float)) and 0. <= cutmix_p <= 1.
        assert 0. < mixup_p + cutmix_p <= 1.
        if mixup_p + cutmix_p != 1.:
            raise NotImplementedError
        if mixup_p == 0.:
            assert mixup_alpha is None
        else:
            assert isinstance(mixup_alpha, (int, float)) and 0. < mixup_alpha
        if cutmix_p == 0.:
            assert cutmix_alpha is None
        else:
            assert isinstance(cutmix_alpha, (int, float)) and 0. < cutmix_alpha
        assert apply_mode in ['batch', 'sample']
        assert lamb_mode in ['batch', 'sample']
        assert shuffle_mode in ['roll', 'flip', 'random']
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_p = mixup_p
        self.cutmix_p = cutmix_p
        self.apply_mode = apply_mode
        self.lamb_mode = lamb_mode
        self.shuffle_mode = shuffle_mode

    @property
    def default_collate_mode(self):
        return 'before'

    @property
    def total_p(self):
        return self.mixup_p + self.cutmix_p

    def collate(self, batch, dataset_mode, ctx=None):
        idx, x, y = None, None, None
        is_binary_classification = False
        if ModeWrapper.has_item(mode=dataset_mode, item='index'):
            idx = ModeWrapper.get_item(mode=dataset_mode, item='index', batch=batch)
        if ModeWrapper.has_item(mode=dataset_mode, item='x'):
            x = ModeWrapper.get_item(mode=dataset_mode, item='x', batch=batch)
        if ModeWrapper.has_item(mode=dataset_mode, item='class'):
            y = ModeWrapper.get_item(mode=dataset_mode, item='class', batch=batch).astype(paddle.float32)
            if y.ndim != 2:
                assert y.ndim == 1 and 0. <= float(y.min()) and float(y.max()) <= 1.
                y = y.unsqueeze(1)
                is_binary_classification = True
        batch_size = len(x)

        if self.apply_mode == 'batch':
            apply = paddle.full(shape=[batch_size], fill_value=self.rng.random() < self.total_p, dtype=paddle.bool)
        elif self.apply_mode == 'sample':
            apply = paddle.to_tensor(self.rng.random(batch_size)) < self.total_p
        else:
            raise NotImplementedError

        permutation = None
        if self.lamb_mode == 'batch':
            use_cutmix = self.rng.random() * self.total_p < self.cutmix_p
            alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
            lamb = paddle.to_tensor([self.rng.beta(alpha, alpha)], dtype=paddle.float32)
            if x is not None:
                x2, permutation = self.shuffle(item=x, permutation=permutation)
                if use_cutmix:
                    h, w = x.shape[2:]
                    bbox, lamb = self.get_random_bbox(h=h, w=w, lamb=lamb)
                    top, left, bot, right = [int(v) for v in bbox[0].tolist()]
                    x[..., top:bot, left:right] = x2[..., top:bot, left:right]
                else:
                    x_lamb = lamb.reshape([-1] + [1] * (x.ndim - 1))
                    x = x * x_lamb + x2 * (1. - x_lamb)
            if y is not None:
                y2, permutation = self.shuffle(item=y, permutation=permutation)
                y_lamb = lamb.reshape([-1, 1])
                y = y * y_lamb + y2 * (1. - y_lamb)
        elif self.lamb_mode == 'sample':
            use_cutmix = paddle.to_tensor(self.rng.random(batch_size) * self.total_p) < self.cutmix_p
            if self.mixup_p > 0.:
                mixup_lamb = paddle.to_tensor(self.rng.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size), dtype=paddle.float32)
            else:
                mixup_lamb = paddle.zeros([batch_size], dtype=paddle.float32)
            if self.cutmix_p > 0.:
                cutmix_lamb = paddle.to_tensor(self.rng.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size), dtype=paddle.float32)
                h, w = x.shape[2:]
                bbox, cutmix_lamb = self.get_random_bbox(h=h, w=w, lamb=cutmix_lamb)
            else:
                cutmix_lamb = paddle.zeros([batch_size], dtype=paddle.float32)
                bbox = None
            lamb = paddle.where(use_cutmix, cutmix_lamb, mixup_lamb)
            if x is not None:
                x2_indices, permutation = self.shuffle(item=paddle.arange(batch_size), permutation=permutation)
                x_clone = paddle.clone(x)
                bbox_idx = 0
                for i in range(batch_size):
                    j = int(x2_indices[i])
                    if bool(use_cutmix[j]):
                        top, left, bot, right = [int(v) for v in bbox[bbox_idx].tolist()]
                        x[i, ..., top:bot, left:right] = x_clone[j, ..., top:bot, left:right]
                        bbox_idx += 1
                    else:
                        x_lamb = lamb[i].reshape([1] * (x.ndim - 1))
                        x[i] = x[i] * x_lamb + x_clone[j] * (1 - x_lamb)
            if y is not None:
                y2, permutation = self.shuffle(item=y, permutation=permutation)
                y_lamb = lamb.reshape([-1, 1])
                y = y * y_lamb + y2 * (1. - y_lamb)
        else:
            raise NotImplementedError

        if ctx is not None:
            ctx['apply'] = apply
            ctx['use_cutmix'] = use_cutmix
            ctx['lambda'] = lamb
        if idx is not None:
            batch = ModeWrapper.set_item(mode=dataset_mode, item='index', batch=batch, value=idx)
        if x is not None:
            batch = ModeWrapper.set_item(mode=dataset_mode, item='x', batch=batch, value=x)
        if y is not None:
            if is_binary_classification:
                y = y.squeeze(1)
            batch = ModeWrapper.set_item(mode=dataset_mode, item='class', batch=batch, value=y)
        return batch

    def get_random_bbox(self, h, w, lamb):
        n_bboxes = len(lamb)
        bbox_hcenter = paddle.to_tensor(self.rng.integers(h, size=(n_bboxes,)), dtype=paddle.float32)
        bbox_wcenter = paddle.to_tensor(self.rng.integers(w, size=(n_bboxes,)), dtype=paddle.float32)
        area_half = paddle.sqrt(1.0 - lamb) * 0.5
        bbox_h_half = paddle.floor(area_half * h)
        bbox_w_half = paddle.floor(area_half * w)
        top = paddle.clip(bbox_hcenter - bbox_h_half, min=0).astype(paddle.int64)
        bot = paddle.clip(bbox_hcenter + bbox_h_half, max=h).astype(paddle.int64)
        left = paddle.clip(bbox_wcenter - bbox_w_half, min=0).astype(paddle.int64)
        right = paddle.clip(bbox_wcenter + bbox_w_half, max=w).astype(paddle.int64)
        bbox = paddle.stack([top, left, bot, right], axis=1)
        lamb_adjusted = 1.0 - (bot - top) * (right - left) / (h * w)
        return bbox, lamb_adjusted.astype(paddle.float32)

    def shuffle(self, item, permutation):
        if len(item) == 1:
            return paddle.clone(item), None
        if self.shuffle_mode == 'roll':
            return paddle.roll(item, shifts=1, axis=0), None
        if self.shuffle_mode == 'flip':
            assert len(item) % 2 == 0
            return paddle.flip(item, axis=[0]), None
        if self.shuffle_mode == 'random':
            if permutation is None:
                permutation = self.rng.permutation(len(item))
            return item[permutation], permutation
        raise NotImplementedError
