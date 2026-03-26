from pathlib import Path

import einops
import numpy as np
import paddle

from kappadata.datasets.kd_wrapper import KDWrapper
from kappadata.utils.global_rng import GlobalRng


class KDPseudoLabelWrapper(KDWrapper):
    def __init__(
            self,
            dataset,
            uri=None,
            pseudo_labels=None,
            threshold=None,
            topk=None,
            tau=None,
            seed=None,
            **kwargs,
    ):
        super().__init__(dataset=dataset, **kwargs)
        assert len(self.dataset.getshape_class()) == 1

        # load pseudo labels
        assert (uri is None) ^ (pseudo_labels is None), "uri and pseudo_labels argument are mutually exclusive"
        if uri is not None:
            if not isinstance(uri, Path):
                uri = Path(uri)
            uri = uri.expanduser()
            assert uri.exists(), f"'{uri.as_posix()}' does not exist"
            data = paddle.load(uri, map_location="cpu")
            if paddle.is_tensor(data):
                pseudo_labels = data
                confidences = None
            else:
                # uri is file with dict(labels=labels, confidences=confidences)
                assert isinstance(data, dict) and "label" in data and "confidence" in data
                pseudo_labels = data["label"]
                confidences = data["confidence"].float()
        elif pseudo_labels is not None:
            confidences = None
        else:
            raise NotImplementedError
        # check pseudo labels
        assert len(pseudo_labels) == len(self.dataset)
        assert pseudo_labels.ndim == 1 or pseudo_labels.ndim == 2
        if pseudo_labels.ndim == 2 and pseudo_labels.size(1) == 1:
            pseudo_labels = pseudo_labels.squeeze()
        if pseudo_labels.ndim == 2:
            assert pseudo_labels.size(1) == self.dataset.getshape_class()[0]
        if confidences is not None:
            assert len(confidences) == len(self.dataset)
            assert confidences.ndim == 1
            assert confidences.shape == pseudo_labels.shape

        # set properties
        self.pseudo_labels = pseudo_labels
        self.confidences = confidences
        self.threshold = threshold
        self.topk = topk
        self.tau = tau
        self.seed = seed

    @property
    def _global_rng(self):
        return GlobalRng()

    # noinspection PyUnusedLocal
    def getitem_class(self, idx, ctx=None):
        item = self._getitem_class(idx)
        if paddle.is_tensor(item):
            item = item.long().item()
        return item

    # noinspection PyUnusedLocal
    def getitem_confidence(self, idx, ctx=None):
        assert self.topk is None and self.tau is None, "geitem_confidence requires static pseudo labels"
        assert self.threshold is None
        assert self.confidences is not None
        confidence = self.confidences[idx]
        return confidence

    def _getitem_class(self, idx):
        if self.seed is not None:
            # static pseudo labels (they dont change from epoch to epoch)
            rng = np.random.default_rng(seed=self.seed + idx)
        else:
            # dynamic pseudo labels (resampled for every epoch)
            rng = self._global_rng

        # sample pseudo labels
        if self.topk is not None:
            assert self.threshold is None, "threshold with sampled pseudo labels is not supported"
            # sample from topk
            assert self.pseudo_labels.ndim == 2
            topk_probs, topk_idxs = self.pseudo_labels[idx].topk(k=self.topk)
            if self.tau == float("inf"):
                # uniform sample
                choice = rng.integers(self.topk)
            else:
                if self.tau is not None:
                    # labels are logits -> divide by temperature and apply softmax
                    weights = (topk_probs / self.tau).softmax(dim=0)
                else:
                    # labels are probabilities
                    weights = topk_probs
                # NOTE: argmax is required because np.random.multinomial "counts" the number of outcomes
                # so if multinomial of 5 values draws the 4th value and 1 trial is used the outcome would
                # be [0, 0, 0, 1, 0] -> with argmax -> 4
                choice = rng.multinomial(1, weights).argmax()
            return topk_idxs[choice]

        # hard labels
        assert self.tau is None
        if self.pseudo_labels.ndim == 1:
            assert self.threshold is None, "provided pseudo labels have no probabilities -> can't apply threshold"
            return self.pseudo_labels[idx]
        elif self.pseudo_labels.ndim == 2:
            if self.threshold is None:
                return self.pseudo_labels[idx].argmax()
            else:
                pseudo_label_probs = self.pseudo_labels[idx].softmax(dim=0)
                argmax = pseudo_label_probs.argmax()
                if pseudo_label_probs[argmax] > self.threshold:
                    return argmax
                return -1
        else:
            raise NotImplementedError

    def getall_class(self):
        if self.tau is not None or self.topk is not None:
            raise NotImplementedError
        if self.pseudo_labels.ndim == 1:
            return self.pseudo_labels.tolist()
        if self.pseudo_labels.ndim == 2:
            return self.pseudo_labels.argmax(dim=1).tolist()
        raise NotImplementedError
