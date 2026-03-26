import fnmatch

import numpy as np
from utils.infer_higher_is_better import higher_is_better_from_metric_key

from .base.summary_summarizer_base import SummarySummarizerBase


class BestMetricSummarySummarizer(SummarySummarizerBase):
    def __init__(self, pattern, **kwargs):
        super().__init__(**kwargs)
        self.pattern = pattern

    def summarize(self):
        filtered_keys = [
            key
            for key in self.summary_provider.keys()
            if "/update" not in key and "/key" not in key
        ]
        matching_keys = []
        for key in filtered_keys:
            if "*" in self.pattern or "?" in self.pattern:
                if not fnmatch.fnmatch(key, self.pattern):
                    continue
            elif self.pattern not in key:
                continue
            if "/atbest/" in key:
                continue
            matching_keys.append(key)
        assert (
            len(matching_keys) > 0
        ), f"no matching_keys found for pattern '{self.pattern}'"
        values = [self.summary_provider[key] for key in matching_keys]
        higher_is_better = higher_is_better_from_metric_key(matching_keys[0])
        assert all(
            higher_is_better == higher_is_better_from_metric_key(key)
            for key in matching_keys[1:]
        )
        best_value = np.max(values) if higher_is_better else np.min(values)
        best_idxs = np.argwhere(values == best_value).squeeze(1)
        if len(best_idxs) > 1:
            self.logger.info(f"multiple best_idxs {best_idxs}")
        best_idx = best_idxs[0]
        best_key = matching_keys[best_idx]
        self.logger.info(
            f"pattern={self.pattern} best_key='{best_key}' best_value={best_value}"
        )
        self.summary_provider[f"{self.pattern}/best"] = float(best_value)
        self.summary_provider[f"{self.pattern}/best/key"] = best_key
