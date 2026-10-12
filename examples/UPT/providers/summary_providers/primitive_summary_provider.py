import numpy as np
import yaml
from utils.infer_higher_is_better import (higher_is_better_from_metric_key,
                                          is_neutral_key)

from ..path_provider import PathProvider
from .base.summary_provider_base import SummaryProviderBase


class PrimitiveSummaryProvider(SummaryProviderBase):
    def __init__(self, path_provider: PathProvider):
        super().__init__()
        self.path_provider = path_provider
        self.summary = {}

    def update(self, *args, **kwargs):
        self.summary.update(*args, **kwargs)

    def __setitem__(self, key, value):
        self.summary[key] = value

    def __getitem__(self, key):
        return self.summary[key]

    def __contains__(self, key):
        return key in self.summary

    def keys(self):
        return self.summary.keys()

    def get_summary_of_previous_stage(self, stage_name, stage_id):
        summary_uri = self.path_provider.get_primitive_summary_uri(
            stage_name=stage_name, stage_id=stage_id
        )
        if not summary_uri.exists():
            return None
        with open(summary_uri) as f:
            return yaml.safe_load(f)

    def flush(self):
        """summary is potentially often updated -> flush in bulks"""
        with open(self.path_provider.primitive_summary_uri, "w") as f:
            yaml.safe_dump(self.summary, f)

    def summarize_logvalues(self):
        entries_uri = self.path_provider.primitive_entries_uri
        if not entries_uri.exists():
            return None
        with open(entries_uri) as f:
            entries = yaml.safe_load(f)
        if entries is None:
            return None
        summary = {}
        for key, update_to_value in entries.items():
            if key[0] == "_":
                continue
            last_update = max(update_to_value.keys())
            last_value = update_to_value[last_update]
            self[key] = last_value
            if key in ["epoch", "update", "sample"]:
                continue
            if is_neutral_key(key):
                continue
            values = list(update_to_value.values())
            higher_is_better = higher_is_better_from_metric_key(key)
            if higher_is_better:
                minmax_key = f"{key}/max"
                minmax_value = max(values)
            else:
                minmax_key = f"{key}/min"
                minmax_value = min(values)
            self[minmax_key] = minmax_value
            summary[minmax_key] = minmax_value
            self.logger.info(f"{minmax_key}: {minmax_value}")
            for running_avg_count in [10, 50]:
                running_avg = float(np.mean(values[-running_avg_count:]))
                running_avg_key = f"{key}/last{running_avg_count}"
                self[running_avg_key] = running_avg
                summary[running_avg_key] = running_avg
            last_key = f"{key}/last"
            last_value = values[-1]
            self[last_key] = last_value
            summary[last_key] = last_value
        return summary
