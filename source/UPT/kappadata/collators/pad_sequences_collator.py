import paddle
from kappadata.compat.sequence import pad_sequence
from kappadata.compat.data import default_collate

from .base.kd_single_collator import KDSingleCollator


class PadSequencesCollator(KDSingleCollator):
    @property
    def default_collate_mode(self):
        return None

    def collate(self, batch, _, ctx=None):
        if isinstance(batch[0], tuple):
            if isinstance(batch[0][0], tuple) and isinstance(batch[0][1], dict):
                # return_ctx=True
                data = [b[0] for b in batch]
                contexts = [b[1] for b in batch]
                return self.collate(data, _, ctx), self.collate(contexts, _, ctx)
            result = []
            for i in range(len(batch[0])):
                first_item = batch[0][i]
                items = [b[i] for b in batch]
                if paddle.is_tensor(first_item) and first_item.ndim > 0:
                    result.append(pad_sequence([b[i] for b in batch], batch_first=True))
                else:
                    result.append(default_collate(items))
            return tuple(result)
        elif isinstance(batch[0], dict):
            return default_collate(batch)
        return pad_sequence(batch, batch_first=True)
