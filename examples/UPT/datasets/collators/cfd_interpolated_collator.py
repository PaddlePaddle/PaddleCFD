import einops
import paddle
from kappadata.collators import KDSingleCollator
from kappadata.wrappers import ModeWrapper
import paddle.nn.functional as F



def paddle_pad_sequence(sequences, batch_first=True, padding_value=0.0):
    # 1. 确定最大长度
    max_len = max([s.shape[0] for s in sequences])
    
    # 2. 对每个样本进行填充并堆叠
    padded_seqs = []
    for s in sequences:
        # Paddle 的 F.pad 接收的 pad 参数格式为 [padding_start, padding_end] 从最后一个维度开始
        # 对于形状为 [L, D] 的 query_pos，我们要在 L 维度末尾补齐
        # 填充列表格式：[倒数第一维开头, 倒数第一维结束, 倒数第二维开头, 倒数第二维结束...]
        # 这里倒数第一维（D）不补，倒数第二维（L）末尾补 (max_len - s.shape[0])
        pad_size = [0, 0, 0, max_len - s.shape[0]] 
        padded_s = F.pad(s, pad=pad_size, value=padding_value)
        padded_seqs.append(padded_s)
    
    # 3. 堆叠成一个 Batch
    return paddle.stack(padded_seqs, axis=0 if batch_first else 1)

class CfdInterpolatedCollator(KDSingleCollator):
    def collate(self, batch, dataset_mode, ctx=None):
        assert isinstance(batch, (tuple, list)) and isinstance(batch[0], tuple)
        batch, ctx = zip(*batch)
        ctx = {}
        collated_batch = {}
        query_pos = []
        query_lens = []
        target = []
        for i in range(len(batch)):
            query_pos_item = ModeWrapper.get_item(
                mode=dataset_mode, batch=batch[i], item="query_pos"
            )
            target_item = ModeWrapper.get_item(
                mode=dataset_mode, batch=batch[i], item="target"
            )
            assert len(query_pos_item) == len(target_item)
            query_lens.append(len(query_pos_item))
            query_pos.append(query_pos_item)
            target.append(target_item)
        assert all(query_lens[0] == query_len for query_len in query_lens[1:])
        collated_batch["query_pos"] = paddle_pad_sequence(
            query_pos, batch_first=True
        )
        collated_batch["target"] = paddle.concat(target)
        result = []
        for item in dataset_mode.split(" "):
            if item in collated_batch:
                result.append(collated_batch[item])
            else:
                result.append(
                    paddle.io.dataloader.collate.default_collate_fn(
                        [
                            ModeWrapper.get_item(
                                mode=dataset_mode, batch=sample, item=item
                            )
                            for sample in batch
                        ]
                    )
                )
        return tuple(result), ctx

    @property
    def default_collate_mode(self):
        raise RuntimeError

    def __call__(self, batch):
        raise NotImplementedError("wrap KDSingleCollator with KDSingleCollatorWrapper")
