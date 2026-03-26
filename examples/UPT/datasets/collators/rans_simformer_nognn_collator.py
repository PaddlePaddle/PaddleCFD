import paddle
from kappadata.collators import KDSingleCollator
from kappadata.wrappers import ModeWrapper
import paddle.nn.functional as F
from paddle.io.dataloader.collate import default_collate_fn


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

class RansSimformerNognnCollator(KDSingleCollator):
    def collate(self, batch, dataset_mode, ctx=None):
        assert isinstance(batch, (tuple, list)) and isinstance(batch[0], tuple)
        batch, ctx = zip(*batch)
        ctx = {}
        collated_batch = {}
        mesh_pos = []
        mesh_lens = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(
                mode=dataset_mode, batch=batch[i], item="mesh_pos"
            )
            mesh_lens.append(len(item))
            mesh_pos.append(item)
        collated_batch["mesh_pos"] = paddle.concat(mesh_pos)
        pressures = [
            ModeWrapper.get_item(mode=dataset_mode, batch=sample, item="pressure")
            for sample in batch
        ]
        query_pos = []
        query_lens = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(
                mode=dataset_mode, batch=batch[i], item="query_pos"
            )
            assert len(item) == len(pressures[i])
            query_lens.append(len(item))
            query_pos.append(item)
        collated_batch["query_pos"] = paddle_pad_sequence(
            query_pos, batch_first=True
        )
        collated_batch["pressure"] = paddle.concat(pressures).unsqueeze(1)
        batch_size = len(mesh_lens)
        batch_idx = paddle.empty([sum(mesh_lens)], dtype=paddle.int64)
        start = 0
        cur_batch_idx = 0
        for i in range(len(mesh_lens)):
            end = start + mesh_lens[i]
            batch_idx[start:end] = cur_batch_idx
            start = end
            cur_batch_idx += 1
        ctx["batch_idx"] = batch_idx
        query_batch_idx = paddle.empty([sum(query_lens)], dtype=paddle.int64)
        start = 0
        cur_query_batch_idx = 0
        for i in range(len(query_lens)):
            end = start + query_lens[i]
            query_batch_idx[start:end] = cur_query_batch_idx
            start = end
            cur_query_batch_idx += 1
        ctx["query_batch_idx"] = query_batch_idx
        maxlen = max(query_lens)
        unbatch_idx = paddle.empty([maxlen * batch_size], dtype=paddle.int64)
        unbatch_select = []
        unbatch_start = 0
        cur_unbatch_idx = 0
        for i in range(len(query_lens)):
            unbatch_end = unbatch_start + query_lens[i]
            unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
            unbatch_select.append(cur_unbatch_idx)
            cur_unbatch_idx += 1
            unbatch_start = unbatch_end
            padding = maxlen - query_lens[i]
            if padding > 0:
                unbatch_end = unbatch_start + padding
                unbatch_idx[unbatch_start:unbatch_end] = cur_unbatch_idx
                cur_unbatch_idx += 1
                unbatch_start = unbatch_end
        unbatch_select = paddle.to_tensor(unbatch_select)
        ctx["unbatch_idx"] = unbatch_idx
        ctx["unbatch_select"] = unbatch_select
        result = []
        for item in dataset_mode.split(" "):
            if item in collated_batch:
                result.append(collated_batch[item])
            else:
                result.append(
                    default_collate_fn(
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
