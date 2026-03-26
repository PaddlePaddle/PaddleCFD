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

class CfdSimformerCollator(KDSingleCollator):
    def __init__(self, num_supernodes=None, **kwargs):
        super().__init__(**kwargs)
        self.num_supernodes = num_supernodes

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
        if self.num_supernodes is not None:
            supernodes_offset = 0
            supernode_idxs = []
            for i in range(len(mesh_lens)):
                perm = (
                    paddle.randperm(len(mesh_pos[i]))[: self.num_supernodes]
                    + supernodes_offset
                )
                supernode_idxs.append(perm)
                supernodes_offset += mesh_lens[i]
            ctx["supernode_idxs"] = paddle.concat(supernode_idxs)
        batch_idx = paddle.empty(sum(mesh_lens), dtype=paddle.long)
        start = 0
        cur_batch_idx = 0
        for i in range(len(mesh_lens)):
            end = start + mesh_lens[i]
            batch_idx[start:end] = cur_batch_idx
            start = end
            cur_batch_idx += 1
        ctx["batch_idx"] = batch_idx
        x = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(mode=dataset_mode, batch=batch[i], item="x")
            assert len(item) == mesh_lens[i]
            x.append(item)
        collated_batch["x"] = paddle.concat(x)
        grid_pos = []
        grid_lens = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(
                mode=dataset_mode, batch=batch[i], item="mesh_pos"
            )
            grid_lens.append(len(item))
            grid_pos.append(item)
        collated_batch["grid_pos"] = paddle.concat(grid_pos)
        batch_idx = paddle.empty(sum(mesh_lens), dtype=paddle.long)
        start = 0
        cur_batch_idx = 0
        for i in range(len(mesh_lens)):
            end = start + mesh_lens[i]
            batch_idx[start:end] = cur_batch_idx
            start = end
            cur_batch_idx += 1
        ctx["batch_idx"] = batch_idx
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
        collated_batch["query_pos"] = paddle_pad_sequence(
            query_pos, batch_first=True
        )
        collated_batch["target"] = paddle.concat(target)
        batch_size = len(query_lens)
        maxlen = max(query_lens)
        unbatch_idx = paddle.empty(maxlen * batch_size, dtype=paddle.long)
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
        unbatch_select = paddle.tensor(unbatch_select)
        ctx["unbatch_idx"] = unbatch_idx
        ctx["unbatch_select"] = unbatch_select
        mesh_edges = []
        mesh_edges_offset = 0
        for i in range(len(batch)):
            item = ModeWrapper.get_item(
                mode=dataset_mode, batch=batch[i], item="mesh_edges"
            )
            if item is None:
                break
            idx = item + mesh_edges_offset
            mesh_edges.append(idx)
            mesh_edges_offset += mesh_lens[i]
        if len(mesh_edges) > 0:
            collated_batch["mesh_edges"] = paddle.concat(mesh_edges)
        else:
            collated_batch["mesh_edges"] = None
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
