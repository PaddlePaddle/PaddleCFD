import einops
import paddle
from kappadata.collators import KDSingleCollator
from kappadata.wrappers import ModeWrapper


class CfdBaselineCollator(KDSingleCollator):
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
                mode=dataset_mode, batch=batch[i], item="grid_pos"
            )
            grid_lens.append(len(item))
            grid_pos.append(item)
        collated_batch["grid_pos"] = paddle.concat(grid_pos)
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
        collated_batch["query_pos"] = paddle.concat(query_pos)
        collated_batch["target"] = paddle.concat(target)
        mesh_to_grid_edges = []
        mesh_offset = 0
        grid_offset = 0
        for i in range(len(batch)):
            idx = ModeWrapper.get_item(
                mode=dataset_mode, batch=batch[i], item="mesh_to_grid_edges"
            )
            if idx is None:
                break
            idx[:, (0)] += grid_offset
            idx[:, (1)] += mesh_offset
            mesh_to_grid_edges.append(idx)
            mesh_offset += mesh_lens[i]
            grid_offset += grid_lens[i]
        if len(mesh_to_grid_edges) > 0:
            collated_batch["mesh_to_grid_edges"] = paddle.concat(mesh_to_grid_edges)
        else:
            collated_batch["mesh_to_grid_edges"] = None
        grid_to_query_edges = []
        query_offset = 0
        grid_offset = 0
        for i in range(len(batch)):
            idx = ModeWrapper.get_item(
                mode=dataset_mode, batch=batch[i], item="grid_to_query_edges"
            )
            if idx is None:
                break
            idx[:, (0)] += query_offset
            idx[:, (1)] += grid_offset
            grid_to_query_edges.append(idx)
            query_offset += query_lens[i]
            grid_offset += grid_lens[i]
        if len(grid_to_query_edges) > 0:
            collated_batch["grid_to_query_edges"] = paddle.concat(grid_to_query_edges)
        else:
            collated_batch["grid_to_query_edges"] = None
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
