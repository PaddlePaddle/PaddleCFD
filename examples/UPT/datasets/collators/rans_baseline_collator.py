import einops
import paddle
from kappadata.collators import KDSingleCollator
from kappadata.wrappers import ModeWrapper


class RansBaselineCollator(KDSingleCollator):
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
        for i in range(len(batch)):
            item = ModeWrapper.get_item(
                mode=dataset_mode, batch=batch[i], item="query_pos"
            )
            query_lens.append(len(item))
            query_pos.append(item)
        collated_batch["query_pos"] = paddle.concat(query_pos)
        mesh_to_grid_edges = []
        mesh_offset = 0
        grid_offset = 0
        for i in range(len(batch)):
            idx = ModeWrapper.get_item(
                mode=dataset_mode, batch=batch[i], item="mesh_to_grid_edges"
            )
            idx[:, (0)] += grid_offset
            idx[:, (1)] += mesh_offset
            mesh_to_grid_edges.append(idx)
            mesh_offset += mesh_lens[i]
            grid_offset += grid_lens[i]
        collated_batch["mesh_to_grid_edges"] = paddle.concat(mesh_to_grid_edges)
        grid_to_query_edges = []
        mesh_offset = 0
        grid_offset = 0
        for i in range(len(batch)):
            idx = ModeWrapper.get_item(
                mode=dataset_mode, batch=batch[i], item="grid_to_query_edges"
            )
            idx[:, (0)] += mesh_offset
            idx[:, (1)] += grid_offset
            grid_to_query_edges.append(idx)
            mesh_offset += mesh_lens[i]
            grid_offset += grid_lens[i]
        collated_batch["grid_to_query_edges"] = paddle.concat(grid_to_query_edges)
        pressure = []
        for i in range(len(batch)):
            item = ModeWrapper.get_item(
                mode=dataset_mode, batch=batch[i], item="pressure"
            )
            assert len(item) == query_lens[i]
            pressure.append(item)
        collated_batch["pressure"] = paddle.concat(pressure).unsqueeze(1)
        batch_idx = paddle.empty(sum(query_lens), dtype=paddle.long)
        start = 0
        cur_batch_idx = 0
        for i in range(len(query_lens)):
            end = start + query_lens[i]
            batch_idx[start:end] = cur_batch_idx
            start = end
            cur_batch_idx += 1
        ctx["batch_idx"] = batch_idx
        query_batch_idx = paddle.empty(sum(query_lens), dtype=paddle.long)
        start = 0
        cur_query_batch_idx = 0
        for i in range(len(query_lens)):
            end = start + query_lens[i]
            query_batch_idx[start:end] = cur_query_batch_idx
            start = end
            cur_query_batch_idx += 1
        ctx["query_batch_idx"] = query_batch_idx
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
