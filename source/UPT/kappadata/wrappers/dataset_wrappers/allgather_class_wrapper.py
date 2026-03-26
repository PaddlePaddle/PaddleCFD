import einops
import paddle

from kappadata.datasets.kd_wrapper import KDWrapper


class AllgatherClassWrapper(KDWrapper):
    """ permute classes as if they were all_gathered by world_size GPUs """

    def __init__(self, dataset, world_size):
        super().__init__(dataset=dataset)
        indices = paddle.arange(len(dataset))
        num_padded_samples = (world_size - len(indices) % world_size) % world_size

        # pad to multiple of world size
        if num_padded_samples > 0:
            indices = paddle.concat([indices, indices[:num_padded_samples]])
        # rearrange
        indices = einops.rearrange(
            tensor=indices,
            pattern="(samples_per_gpu world_size) -> (world_size samples_per_gpu)",
            world_size=world_size
        )
        # cut away padding
        if num_padded_samples > 0:
            indices = indices[:-num_padded_samples]
        self.indices = indices.tolist()

    def getitem_class(self, idx, ctx=None):
        return self.dataset.getitem_class(self.indices[idx], ctx=ctx)

    def getall_class(self):
        return [self.getitem_class(self.indices[idx]) for idx in range(len(self))]