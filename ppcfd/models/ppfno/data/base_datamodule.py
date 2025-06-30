import paddle
from paddle.io import DistributedBatchSampler
from paddle.distributed import get_rank

class BaseDataModule:

    @property
    def train_dataset(self) ->paddle.io.Dataset:
        raise NotImplementedError

    @property
    def val_dataset(self) ->paddle.io.Dataset:
        raise NotImplementedError

    @property
    def test_dataset(self) ->paddle.io.Dataset:
        raise NotImplementedError
    
    def inference_dataset(self) ->paddle.io.Dataset:
        raise NotImplementedError

    def train_dataloader(self, enable_ddp=False, **kwargs) -> paddle.io.DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        if enable_ddp is True:
            sampler = DistributedBatchSampler(self.train_data,
                                              rank=get_rank(),
                                              shuffle=True,
                                              batch_size=1)
            return paddle.io.DataLoader(self.train_data, batch_sampler=sampler, num_workers=0, collate_fn=collate_fn, **kwargs)
        else:
            return paddle.io.DataLoader(self.train_data, collate_fn=collate_fn, **kwargs)

    def val_dataloader(self, enable_ddp=False, **kwargs) -> paddle.io.DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        if enable_ddp is True:
            sampler = DistributedBatchSampler(self.val_data,
                                    rank=get_rank(),
                                    shuffle=True,
                                    batch_size=1)
            return paddle.io.DataLoader(self.val_data, batch_sampler=sampler, num_workers=0, collate_fn=collate_fn, **kwargs)
        else:
            return paddle.io.DataLoader(self.val_data, collate_fn=collate_fn, **kwargs)


    def test_dataloader(self, enable_ddp=False, **kwargs) -> paddle.io.DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        if enable_ddp is True:

            sampler = DistributedBatchSampler(self.test_data,
                                    rank=get_rank(),
                                    shuffle=False,
                                    batch_size=1)
            return paddle.io.DataLoader(self.test_data, batch_sampler=sampler, num_workers=0, collate_fn=collate_fn, **kwargs)
        else:
            return paddle.io.DataLoader(self.test_data, collate_fn=collate_fn, **kwargs)

    def inference_dataloader(self, enable_ddp=False, **kwargs) -> paddle.io.DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        if enable_ddp is True:

            sampler = DistributedBatchSampler(self.inference_data,
                                    rank=get_rank(),
                                    shuffle=False,
                                    batch_size=1)
            return paddle.io.DataLoader(self.inference_data, batch_sampler=sampler, num_workers=0, collate_fn=collate_fn, **kwargs)
        else:
            return paddle.io.DataLoader(self.inference_data, collate_fn=collate_fn, **kwargs)
