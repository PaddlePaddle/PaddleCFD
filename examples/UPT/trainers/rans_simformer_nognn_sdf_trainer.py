from functools import cached_property
from datasets.collators.rans_simformer_nognn_collator import RansSimformerNognnCollator
import paddle
from callbacks.online_callbacks.update_output_callback import UpdateOutputCallback
from kappadata.wrappers import ModeWrapper
from losses import loss_fn_from_kwargs
# from paddle_scatter import segment_csr
from paddle_scatter import segment_csr
from utils.factory import create

from .base.sgd_trainer import SgdTrainer


class RansSimformerNognnSdfTrainer(SgdTrainer):
    def __init__(self, loss_function, max_batch_size=None, **kwargs):
        disable_gradient_accumulation = max_batch_size is None
        super().__init__(
            max_batch_size=max_batch_size,
            disable_gradient_accumulation=disable_gradient_accumulation,
            **kwargs
        )
        self.loss_function = create(
            loss_function, loss_fn_from_kwargs, update_counter=self.update_counter
        )

    @cached_property
    def input_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="mesh_pos")
        assert isinstance(
            collator.collator, RansSimformerNognnCollator,
        )
        mesh_pos, _ = dataset[0]
        # assert mesh_pos.ndim == 2 and 2 <= mesh_pos.size(1) <= 3
        assert mesh_pos.ndim == 2 and 2 <= mesh_pos.shape[1] <= 3
        # return None, mesh_pos.size(1)
        return None, mesh_pos.shape[1]

    @cached_property
    def output_shape(self):
        return None, 1

    @cached_property
    def dataset_mode(self):
        return "pressure mesh_pos sdf query_pos"

    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(paddle.nn.Layer):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer

        def to_device(self, item, batch):
            # data = ModeWrapper.get_item(
            #     mode=self.trainer.dataset_mode, item=item, batch=batch
            # )
            data = ModeWrapper.get_item(
                mode=self.trainer.dataset_mode, item=item, batch=batch
            )
            data = data.to(self.model.device)
            return data

        def prepare(self, batch):
            
            batch, ctx = batch
            return dict(
                mesh_pos=self.to_device(item="mesh_pos", batch=batch),
                sdf=self.to_device(item="sdf", batch=batch),
                query_pos=self.to_device(item="query_pos", batch=batch),
                batch_idx=ctx["batch_idx"].to(self.model.device),
                unbatch_idx=ctx["unbatch_idx"].to(self.model.device),
                unbatch_select=ctx["unbatch_select"].to(
                    self.model.device
                ),
                target=self.to_device(item="pressure", batch=batch),
            )

        def forward(self, batch, reduction="mean"):
            data = self.prepare(batch)
            target = data.pop("target")
            model_outputs = self.model(**data)
            loss = self.trainer.loss_function(
                prediction=model_outputs["x_hat"], target=target, reduction=reduction
            )
            if reduction == "mean_per_sample":
                _, ctx = batch
                query_batch_idx = ctx["query_batch_idx"].to(
                    self.model.device
                )
                _, counts = query_batch_idx.unique(return_counts=True)
                indptr = paddle.concat([counts[:1] * 0, counts], axis=0).cumsum(
                    axis=0
                )
                loss = segment_csr(src=loss, indptr=indptr, reduce="mean")
            return dict(total=loss, x_hat=loss), {}
