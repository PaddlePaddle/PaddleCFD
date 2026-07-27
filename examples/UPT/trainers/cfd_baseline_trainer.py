from functools import cached_property
from datasets.collators.cfd_baseline_collator import CfdBaselineCollator
import einops
import paddle
from callbacks.online_callbacks.update_output_callback import \
    UpdateOutputCallback
from kappadata.wrappers import ModeWrapper
from losses import loss_fn_from_kwargs
from paddle_utils import *
from paddle_geometric.nn.pool import radius
from paddle_scatter import segment_csr
from utils.factory import create

from .base.sgd_trainer import SgdTrainer


class CfdBaselineTrainer(SgdTrainer):
    def __init__(
        self,
        loss_function,
        radius_graph_r=None,
        radius_graph_max_num_neighbors=None,
        max_batch_size=None,
        **kwargs,
    ):
        disable_gradient_accumulation = max_batch_size is None
        super().__init__(
            max_batch_size=max_batch_size,
            disable_gradient_accumulation=disable_gradient_accumulation,
            **kwargs,
        )
        self.radius_graph_r = radius_graph_r
        self.radius_graph_max_num_neighbors = radius_graph_max_num_neighbors
        self.loss_function = create(
            loss_function, loss_fn_from_kwargs, update_counter=self.update_counter
        )

    def get_trainer_callbacks(self, model=None):
        return [
            UpdateOutputCallback(
                keys=["degree/input", "degree/output"],
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
            UpdateOutputCallback(
                keys=["degree/input", "degree/output"],
                **self.get_default_callback_intervals(),
                **self.get_default_callback_kwargs(),
            ),
        ]

    @cached_property
    def input_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="x")
        assert dataset.root_dataset.num_query_points is not None
        assert isinstance(
            collator.collator, CfdBaselineCollator,
        )
        input_shape = dataset.getshape_x()
        self.logger.info(f"input_shape: {input_shape}")
        return input_shape

    @cached_property
    def output_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="target")
        assert isinstance(
            collator.collator, CfdBaselineCollator,
        )
        output_shape = dataset.getshape_target()
        self.logger.info(f"output_shape: {output_shape}")
        return output_shape

    @cached_property
    def dataset_mode(self):
        return "x mesh_pos grid_pos query_pos mesh_to_grid_edges grid_to_query_edges timestep velocity target"

    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(paddle.nn.Layer):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer

        def to_device(self, item, batch, dataset_mode):
            data = ModeWrapper.get_item(mode=dataset_mode, item=item, batch=batch)
            data = data.to(self.model.device)
            return data

        def prepare(self, batch, dataset_mode=None):
            dataset_mode = dataset_mode or self.trainer.dataset_mode
            batch, ctx = batch
            batch_idx = ctx["batch_idx"].to(self.model.device)
            data = dict(
                x=self.to_device(item="x", batch=batch, dataset_mode=dataset_mode),
                mesh_pos=self.to_device(
                    item="mesh_pos", batch=batch, dataset_mode=dataset_mode
                ),
                grid_pos=self.to_device(
                    item="grid_pos", batch=batch, dataset_mode=dataset_mode
                ),
                query_pos=self.to_device(
                    item="query_pos", batch=batch, dataset_mode=dataset_mode
                ),
                timestep=self.to_device(
                    item="timestep", batch=batch, dataset_mode=dataset_mode
                ),
                velocity=self.to_device(
                    item="velocity", batch=batch, dataset_mode=dataset_mode
                ),
                target=self.to_device(
                    item="target", batch=batch, dataset_mode=dataset_mode
                ),
                batch_idx=batch_idx,
            )
            mesh_to_grid_edges = ModeWrapper.get_item(
                item="mesh_to_grid_edges", batch=batch, mode=dataset_mode
            )
            grid_to_query_edges = ModeWrapper.get_item(
                item="grid_to_query_edges", batch=batch, mode=dataset_mode
            )
            batch_size = len(data["timestep"])
            if mesh_to_grid_edges is None or grid_to_query_edges is None:
                assert len(data["grid_pos"]) % batch_size == 0
                num_grid_points = len(data["grid_pos"]) // batch_size
                grid_batch_idx = paddle.arange(
                    batch_size, device=self.model.device
                ).repeat_interleave(num_grid_points)
            else:
                grid_batch_idx = None
            if mesh_to_grid_edges is None:
                assert self.trainer.radius_graph_r is not None
                assert self.trainer.radius_graph_max_num_neighbors is not None
                mesh_to_grid_edges = radius(
                    x=data["mesh_pos"],
                    y=data["grid_pos"],
                    batch_x=batch_idx,
                    batch_y=grid_batch_idx,
                    r=self.trainer.radius_graph_r,
                    max_num_neighbors=self.trainer.radius_graph_max_num_neighbors,
                ).T
            else:
                assert self.trainer.radius_graph_r is None
                assert self.trainer.radius_graph_max_num_neighbors is None
                mesh_to_grid_edges = mesh_to_grid_edges.to(
                    self.model.device
                )
            data["mesh_to_grid_edges"] = mesh_to_grid_edges
            if grid_to_query_edges is None:
                assert self.trainer.radius_graph_r is not None
                assert self.trainer.radius_graph_max_num_neighbors is not None
                assert len(data["query_pos"]) % batch_size == 0
                num_query_pos = len(data["query_pos"]) // batch_size
                query_batch_idx = paddle.arange(
                    batch_size, device=self.model.device
                ).repeat_interleave(num_query_pos)
                grid_to_query_edges = radius(
                    x=data["grid_pos"],
                    y=data["query_pos"],
                    batch_x=grid_batch_idx,
                    batch_y=query_batch_idx,
                    r=self.trainer.radius_graph_r,
                    max_num_neighbors=self.trainer.radius_graph_max_num_neighbors,
                ).T
            else:
                assert self.trainer.radius_graph_r is None
                assert self.trainer.radius_graph_max_num_neighbors is None
                grid_to_query_edges = grid_to_query_edges.to(
                    self.model.device
                )
            data["grid_to_query_edges"] = grid_to_query_edges
            return data

        def forward(self, batch, reduction="mean"):
            data = self.prepare(batch)
            target = data.pop("target")
            model_outputs = self.model(**data)
            x_hat_loss = self.trainer.loss_function(
                prediction=model_outputs["x_hat"], target=target, reduction="none"
            )
            losses = {}
            if reduction == "mean":
                losses["x_hat"] = x_hat_loss.mean()
            if reduction == "mean_per_sample":
                batch_index = data["batch_idx"]
                batch_size = batch_index._max() + 1
                query_pos_len = data["query_pos"].size(1)
                query_batch_idx = paddle.arange(
                    batch_size, device=self.model.device
                ).repeat_interleave(query_pos_len)
                indices, counts = query_batch_idx.unique(return_counts=True)
                padded_counts = paddle.zeros(
                    len(indices) + 1, device=counts.device, dtype=counts.dtype
                )
                padded_counts[indices + 1] = counts
                indptr = padded_counts.cumsum(dim=0)
                losses["x_hat"] = segment_csr(
                    src=x_hat_loss.mean(dim=1), indptr=indptr, reduce="mean"
                )
            infos = {
                "degree/input": len(data["mesh_to_grid_edges"]) / len(data["grid_pos"]),
                "degree/output": len(data["grid_to_query_edges"]) / len(target),
            }
            return dict(total=losses["x_hat"], **losses), infos
