from functools import cached_property
from datasets.collators.cfd_simformer_collator import CfdSimformerCollator
import kappamodules.utils.tensor_cache as tc
import paddle
from callbacks.online_callbacks.update_output_callback import \
    UpdateOutputCallback
from kappadata.wrappers import ModeWrapper
from losses import loss_fn_from_kwargs
from paddle_utils import *
from paddle_geometric.nn.pool import radius_graph
from paddle_scatter import segment_csr
from utils.checkpoint import Checkpoint
from utils.factory import create

from .base.sgd_trainer import SgdTrainer


class CfdSimformerTrainer(SgdTrainer):
    def __init__(
        self,
        loss_function,
        detach_reconstructions=False,
        reconstruct_from_target=False,
        reconstruct_prev_x_weight=0,
        reconstruct_dynamics_weight=0,
        radius_graph_r=None,
        radius_graph_max_num_neighbors=None,
        max_batch_size=None,
        mask_loss_start_checkpoint=None,
        mask_loss_threshold=None,
        **kwargs,
    ):
        disable_gradient_accumulation = max_batch_size is None
        super().__init__(
            max_batch_size=max_batch_size,
            disable_gradient_accumulation=disable_gradient_accumulation,
            **kwargs,
        )
        self.loss_function = create(
            loss_function, loss_fn_from_kwargs, update_counter=self.update_counter
        )
        self.detach_reconstructions = detach_reconstructions
        self.reconstruct_from_target = reconstruct_from_target
        self.reconstruct_prev_x_weight = reconstruct_prev_x_weight
        self.reconstruct_dynamics_weight = reconstruct_dynamics_weight
        self.radius_graph_r = radius_graph_r
        self.radius_graph_max_num_neighbors = radius_graph_max_num_neighbors
        self.mask_loss_start_checkpoint = create(mask_loss_start_checkpoint, Checkpoint)
        if self.mask_loss_start_checkpoint is not None:
            assert self.mask_loss_start_checkpoint.is_minimally_specified
            self.mask_loss_start_checkpoint = (
                self.mask_loss_start_checkpoint.to_fully_specified(
                    updates_per_epoch=self.update_counter.updates_per_epoch,
                    effective_batch_size=self.update_counter.effective_batch_size,
                )
            )
        self.mask_loss_threshold = mask_loss_threshold
        self.num_supernodes = None

    def get_trainer_callbacks(self, model=None):
        keys = ["degree/input"]
        patterns = ["loss_stats", "tensor_stats"]
        return [
            UpdateOutputCallback(
                keys=keys,
                patterns=patterns,
                every_n_updates=self.track_every_n_updates,
                every_n_samples=self.track_every_n_samples,
                **self.get_default_callback_kwargs(),
            ),
            UpdateOutputCallback(
                keys=keys,
                patterns=patterns,
                **self.get_default_callback_intervals(),
                **self.get_default_callback_kwargs(),
            ),
        ]

    @cached_property
    def input_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="x")
        assert isinstance(
            collator.collator, CfdSimformerCollator,
        )
        self.num_supernodes = collator.collator.num_supernodes
        input_shape = dataset.getshape_x()
        self.logger.info(f"input_shape: {input_shape}")
        if self.reconstruct_prev_x_weight > 0 or self.reconstruct_dynamics_weight > 0:
            assert dataset.couple_query_with_input
        elif self.end_checkpoint.is_zero:
            pass
        else:
            assert dataset.root_dataset.num_query_points is not None
        return input_shape

    @cached_property
    def output_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="x")
        assert isinstance(
            collator.collator, CfdSimformerCollator,
        )
        output_shape = dataset.getshape_target()
        self.logger.info(f"output_shape: {output_shape}")
        return output_shape

    @cached_property
    def dataset_mode(self):
        return "x mesh_pos query_pos mesh_edges geometry2d timestep velocity target"

    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(paddle.nn.Layer):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer

        def to_device(self, item, batch, dataset_mode):
            data = ModeWrapper.get_item(mode=dataset_mode, item=item, batch=batch)
            data = data.to(self.model.device, non_blocking=True)
            return data

        def prepare(self, batch, dataset_mode=None):
            dataset_mode = dataset_mode or self.trainer.dataset_mode
            batch, ctx = batch
            mesh_pos = self.to_device(
                item="mesh_pos", batch=batch, dataset_mode=dataset_mode
            )
            batch_idx = ctx["batch_idx"].to(self.model.device, non_blocking=True)
            data = dict(
                x=self.to_device(item="x", batch=batch, dataset_mode=dataset_mode),
                geometry2d=self.to_device(
                    item="geometry2d", batch=batch, dataset_mode=dataset_mode
                ),
                timestep=self.to_device(
                    item="timestep", batch=batch, dataset_mode=dataset_mode
                ),
                velocity=self.to_device(
                    item="velocity", batch=batch, dataset_mode=dataset_mode
                ),
                query_pos=self.to_device(
                    item="query_pos", batch=batch, dataset_mode=dataset_mode
                ),
                mesh_pos=mesh_pos,
                batch_idx=batch_idx,
                unbatch_idx=ctx["unbatch_idx"].to(self.model.device, non_blocking=True),
                unbatch_select=ctx["unbatch_select"].to(
                    self.model.device, non_blocking=True
                ),
                target=self.to_device(
                    item="target", batch=batch, dataset_mode=dataset_mode
                ),
            )
            mesh_edges = ModeWrapper.get_item(
                item="mesh_edges", batch=batch, mode=dataset_mode
            )
            if mesh_edges is None:
                assert self.trainer.radius_graph_r is not None
                assert self.trainer.radius_graph_max_num_neighbors is not None
                if self.trainer.num_supernodes is None:
                    flow = "source_to_target"
                    supernode_idxs = None
                else:
                    flow = "target_to_source"
                    supernode_idxs = ctx["supernode_idxs"].to(
                        self.model.device, non_blocking=True
                    )
                mesh_edges = radius_graph(
                    x=mesh_pos,
                    r=self.trainer.radius_graph_r,
                    max_num_neighbors=self.trainer.radius_graph_max_num_neighbors,
                    batch=batch_idx,
                    loop=True,
                    flow=flow,
                )
                if supernode_idxs is not None:
                    is_supernode_edge = paddle.isin(mesh_edges[0], supernode_idxs)
                    mesh_edges = mesh_edges[:, (is_supernode_edge)]
                mesh_edges = mesh_edges.T
            else:
                assert self.trainer.radius_graph_r is None
                assert self.trainer.radius_graph_max_num_neighbors is None
                assert self.trainer.num_supernodes is None
                mesh_edges = mesh_edges.to(self.model.device, non_blocking=True)
            data["mesh_edges"] = mesh_edges
            return data

        def forward(self, batch, reduction="mean"):
            data = self.prepare(batch=batch)
            x = data.pop("x")
            target = data.pop("target")
            batch_idx = data["batch_idx"]
            batch_size = batch_idx._max() + 1
            forward_kwargs = {}
            if self.trainer.reconstruct_from_target:
                forward_kwargs["target"] = target
            model_outputs = self.model(
                x,
                **data,
                **forward_kwargs,
                detach_reconstructions=self.trainer.detach_reconstructions,
                reconstruct_prev_x=self.trainer.reconstruct_prev_x_weight > 0,
                reconstruct_dynamics=self.trainer.reconstruct_dynamics_weight > 0,
            )
            infos = {}
            losses = {}
            x_hat_loss = self.trainer.loss_function(
                prediction=model_outputs["x_hat"], target=target, reduction="none"
            )
            infos.update(
                {
                    "loss_stats/x_hat/min": x_hat_loss._min(),
                    "loss_stats/x_hat/max": x_hat_loss._max(),
                    "loss_stats/x_hat/gt1": (x_hat_loss > 1).sum() / x_hat_loss.size,
                    "loss_stats/x_hat/eq0": (x_hat_loss == 0).sum() / x_hat_loss.size,
                }
            )
            if self.trainer.mask_loss_start_checkpoint is not None:
                if (
                    self.trainer.mask_loss_start_checkpoint
                    > self.trainer.update_counter.cur_checkpoint
                ):
                    x_hat_loss_mask = x_hat_loss > self.trainer.mask_loss_threshold
                    x_hat_loss = x_hat_loss[x_hat_loss_mask]
                    infos["loss_stats/x_hat/gt_loss_threshold"] = (
                        x_hat_loss_mask.sum() / x_hat_loss_mask.size
                    )
            if reduction == "mean":
                losses["x_hat"] = x_hat_loss.mean()
            elif reduction == "mean_per_sample":
                _, ctx = batch
                num_zero_pos = (data["query_pos"] == 0).sum()
                assert (
                    num_zero_pos == 0
                ), f"padded query_pos not supported {num_zero_pos}"
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
            else:
                raise NotImplementedError
            total_loss = losses["x_hat"]
            if self.trainer.reconstruct_prev_x_weight > 0:
                num_channels = model_outputs["prev_x_hat"].size(1)
                prev_x_hat_loss = self.trainer.loss_function(
                    prediction=model_outputs["prev_x_hat"],
                    target=x[:, -num_channels:],
                    reduction="none",
                )
                if reduction == "mean":
                    timestep = data["timestep"]
                    timestep_per_point = paddle.gather(timestep, dim=0, index=batch_idx)
                    prev_x_hat_loss = prev_x_hat_loss[timestep_per_point != 0]
                    if self.trainer.mask_loss_start_checkpoint is not None:
                        if (
                            self.trainer.mask_loss_start_checkpoint
                            > self.trainer.update_counter.cur_checkpoint
                        ):
                            prev_x_hat_loss_mask = (
                                prev_x_hat_loss > self.trainer.mask_loss_threshold
                            )
                            prev_x_hat_loss = prev_x_hat_loss[prev_x_hat_loss_mask]
                            infos["loss_stats/prev_x_hat/gt_loss_threshold"] = (
                                prev_x_hat_loss_mask.sum() / prev_x_hat_loss_mask.size
                            )
                    prev_x_hat_loss = prev_x_hat_loss.mean()
                elif reduction == "mean_per_sample":
                    raise NotImplementedError
                else:
                    raise NotImplementedError
                losses["prev_x_hat"] = prev_x_hat_loss
                total_loss = (
                    total_loss
                    + self.trainer.reconstruct_prev_x_weight * prev_x_hat_loss
                )
            if self.trainer.reconstruct_dynamics_weight > 0:
                dynamics_hat_loss = self.trainer.loss_function(
                    prediction=model_outputs["dynamics_hat"],
                    target=model_outputs["dynamics"],
                    reduction="none",
                )
                max_timestep = self.model.conditioner.num_total_timesteps - 1
                timestep = data["timestep"]
                if reduction == "mean":
                    dynamics_hat_mask = timestep != max_timestep
                    if dynamics_hat_mask.sum() > 0:
                        dynamics_hat_loss = dynamics_hat_loss[dynamics_hat_mask].mean()
                    else:
                        dynamics_hat_loss = tc.zeros(size=(1,), device=timestep.device)
                elif reduction == "mean_per_sample":
                    dynamics_hat_loss[timestep == max_timestep] = 0.0
                    dynamics_hat_loss = dynamics_hat_loss.flatten(start_dim=1).mean(
                        dim=1
                    )
                else:
                    raise NotImplementedError
                losses["dynamics_hat"] = dynamics_hat_loss
                total_loss = (
                    total_loss
                    + self.trainer.reconstruct_dynamics_weight * dynamics_hat_loss
                )
            infos.update({})
            if self.trainer.num_supernodes is None:
                infos["degree/input"] = len(data["mesh_edges"]) / len(x)
            else:
                infos["degree/input"] = len(data["mesh_edges"]) / (
                    self.trainer.num_supernodes * batch_size
                )
            return dict(total=total_loss, **losses), infos
