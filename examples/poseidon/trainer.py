"""
Single-card trainer for Paddle scOT.
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import paddle

from ppcfd.models.poseidon.model import ConditionalLayerNorm, LayerNorm, ScOTOutput


@dataclass
class TrainingArguments:
    output_dir: str = "."
    overwrite_output_dir: bool = True
    evaluation_strategy: str = "no"
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    eval_accumulation_steps: int = 16
    max_grad_norm: float = 0.0
    num_train_epochs: int = 1
    optim: str = "adamw"
    learning_rate: float = 5e-5
    learning_rate_embedding_recovery: Optional[float] = field(default=None)
    learning_rate_time_embedding: Optional[float] = field(default=None)
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0
    log_level: str = "passive"
    logging_strategy: str = "steps"
    logging_steps: int = 5
    logging_nan_inf_filter: bool = False
    save_strategy: str = "epoch"
    save_total_limit: int = 1
    seed: int = 0
    fp16: bool = False
    dataloader_num_workers: int = 0
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "loss"
    greater_is_better: bool = False
    dataloader_pin_memory: bool = False
    gradient_checkpointing: bool = False
    auto_find_batch_size: bool = False
    full_determinism: bool = False
    torch_compile: bool = False
    report_to: Optional[str] = None
    run_name: Optional[str] = None
    past_index: int = -1
    early_stopping_patience: Optional[int] = None
    disable_tqdm: bool = False
    debug_batch_order_path: Optional[str] = None


@dataclass
class EarlyStoppingCallback:
    early_stopping_patience: int
    early_stopping_threshold: float = 0.0


@dataclass
class EvalPrediction:
    predictions: np.ndarray
    label_ids: np.ndarray


@dataclass
class PredictionOutput:
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Dict[str, Any]


def nested_detach(value):
    if isinstance(value, paddle.Tensor):
        return value.detach()
    if isinstance(value, (list, tuple)):
        return type(value)(nested_detach(v) for v in value)
    return value


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, paddle.Tensor):
        return value.numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.array(value)
    return np.asarray(value)


def get_parameter_names(model, forbidden_layer_types, prefix=""):
    result = []
    for child_name, child in model.named_children():
        child_prefix = f"{prefix}{child_name}."
        if isinstance(child, forbidden_layer_types):
            continue
        result.extend(get_parameter_names(child, forbidden_layer_types, child_prefix))
    for param_name, _ in model.named_parameters(prefix="", include_sublayers=False):
        result.append(f"{prefix}{param_name}")
    return result


class Trainer:
    def __init__(
        self,
        model,
        args: TrainingArguments,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=None,
        callbacks=None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks or []
        self.optimizer = None
        self.lr_scheduler = None
        self._all_lr_schedulers = []
        self.ar_steps = None
        self.output_all_steps = False
        self.best_metric = None
        self.best_model_dir = os.path.join(self.args.output_dir, "best_model")
        self.global_step = 0
        self.completed_epochs = 0
        self.log_history: List[Dict[str, Any]] = []
        self.device = paddle.device.get_device()
        self.device_type = self.device.split(":")[0]
        self._warned_single_process_loading = False
        for callback in self.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                self.args.early_stopping_patience = callback.early_stopping_patience
                break

    def _state_dict_file(self, output_dir: str, filename: str) -> str:
        return os.path.join(output_dir, filename)

    def _metric_key_name(self, metric_key_prefix: str, key: str) -> str:
        return f"{metric_key_prefix}_{key}" if metric_key_prefix else f"_{key}"

    def _resolve_best_metric_key(self, metrics: Dict[str, Any]) -> str:
        metric_key = self.args.metric_for_best_model
        if metric_key in metrics:
            return metric_key
        prefixed_key = f"eval_{metric_key}"
        if prefixed_key in metrics:
            return prefixed_key
        if "eval_loss" in metrics:
            return "eval_loss"
        return metric_key

    def _scalarize(self, value: Any) -> Any:
        if isinstance(value, paddle.Tensor):
            array = value.numpy()
            return float(array.reshape([-1])[0]) if array.size == 1 else array
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return float(value.reshape([-1])[0]) if value.size == 1 else value
        return value

    def _normalize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {key: self._scalarize(value) for key, value in metrics.items()}

    def _maybe_log_to_wandb(self, metrics: Dict[str, Any]):
        if self.args.report_to != "wandb":
            return
        try:
            import wandb
        except ImportError:
            return
        wandb.log(self._normalize_metrics(metrics))

    def _log(self, metrics: Dict[str, Any]):
        normalized = self._normalize_metrics(metrics)
        self.log_history.append(normalized)
        print(normalized)
        self._maybe_log_to_wandb(normalized)

    def _compute_grad_norm(self) -> Optional[float]:
        total = 0.0
        found_grad = False
        for parameter in self.model.parameters():
            grad = parameter.grad
            if grad is None:
                continue
            found_grad = True
            grad_value = grad.detach().astype("float32")
            total += float(paddle.sum(paddle.square(grad_value)).numpy())
        if not found_grad:
            return None
        return float(math.sqrt(total))

    def _current_learning_rate(self) -> float:
        if self.optimizer is None:
            return float(self.args.learning_rate)
        return float(self.optimizer.get_lr())

    def _save_training_state(self, output_dir: str, epoch: Optional[int] = None):
        os.makedirs(output_dir, exist_ok=True)
        if self.optimizer is not None:
            paddle.save(
                self.optimizer.state_dict(),
                self._state_dict_file(output_dir, "optimizer_state.pdopt"),
            )
        for i, scheduler in enumerate(self._all_lr_schedulers):
            paddle.save(
                scheduler.state_dict(),
                self._state_dict_file(output_dir, f"scheduler_state_{i}.pdsch"),
            )
        trainer_state = {
            "global_step": self.global_step,
            "completed_epochs": self.completed_epochs,
            "best_metric": self.best_metric,
            "epoch": epoch,
            "log_history": self.log_history,
            "num_schedulers": len(self._all_lr_schedulers),
        }
        with open(
            self._state_dict_file(output_dir, "trainer_state.json"),
            "w",
            encoding="utf-8",
        ) as stream:
            json.dump(trainer_state, stream, ensure_ascii=True, indent=2)

    def _load_training_state(self, checkpoint_dir: str):
        state_path = self._state_dict_file(checkpoint_dir, "model_state.pdparams")
        if os.path.exists(state_path):
            self.model.set_state_dict(paddle.load(state_path))

        optimizer_path = self._state_dict_file(checkpoint_dir, "optimizer_state.pdopt")
        if self.optimizer is not None and os.path.exists(optimizer_path):
            self.optimizer.set_state_dict(paddle.load(optimizer_path))

        # Restore all scheduler state dicts (supports per-group schedulers).
        trainer_state_path = self._state_dict_file(checkpoint_dir, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, "r", encoding="utf-8") as stream:
                trainer_state = json.load(stream)
            self.global_step = int(trainer_state.get("global_step", 0))
            self.completed_epochs = int(trainer_state.get("completed_epochs", 0))
            self.best_metric = trainer_state.get("best_metric")
            self.log_history = trainer_state.get("log_history", [])
            num_schedulers = trainer_state.get("num_schedulers", 0)
            for i in range(num_schedulers):
                sch_path = self._state_dict_file(
                    checkpoint_dir, f"scheduler_state_{i}.pdsch"
                )
                if i < len(self._all_lr_schedulers) and os.path.exists(sch_path):
                    self._all_lr_schedulers[i].set_state_dict(paddle.load(sch_path))

    def get_decay_parameter_names(self, model) -> List[str]:
        all_layernorm_layers = (paddle.nn.LayerNorm, LayerNorm, ConditionalLayerNorm)
        decay_parameters = get_parameter_names(model, all_layernorm_layers)
        return [name for name in decay_parameters if "bias" not in name]

    def get_conditional_norm_params(self, model):
        params = []
        for name, module in model.named_sublayers():
            if isinstance(module, ConditionalLayerNorm):
                for param_name, _ in module.named_parameters(
                    prefix="", include_sublayers=True
                ):
                    params.append(f"{name}.{param_name}")
        return params

    def create_optimizer(self, num_training_steps: Optional[int] = None):
        if self.optimizer is not None:
            return self.optimizer

        decay_parameters = set(self.get_decay_parameter_names(self.model))
        time_embedding_params = set(self.get_conditional_norm_params(self.model))
        params = {
            "standard": [],
            "no_weight_decay": [],
            "embeddings": [],
            "time_embedding": [],
        }

        for name, parameter in self.model.named_parameters():
            if parameter.stop_gradient:
                continue
            if ("embeddings" in name or "patch_recovery" in name) and (
                self.args.learning_rate_embedding_recovery is not None
            ):
                params["embeddings"].append(parameter)
            elif name in time_embedding_params and (
                self.args.learning_rate_time_embedding is not None
            ):
                params["time_embedding"].append(parameter)
            elif name in decay_parameters:
                params["standard"].append(parameter)
            else:
                params["no_weight_decay"].append(parameter)

        # Collect all LR schedulers so we can step them together in the training loop.
        # Each group that needs a different base LR gets its own scheduler instance,
        # matching the HF Trainer behavior where the scheduler scales every group proportionally.
        self._all_lr_schedulers = []

        if num_training_steps is None:
            learning_rate = self.args.learning_rate
        else:
            learning_rate = self.create_scheduler(num_training_steps)
            self._all_lr_schedulers.append(self.lr_scheduler)

        parameter_groups = []
        if params["standard"]:
            parameter_groups.append(
                {"params": params["standard"], "weight_decay": self.args.weight_decay}
            )
        if params["no_weight_decay"]:
            parameter_groups.append(
                {"params": params["no_weight_decay"], "weight_decay": 0.0}
            )
        if params["embeddings"]:
            emb_scheduler = self._create_group_scheduler(
                num_training_steps, self.args.learning_rate_embedding_recovery
            )
            if emb_scheduler is not None:
                self._all_lr_schedulers.append(emb_scheduler)
                emb_lr = emb_scheduler
            else:
                emb_lr = self.args.learning_rate_embedding_recovery
            parameter_groups.append(
                {
                    "params": params["embeddings"],
                    "weight_decay": self.args.weight_decay,
                    "learning_rate": emb_lr,
                }
            )
        if params["time_embedding"]:
            time_emb_scheduler = self._create_group_scheduler(
                num_training_steps, self.args.learning_rate_time_embedding
            )
            if time_emb_scheduler is not None:
                self._all_lr_schedulers.append(time_emb_scheduler)
                time_emb_lr = time_emb_scheduler
            else:
                time_emb_lr = self.args.learning_rate_time_embedding
            parameter_groups.append(
                {
                    "params": params["time_embedding"],
                    "weight_decay": 0.0,
                    "learning_rate": time_emb_lr,
                }
            )

        self.optimizer = paddle.optimizer.AdamW(
            learning_rate=learning_rate,
            beta1=self.args.adam_beta1,
            beta2=self.args.adam_beta2,
            epsilon=self.args.adam_epsilon,
            parameters=parameter_groups,
            weight_decay=self.args.weight_decay,
            grad_clip=(
                paddle.nn.ClipGradByGlobalNorm(self.args.max_grad_norm)
                if self.args.max_grad_norm and self.args.max_grad_norm > 0
                else None
            ),
        )
        return self.optimizer

    def _create_group_scheduler(self, num_training_steps, base_lr):
        """Create a LambdaDecay scheduler with a given base_lr for a parameter group.

        Returns the scheduler instance if num_training_steps is provided, else None.
        This ensures each group follows the same warmup/decay schedule shape
        (matching HF Trainer behavior) but scaled by its own base learning rate.
        """
        if num_training_steps is None:
            return None
        return self.create_scheduler(num_training_steps, base_lr=base_lr)

    def create_scheduler(self, num_training_steps: int, base_lr=None):
        if base_lr is None:
            base_lr = self.args.learning_rate
        warmup_steps = int(num_training_steps * self.args.warmup_ratio)

        def lr_lambda(current_step):
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, num_training_steps - warmup_steps)
            )
            progress = min(max(progress, 0.0), 1.0)
            scheduler_type = self.args.lr_scheduler_type
            if scheduler_type == "constant":
                return 1.0
            if scheduler_type == "linear":
                return max(0.0, 1.0 - progress)
            if scheduler_type == "cosine":
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}")

        scheduler = paddle.optimizer.lr.LambdaDecay(
            learning_rate=base_lr,
            lr_lambda=lr_lambda,
        )
        self.lr_scheduler = scheduler
        return scheduler

    def set_ar_steps(self, ar_steps=None, output_all_steps=False):
        self.ar_steps = ar_steps
        self.output_all_steps = self.ar_steps is not None and output_all_steps

    def _model_forward(self, model, inputs):
        if self.ar_steps is not None and model.config.use_conditioning:
            channel_difference = model.config.num_channels > model.config.num_out_channels
            if isinstance(self.ar_steps, int):
                inputs = {**inputs, "time": inputs["time"] / self.ar_steps}
                if self.output_all_steps:
                    losses, outputs_ = [], []
                    hidden_states_, attentions_, reshaped_hidden_states_ = [], [], []
                else:
                    loss = 0
                for _ in range(self.ar_steps):
                    outputs = model(**inputs)
                    if self.output_all_steps:
                        outputs_.append(outputs.output.detach())
                        if outputs.hidden_states is not None:
                            hidden_states_.append(outputs.hidden_states)
                        if outputs.attentions is not None:
                            attentions_.append(outputs.attentions)
                        if outputs.reshaped_hidden_states is not None:
                            reshaped_hidden_states_.append(outputs.reshaped_hidden_states)
                        if outputs.loss is not None:
                            losses.append(outputs.loss)
                    else:
                        if outputs.loss is not None:
                            loss = loss + outputs.loss
                    next_output = outputs.output.detach()
                    if channel_difference:
                        next_output = paddle.concat(
                            [
                                next_output,
                                inputs["pixel_values"][:, model.config.num_out_channels :],
                            ],
                            axis=1,
                        )
                    inputs = {**inputs, "pixel_values": next_output}
                if self.output_all_steps:
                    outputs.output = paddle.stack(outputs_, axis=1)
                    if losses:
                        outputs.loss = paddle.stack(losses, axis=0)
                    if hidden_states_:
                        outputs.hidden_states = tuple(
                            paddle.stack(states, axis=1) for states in zip(*hidden_states_)
                        )
                    if attentions_:
                        outputs.attentions = tuple(
                            paddle.stack(states, axis=1) for states in zip(*attentions_)
                        )
                    if reshaped_hidden_states_:
                        outputs.reshaped_hidden_states = tuple(
                            paddle.stack(states, axis=1)
                            for states in zip(*reshaped_hidden_states_)
                        )
                else:
                    outputs.loss = loss / self.ar_steps
            elif isinstance(self.ar_steps, list):
                if self.output_all_steps:
                    losses, outputs_ = [], []
                    hidden_states_, attentions_, reshaped_hidden_states_ = [], [], []
                else:
                    loss = 0
                lead_time = inputs["time"]
                for step in self.ar_steps:
                    inputs = {**inputs, "time": lead_time * step}
                    outputs = model(**inputs)
                    if self.output_all_steps:
                        outputs_.append(outputs.output.detach())
                        if outputs.hidden_states is not None:
                            hidden_states_.append(outputs.hidden_states)
                        if outputs.attentions is not None:
                            attentions_.append(outputs.attentions)
                        if outputs.reshaped_hidden_states is not None:
                            reshaped_hidden_states_.append(outputs.reshaped_hidden_states)
                        if outputs.loss is not None:
                            losses.append(outputs.loss)
                    else:
                        if outputs.loss is not None:
                            loss = loss + outputs.loss
                    next_output = outputs.output.detach()
                    if channel_difference:
                        next_output = paddle.concat(
                            [
                                next_output,
                                inputs["pixel_values"][:, model.config.num_out_channels :],
                            ],
                            axis=1,
                        )
                    inputs = {**inputs, "pixel_values": next_output}
                if self.output_all_steps:
                    outputs.output = paddle.stack(outputs_, axis=1)
                    if losses:
                        loss_axis = 0 if losses[0].ndim == 0 else 1
                        outputs.loss = paddle.stack(losses, axis=loss_axis)
                    if hidden_states_:
                        outputs.hidden_states = tuple(
                            paddle.stack(states, axis=1) for states in zip(*hidden_states_)
                        )
                    if attentions_:
                        outputs.attentions = tuple(
                            paddle.stack(states, axis=1) for states in zip(*attentions_)
                        )
                    if reshaped_hidden_states_:
                        outputs.reshaped_hidden_states = tuple(
                            paddle.stack(states, axis=1)
                            for states in zip(*reshaped_hidden_states_)
                        )
                else:
                    outputs.loss = loss / len(self.ar_steps)
            else:
                raise ValueError("num_ar_steps must be an integer or a list of integers.")
        else:
            if os.environ.get("POSEIDON_USE_CINN", "0") == "1":
                # to_static(full_graph=True) 下 dataclass(ScOTOutput) 返回 pir.Value 不能 backward；
                # 改用 return_dict=False 拿 eager tuple 再包装。AR rollout 路径暂不支持 CINN。
                out_tuple = model(**inputs, return_dict=False)
                outputs = ScOTOutput(loss=out_tuple[0], output=out_tuple[1])
            else:
                outputs = model(**inputs)
        return outputs

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = self._model_forward(model, inputs)
        if outputs.loss is None:
            raise ValueError(
                "The model did not return a loss from the inputs."
            )
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def _effective_num_workers(self) -> int:
        if self.device_type != "cpu":
            if not self._warned_single_process_loading:
                print(
                    f"Using single-process data loading on {self.device} to avoid Paddle worker instability."
                )
                self._warned_single_process_loading = True
            return 0
        return self.args.dataloader_num_workers

    def _prepare_batch(self, batch):
        prepared = {}
        for key, value in batch.items():
            if isinstance(value, paddle.Tensor):
                if self.device_type != "cpu" and value.place.is_cpu_place():
                    value = paddle.to_tensor(value.numpy(), place=self.device)
            else:
                value = paddle.to_tensor(value, place=self.device)
            prepared[key] = value
        return prepared

    def _load_debug_epoch_batches(self):
        path = getattr(self.args, "debug_batch_order_path", None)
        if not path:
            return None
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        epoch_batches = payload.get("epoch_batches")
        if epoch_batches is None:
            epoch_batches = [payload["batches"]]
        if len(epoch_batches) == 0:
            raise ValueError("debug epoch batch order must contain at least one epoch")
        return epoch_batches

    def _collate_indexed_batch(self, indices):
        samples = [self.train_dataset[int(index)] for index in indices]
        batch = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if isinstance(values[0], paddle.Tensor):
                batch[key] = paddle.stack(values, axis=0)
            else:
                batch[key] = paddle.to_tensor(values)
        return batch

    def _iter_debug_epoch_batches(self, epoch_batches, epoch):
        current_batches = epoch_batches[epoch % len(epoch_batches)]
        for indices in current_batches:
            yield self._collate_indexed_batch(indices)

    def _create_dataloader(self, dataset, batch_size, shuffle, drop_last=False):
        return paddle.io.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self._effective_num_workers(),
            return_list=True,
            drop_last=drop_last,
        )

    def evaluate(self):
        if self.eval_dataset is None:
            return {}
        return self.predict(self.eval_dataset, metric_key_prefix="eval").metrics

    def train(self, resume_from_checkpoint=False):
        if self.train_dataset is None:
            raise ValueError("train_dataset is required for training.")
        os.makedirs(self.args.output_dir, exist_ok=True)
        # CINN 动转静下，末 batch 维度变化会触发 retrace + CINN 重编译(~100s/次)；
        # 故 CINN 模式训练强制 drop_last=True 保证 batch 维度稳定。eval/predict 不受影响。
        train_drop_last = os.environ.get("POSEIDON_USE_CINN", "0") == "1"
        train_dataloader = self._create_dataloader(
            self.train_dataset,
            self.args.per_device_train_batch_size,
            shuffle=True,
            drop_last=train_drop_last,
        )
        debug_epoch_batches = self._load_debug_epoch_batches()
        if debug_epoch_batches is not None:
            train_dataloader = None
            steps_per_epoch = len(debug_epoch_batches[0])
        else:
            steps_per_epoch = len(train_dataloader)
        num_training_steps = steps_per_epoch * int(self.args.num_train_epochs)
        optimizer = self.create_optimizer(num_training_steps)

        if resume_from_checkpoint:
            checkpoint_dir = (
                resume_from_checkpoint
                if isinstance(resume_from_checkpoint, str)
                else self.args.output_dir
            )
            self._load_training_state(checkpoint_dir)

        patience = self.args.early_stopping_patience
        no_improve_epochs = 0
        train_start = time.perf_counter()
        all_epoch_losses = []

        for epoch in range(int(self.args.num_train_epochs)):
            self.model.train()
            epoch_losses = []
            epoch_iterator = (
                train_dataloader
                if debug_epoch_batches is None
                else self._iter_debug_epoch_batches(debug_epoch_batches, epoch)
            )
            steps_in_epoch = max(steps_per_epoch, 1)
            last_grad_norm = None

            for step, batch in enumerate(epoch_iterator, start=1):
                inputs = self._prepare_batch(batch)
                loss = self.compute_loss(self.model, inputs)
                loss_value = float(loss.detach().numpy())
                epoch_losses.append(loss_value)
                all_epoch_losses.append(loss_value)

                loss.backward()
                last_grad_norm = self._compute_grad_norm()
                optimizer.step()
                if self.lr_scheduler is not None:
                    for scheduler in self._all_lr_schedulers:
                        scheduler.step()
                optimizer.clear_grad()
                self.global_step += 1

                if (
                    self.args.logging_strategy == "steps"
                    and self.args.logging_steps > 0
                    and self.global_step % self.args.logging_steps == 0
                ):
                    self._log(
                        {
                            "loss": float(
                                np.mean(epoch_losses[-self.args.logging_steps :])
                            ),
                            "grad_norm": last_grad_norm,
                            "learning_rate": self._current_learning_rate(),
                            "epoch": epoch + step / steps_in_epoch,
                        }
                    )

            train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            self.completed_epochs = epoch + 1

            if self.args.logging_strategy == "epoch":
                self._log(
                    {
                        "loss": train_loss,
                        "grad_norm": last_grad_norm,
                        "learning_rate": self._current_learning_rate(),
                        "epoch": float(self.completed_epochs),
                    }
                )

            if self.args.evaluation_strategy == "epoch" and self.eval_dataset is not None:
                eval_metrics = self.evaluate()
                eval_metrics["epoch"] = float(self.completed_epochs)
                self._log(eval_metrics)
                metric_key = self._resolve_best_metric_key(eval_metrics)
                metric_value = eval_metrics.get(metric_key)
                if metric_value is not None:
                    better = (
                        self.best_metric is None
                        or (
                            metric_value > self.best_metric
                            if self.args.greater_is_better
                            else metric_value < self.best_metric
                        )
                    )
                    if better:
                        self.best_metric = metric_value
                        no_improve_epochs = 0
                        self.model.save_pretrained(self.best_model_dir)
                        self._save_training_state(self.best_model_dir, epoch=epoch + 1)
                    else:
                        no_improve_epochs += 1
                if self.args.save_strategy == "epoch":
                    self.model.save_pretrained(self.args.output_dir)
                    self._save_training_state(self.args.output_dir, epoch=epoch + 1)
                if patience is not None and no_improve_epochs >= patience:
                    break
            elif self.args.save_strategy == "epoch":
                self.model.save_pretrained(self.args.output_dir)
                self._save_training_state(self.args.output_dir, epoch=epoch + 1)

        train_runtime = time.perf_counter() - train_start
        num_train_samples = (
            len(self.train_dataset) * max(self.completed_epochs, 1)
            if hasattr(self.train_dataset, "__len__")
            else 0
        )
        train_summary = {
            "train_runtime": train_runtime,
            "train_samples_per_second": (
                num_train_samples / train_runtime if train_runtime > 0 else 0.0
            ),
            "train_steps_per_second": (
                self.global_step / train_runtime if train_runtime > 0 else 0.0
            ),
            "train_loss": float(np.mean(all_epoch_losses)) if all_epoch_losses else 0.0,
            "epoch": float(self.completed_epochs),
        }
        self._log(train_summary)

        if self.args.load_best_model_at_end and os.path.exists(self.best_model_dir):
            best_model = self.model.__class__.from_pretrained(
                self.best_model_dir, config=self.model.config, ignore_mismatched_sizes=True
            )
            self.model.set_state_dict(best_model.state_dict())

        return train_summary

    def predict(self, dataset, metric_key_prefix="eval"):
        dataloader = self._create_dataloader(
            dataset, self.args.per_device_eval_batch_size, shuffle=False
        )
        self.model.eval()
        losses = []
        predictions = []
        label_ids = []
        start_time = time.perf_counter()

        with paddle.no_grad():
            for batch in dataloader:
                inputs = self._prepare_batch(batch)
                outputs = self._model_forward(self.model, inputs)
                if outputs.loss is not None:
                    losses.append(float(outputs.loss.mean().numpy()))
                predictions.append(_to_numpy(outputs.output))
                labels = inputs.get("labels")
                if labels is not None:
                    label_ids.append(_to_numpy(labels))

        predictions = np.concatenate(predictions, axis=0) if predictions else None
        labels_np = np.concatenate(label_ids, axis=0) if label_ids else None
        metrics = {}
        loss_key = f"{metric_key_prefix}_loss" if metric_key_prefix else "_loss"
        if losses:
            metrics[loss_key] = float(np.mean(losses))
        runtime = time.perf_counter() - start_time
        metrics[self._metric_key_name(metric_key_prefix, "runtime")] = runtime
        num_samples = len(dataset) if hasattr(dataset, "__len__") else 0
        if runtime > 0:
            metrics[self._metric_key_name(metric_key_prefix, "samples_per_second")] = (
                num_samples / runtime
            )
            metrics[self._metric_key_name(metric_key_prefix, "steps_per_second")] = (
                len(dataloader) / runtime
            )
        if self.compute_metrics is not None and predictions is not None and labels_np is not None:
            computed_metrics = self.compute_metrics(
                EvalPrediction(predictions=predictions, label_ids=labels_np)
            )
            if computed_metrics:
                metrics.update(
                    {
                        (
                            f"{metric_key_prefix}_{key}"
                            if metric_key_prefix
                            else f"_{key}"
                        ): value
                        for key, value in computed_metrics.items()
                    }
                )
        return PredictionOutput(predictions=predictions, label_ids=labels_np, metrics=metrics)

    def save_model(self, output_dir: Optional[str] = None):
        self.model.save_pretrained(output_dir or self.args.output_dir)
