from __future__ import annotations

import math
import inspect
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zcp_test.artifacts import JsonlWriter, read_jsonl
from zcp_test.training.checkpoint import atomic_torch_save, load_checkpoint, restore_rng, rng_state


@dataclass
class TrainingConfig:
    epochs: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    scheduler: str = "cosine"
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.97
    warmup_epochs: int = 0
    warmup_learning_rate: float | None = None
    minimum_learning_rate: float = 0.0
    label_smoothing: float = 0.0
    validation_label_smoothing: float = 0.0
    amp: bool = True
    momentum: float = 0.9
    nesterov: bool = True
    auxiliary_weight: float = 0.0
    drop_path_prob: float = 0.0
    grad_clip: float | None = None
    amp_initial_scale: float = 65536.0
    mixup: float = 0.0
    cutmix: float = 0.0
    mixup_probability: float = 1.0
    mixup_switch_probability: float = 0.5
    mixup_mode: str = "batch"
    gradient_accumulation_steps: int = 1
    schedule_epochs: int | None = None
    exclude_bias_norm_from_weight_decay: bool = False


def _optimizer_parameter_groups(
    model: Any, weight_decay: float, exclude_bias_norm: bool
) -> Any:
    if not exclude_bias_norm:
        return model.parameters()
    unwrapped = getattr(model, "module", model)
    declared = (
        set(unwrapped.no_weight_decay())
        if callable(getattr(unwrapped, "no_weight_decay", None))
        else set()
    )
    decay = []
    no_decay = []
    for name, parameter in unwrapped.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or name in declared:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if not decay or not no_decay:
        raise ValueError("weight-decay exemption requires both decay and no-decay parameters")
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _cosine_learning_rate(config: TrainingConfig, epoch: int, schedule_epochs: int) -> float:
    if schedule_epochs <= 0 or epoch < 0:
        raise ValueError("schedule epoch values must be non-negative and non-empty")
    minimum = float(config.minimum_learning_rate)
    base = float(config.learning_rate)
    if not 0 <= minimum <= base:
        raise ValueError("minimum_learning_rate must be between zero and learning_rate")
    if config.warmup_epochs and epoch < config.warmup_epochs:
        if config.warmup_learning_rate is None:
            return base * float(epoch + 1) / config.warmup_epochs
        warmup = float(config.warmup_learning_rate)
        if not 0 <= warmup <= base:
            raise ValueError("warmup_learning_rate must be between zero and learning_rate")
        return warmup + (base - warmup) * epoch / config.warmup_epochs
    progress = (epoch - config.warmup_epochs) / max(
        1, schedule_epochs - config.warmup_epochs
    )
    return minimum + 0.5 * (base - minimum) * (
        1.0 + math.cos(math.pi * min(1.0, progress))
    )


def _restore_training_log(
    writer: JsonlWriter | None,
    source_training_log: str | Path | None,
    checkpoint_epoch: int,
    training_history: list[dict[str, Any]] | None = None,
) -> int:
    if writer is None:
        return 0
    source_records: Any = training_history or []
    if source_training_log:
        source = Path(source_training_log)
        if source.exists():
            if source.resolve() == writer.path.resolve():
                return 0
            source_records = read_jsonl(source)
    existing_epochs = {
        int(record["epoch"])
        for record in read_jsonl(writer.path)
        if record.get("epoch") is not None
    }
    restored_rows = 0
    for record in source_records:
        epoch = int(record["epoch"])
        if epoch <= checkpoint_epoch and epoch not in existing_epochs:
            writer.append(record)
            existing_epochs.add(epoch)
            restored_rows += 1
    return restored_rows


def _collect_checkpoint_rng(distributed: bool, distributed_rank: int) -> dict[str, Any]:
    import torch

    local_state = rng_state()
    if not distributed:
        return {"rng": local_state}
    states: list[dict[str, Any] | None] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(states, local_state)
    if any(state is None for state in states):
        raise RuntimeError("Failed to collect RNG state from every distributed rank")
    return {"rng": states[0], "rng_by_rank": states}


def _restore_checkpoint_rng(
    checkpoint: dict[str, Any], distributed: bool, distributed_rank: int
) -> None:
    import torch

    states = checkpoint.get("rng_by_rank")
    if not distributed:
        restore_rng(checkpoint["rng"])
        return
    if not isinstance(states, list):
        raise ValueError(
            "Distributed resume requires a checkpoint with rank-local RNG states"
        )
    world_size = torch.distributed.get_world_size()
    if len(states) != world_size:
        raise ValueError(
            "Checkpoint RNG world size does not match the distributed resume world size"
        )
    state = states[distributed_rank]
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint RNG state for rank {distributed_rank} is invalid")
    restore_rng(state)


def train_model(
    model: Any,
    train_loader: Any,
    valid_loader: Any,
    config: TrainingConfig,
    output: str | Path,
    device: Any,
    resume: str | Path | None = None,
    *,
    resume_trusted: bool = False,
    run_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    output = Path(output)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    distributed_rank = torch.distributed.get_rank() if distributed else 0
    primary_process = distributed_rank == 0
    writer = JsonlWriter(output / "training.jsonl", fsync_every=1) if primary_process else None
    model.to(device)
    train_criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    valid_criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.validation_label_smoothing)
    mixup_fn = None
    if config.mixup > 0 or config.cutmix > 0:
        from timm.data import Mixup
        from timm.loss import SoftTargetCrossEntropy

        num_classes = next(
            module.out_features
            for module in reversed(list(model.modules()))
            if isinstance(module, torch.nn.Linear)
        )
        mixup_fn = Mixup(
            mixup_alpha=config.mixup,
            cutmix_alpha=config.cutmix,
            prob=config.mixup_probability,
            switch_prob=config.mixup_switch_probability,
            mode=config.mixup_mode,
            label_smoothing=config.label_smoothing,
            num_classes=num_classes,
        )
        train_criterion = SoftTargetCrossEntropy()
    if config.optimizer.lower() == "adamw":
        parameters = _optimizer_parameter_groups(
            model,
            config.weight_decay,
            config.exclude_bias_norm_from_weight_decay,
        )
        optimizer = torch.optim.AdamW(
            parameters,
            config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            config.learning_rate,
            momentum=config.momentum,
            nesterov=config.nesterov,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer {config.optimizer}")
    scheduler_name = config.scheduler.casefold()
    schedule_epochs = config.epochs if config.schedule_epochs is None else config.schedule_epochs
    if schedule_epochs < config.epochs:
        raise ValueError("schedule_epochs must be greater than or equal to epochs")
    if scheduler_name == "cosine":
        def learning_rate_multiplier(epoch: int) -> float:
            return _cosine_learning_rate(config, epoch, schedule_epochs) / config.learning_rate

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)
    elif scheduler_name == "step":
        if config.warmup_epochs:
            raise ValueError("step scheduler does not support warmup_epochs")
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    elif scheduler_name == "none":
        if config.warmup_epochs:
            raise ValueError("none scheduler does not support warmup_epochs")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    else:
        raise ValueError(f"Unsupported scheduler {config.scheduler!r}")
    scaler_enabled = config.amp and device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(
            "cuda", enabled=scaler_enabled, init_scale=config.amp_initial_scale
        )
    else:
        scaler = torch.cuda.amp.GradScaler(
            enabled=scaler_enabled, init_scale=config.amp_initial_scale
        )
    start_epoch, best_accuracy = 0, float("-inf")
    resumed_training_rows = 0
    if resume:
        checkpoint = load_checkpoint(resume, trusted=resume_trusted)
        if checkpoint.get("config") != config.__dict__:
            raise ValueError("Checkpoint training config does not match the requested config")
        if checkpoint.get("run_identity") != run_identity:
            raise ValueError("Checkpoint architecture or protocol identity does not match this run")
        getattr(model, "module", model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        _restore_checkpoint_rng(checkpoint, distributed, distributed_rank)
        start_epoch, best_accuracy = checkpoint["epoch"] + 1, checkpoint["best_accuracy"]
        resumed_training_rows = _restore_training_log(
            writer,
            checkpoint.get("training_log"),
            int(checkpoint["epoch"]),
            checkpoint.get("training_history"),
        )
    for epoch in range(start_epoch, config.epochs):
        for loader in (train_loader, valid_loader):
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        unwrapped_model = getattr(model, "module", model)
        if hasattr(unwrapped_model, "drop_path_prob"):
            unwrapped_model.drop_path_prob = config.drop_path_prob * epoch / max(
                1, schedule_epochs
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        train_started = started
        train_loss, train_top1, train_top5, train_count, optimizer_steps = _epoch(
            model,
            train_loader,
            train_criterion,
            device,
            config,
            optimizer,
            scaler,
            mixup_fn,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_duration = time.perf_counter() - train_started
        valid_started = time.perf_counter()
        valid_loss, valid_top1, valid_top5, valid_count, _ = _epoch(
            model, valid_loader, valid_criterion, device, config
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        valid_duration = time.perf_counter() - valid_started
        duration = time.perf_counter() - started
        if optimizer_steps:
            scheduler.step()
        train_accuracy = 100.0 * train_top1 / max(1, train_count)
        valid_accuracy = 100.0 * valid_top1 / max(1, valid_count)
        record = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_count),
            "valid_loss": valid_loss / max(1, valid_count),
            "train_top1": train_accuracy,
            "valid_top1": valid_accuracy,
            "train_top5": 100.0 * train_top5 / max(1, train_count),
            "valid_top5": 100.0 * valid_top5 / max(1, valid_count),
            "learning_rate": epoch_learning_rate,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "drop_path_prob": float(getattr(unwrapped_model, "drop_path_prob", 0.0)),
            "duration_seconds": duration,
            "train_duration_seconds": train_duration,
            "valid_duration_seconds": valid_duration,
            "train_samples_per_second": train_count / max(train_duration, 1e-12),
            "valid_samples_per_second": valid_count / max(valid_duration, 1e-12),
            "peak_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
            "peak_reserved_memory_mb": (
                torch.cuda.max_memory_reserved(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
            "optimizer_steps": optimizer_steps,
            "best": valid_accuracy > best_accuracy,
        }
        if writer is not None:
            writer.append(record)
        best_accuracy = max(best_accuracy, valid_accuracy)
        checkpoint_rng = _collect_checkpoint_rng(distributed, distributed_rank)
        if primary_process:
            payload = {"epoch": epoch, "best_accuracy": best_accuracy, "model": unwrapped_model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), **checkpoint_rng, "config": config.__dict__, "run_identity": run_identity, "training_log": str(writer.path.resolve()), "training_history": list(read_jsonl(writer.path)), "run_directory": str(output.resolve())}
            atomic_torch_save(payload, output / "checkpoints" / "last.pt")
            if record["best"]:
                atomic_torch_save(payload, output / "checkpoints" / "best.pt")
        if distributed:
            torch.distributed.barrier()
    return {
        "best_accuracy": best_accuracy,
        "last_epoch": config.epochs - 1,
        "resumed_training_rows": resumed_training_rows,
    }


def _epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    config: TrainingConfig,
    optimizer: Any | None = None,
    scaler: Any | None = None,
    mixup_fn: Any | None = None,
) -> tuple[float, int, int, int, int]:
    import torch

    training = optimizer is not None
    model.train(training)
    forward = getattr(model, "module", model).forward
    supports_auxiliary = "return_auxiliary" in inspect.signature(forward).parameters
    total_loss = top1_correct = top5_correct = count = optimizer_steps = 0
    if config.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    loader_batches = len(loader)
    accumulated_batches = 0
    for batch_index, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        accuracy_labels = labels
        if training and mixup_fn is not None:
            inputs, labels = mixup_fn(inputs, labels)
        if training and accumulated_batches == 0:
            optimizer.zero_grad(set_to_none=True)
        accumulated_batches += int(training)
        should_step = training and (
            accumulated_batches == config.gradient_accumulation_steps
            or batch_index + 1 == loader_batches
        )
        synchronization = (
            model.no_sync()
            if training and not should_step and hasattr(model, "no_sync")
            else nullcontext()
        )
        with synchronization:
            with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=scaler is not None and scaler.is_enabled()):
                raw_output = (
                    model(inputs, return_auxiliary=True)
                    if training and supports_auxiliary and config.auxiliary_weight
                    else model(inputs)
                )
                auxiliary = None
                if isinstance(raw_output, (tuple, list)):
                    output = raw_output[0]
                    auxiliary = raw_output[1] if len(raw_output) > 1 else None
                else:
                    output = raw_output
                loss = criterion(output, labels)
                if training and auxiliary is not None and config.auxiliary_weight:
                    loss = loss + config.auxiliary_weight * criterion(auxiliary, labels)
            if training:
                scaler.scale(loss / config.gradient_accumulation_steps).backward()
        if training:
            if should_step and (
                config.grad_clip is not None
                or accumulated_batches != config.gradient_accumulation_steps
            ):
                scaler.unscale_(optimizer)
            if should_step and accumulated_batches != config.gradient_accumulation_steps:
                correction = config.gradient_accumulation_steps / accumulated_batches
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            if should_step and config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if should_step:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_steps += int(scaler.get_scale() >= scale_before)
                accumulated_batches = 0
        total_loss += float(loss.detach()) * labels.size(0)
        predictions = output.topk(min(5, output.shape[1]), dim=1).indices
        matches = predictions.eq(accuracy_labels.view(-1, 1))
        top1_correct += int(matches[:, :1].any(dim=1).sum())
        top5_correct += int(matches.any(dim=1).sum())
        count += accuracy_labels.size(0)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        totals = torch.tensor(
            [total_loss, top1_correct, top5_correct, count],
            dtype=torch.float64,
            device=device,
        )
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        step_count = torch.tensor(optimizer_steps, dtype=torch.int64, device=device)
        torch.distributed.all_reduce(step_count, op=torch.distributed.ReduceOp.MAX)
        total_loss, top1_correct, top5_correct, count = totals.tolist()
        optimizer_steps = int(step_count.item())
    return total_loss, int(top1_correct), int(top5_correct), int(count), optimizer_steps
