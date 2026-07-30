from __future__ import annotations

import math
import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zcp_test.artifacts import JsonlWriter
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
    label_smoothing: float = 0.0
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
    writer = JsonlWriter(output / "training.jsonl", fsync_every=1)
    model.to(device)
    valid_criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    mixup_fn = None
    train_criterion = valid_criterion
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
        optimizer = torch.optim.AdamW(model.parameters(), config.learning_rate, weight_decay=config.weight_decay)
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
    if scheduler_name == "cosine":
        def learning_rate_multiplier(epoch: int) -> float:
            if config.warmup_epochs and epoch < config.warmup_epochs:
                return float(epoch + 1) / config.warmup_epochs
            progress = (epoch - config.warmup_epochs) / max(
                1, config.epochs - config.warmup_epochs
            )
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

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
    if resume:
        checkpoint = load_checkpoint(resume, trusted=resume_trusted)
        if checkpoint.get("config") != config.__dict__:
            raise ValueError("Checkpoint training config does not match the requested config")
        if checkpoint.get("run_identity") != run_identity:
            raise ValueError("Checkpoint architecture or protocol identity does not match this run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"])
        start_epoch, best_accuracy = checkpoint["epoch"] + 1, checkpoint["best_accuracy"]
    for epoch in range(start_epoch, config.epochs):
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        unwrapped_model = getattr(model, "module", model)
        if hasattr(unwrapped_model, "drop_path_prob"):
            unwrapped_model.drop_path_prob = config.drop_path_prob * epoch / max(
                1, config.epochs - 1
            )
        started = time.perf_counter()
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
        valid_loss, valid_top1, valid_top5, valid_count, _ = _epoch(
            model, valid_loader, valid_criterion, device, config
        )
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
            "duration_seconds": time.perf_counter() - started,
            "optimizer_steps": optimizer_steps,
            "best": valid_accuracy > best_accuracy,
        }
        writer.append(record)
        best_accuracy = max(best_accuracy, valid_accuracy)
        payload = {"epoch": epoch, "best_accuracy": best_accuracy, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(), "config": config.__dict__, "run_identity": run_identity}
        atomic_torch_save(payload, output / "checkpoints" / "last.pt")
        if record["best"]:
            atomic_torch_save(payload, output / "checkpoints" / "best.pt")
    return {"best_accuracy": best_accuracy, "last_epoch": config.epochs - 1}


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
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        accuracy_labels = labels
        if training and mixup_fn is not None:
            inputs, labels = mixup_fn(inputs, labels)
        if training:
            optimizer.zero_grad(set_to_none=True)
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
            scaler.scale(loss).backward()
            if config.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += int(scaler.get_scale() >= scale_before)
        total_loss += float(loss.detach()) * labels.size(0)
        predictions = output.topk(min(5, output.shape[1]), dim=1).indices
        matches = predictions.eq(accuracy_labels.view(-1, 1))
        top1_correct += int(matches[:, :1].any(dim=1).sum())
        top5_correct += int(matches.any(dim=1).sum())
        count += accuracy_labels.size(0)
    return total_loss, top1_correct, top5_correct, count, optimizer_steps
