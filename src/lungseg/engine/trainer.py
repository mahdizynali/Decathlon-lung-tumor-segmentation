"""Auditable plain-PyTorch training primitives for 3D segmentation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lungseg.engine.inference import predict_volume
from lungseg.metrics.segmentation import binary_case_metrics
from lungseg.utils.io import write_json


def train_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: bool = True,
    gradient_clip_norm: float | None = None,
) -> float:
    """Train for one epoch and return sample-weighted mean loss."""
    model.train()
    total_loss = 0.0
    sample_count = 0
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for batch in loader:
        images = batch["image"].to(device=device, dtype=torch.float32)
        labels = batch["label"].to(device=device, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
        with context:
            logits = model(images)
            loss = loss_function(logits, labels)
        scaler.scale(loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        batch_size = images.shape[0]
        total_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("Training loader produced no samples")
    return total_loss / sample_count


def validate_cases(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    roi_size: Sequence[int],
    *,
    overlap: float = 0.5,
) -> list[dict[str, Any]]:
    """Evaluate complete validation volumes and return one row per case."""
    rows: list[dict[str, Any]] = []
    for batch in loader:
        images = batch["image"].to(device=device, dtype=torch.float32)
        labels = batch["label"]
        if images.shape[0] != 1:
            raise ValueError("Whole-volume validation requires batch_size=1")
        prediction = predict_volume(model, images, roi_size, overlap=overlap)[0, 0].cpu().numpy()
        target = labels[0, 0].cpu().numpy()
        meta = batch.get("image_meta_dict", {})
        spacing = meta.get("pixdim")
        spacing_values = tuple(float(value) for value in spacing[0][1:4]) if spacing is not None else (1, 1, 1)
        metrics = binary_case_metrics(prediction, target, spacing_values)
        case_value = batch.get("case_id", ["unknown"])
        case_id = case_value[0] if isinstance(case_value, (list, tuple)) else str(case_value)
        rows.append({"case_id": case_id, **metrics})
    if not rows:
        raise ValueError("Validation loader produced no cases")
    return rows


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        destination,
    )
    write_json(destination.with_suffix(".metrics.json"), metrics)
    return destination
