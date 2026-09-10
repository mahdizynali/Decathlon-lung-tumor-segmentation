"""Unified command-line management for audit, training, inference, and tests."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

from lungseg.config import load_yaml, merge_configs
from lungseg.data.audit import CaseAudit, audit_case, summarize_audits
from lungseg.data.dataset import make_loader
from lungseg.data.discovery import LungCase, discover_cases
from lungseg.data.splits import make_folds
from lungseg.data.transforms import deterministic_transforms, training_transforms
from lungseg.engine.inference import predict_volume
from lungseg.engine.trainer import save_checkpoint, train_epoch, validate_cases
from lungseg.losses import build_loss
from lungseg.models import build_model
from lungseg.utils.environment import collect_environment
from lungseg.utils.io import write_json
from lungseg.utils.reproducibility import seed_everything


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_CONFIG = ROOT / "configs/data/msd_lung.yaml"
DEFAULT_MODEL_CONFIG = ROOT / "configs/model/unet3d.yaml"
DEFAULT_EXPERIMENT_CONFIG = ROOT / "configs/experiment/smoke.yaml"


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_configs(args: argparse.Namespace) -> dict[str, Any]:
    return merge_configs(
        load_yaml(_resolve(args.data_config)),
        load_yaml(_resolve(args.model_config)),
        load_yaml(_resolve(args.experiment_config)),
    )


def _required_preprocessing(config: dict[str, Any]) -> tuple[tuple[float, ...], float, float, tuple[int, ...]]:
    data = config["data"]
    intensity = data["intensity"]
    missing = [
        name
        for name, value in {
            "data.target_spacing": data.get("target_spacing"),
            "data.patch_size": data.get("patch_size"),
            "data.intensity.lower_hu": intensity.get("lower_hu"),
            "data.intensity.upper_hu": intensity.get("upper_hu"),
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "Training/inference parameters are unset: "
            + ", ".join(missing)
            + ". Run 'python main.py inspect' and set audit-supported values in the data config."
        )
    spacing = tuple(float(value) for value in data["target_spacing"])
    patch = tuple(int(value) for value in data["patch_size"])
    return spacing, float(intensity["lower_hu"]), float(intensity["upper_hu"]), patch


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _load_split(path: Path, fold_index: int, cases: list[LungCase]) -> tuple[list[LungCase], list[LungCase]]:
    split_path = _resolve(path)
    if not split_path.is_file():
        raise FileNotFoundError(
            f"Split file does not exist: {split_path}. Run 'python main.py split' first."
        )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    records = {int(item["fold"]): item for item in split.get("folds", [])}
    if fold_index not in records:
        raise ValueError(f"Fold {fold_index} is absent from {split_path}")
    indexed = {case.case_id: case for case in cases}
    record = records[fold_index]
    requested = set(record["train"]) | set(record["validation"])
    missing = sorted(requested - indexed.keys())
    if missing:
        raise ValueError(f"Split references undiscovered cases: {missing}")
    train = [indexed[case_id] for case_id in record["train"]]
    validation = [indexed[case_id] for case_id in record["validation"]]
    if set(record["train"]) & set(record["validation"]):
        raise ValueError(f"Fold {fold_index} has patient leakage")
    return train, validation


def command_environment(args: argparse.Namespace) -> int:
    report = collect_environment(ROOT)
    if args.output:
        write_json(_resolve(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_status(args: argparse.Namespace) -> int:
    config = load_yaml(_resolve(args.data_config))["data"]
    data_root = _resolve(Path(config["root"]))
    checks = {
        "repository": str(ROOT),
        "python": sys.version.split()[0],
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "data_root": str(data_root),
        "dataset_present": (data_root / "dataset.json").is_file(),
        "target_spacing_set": config.get("target_spacing") is not None,
        "ct_window_set": all(config["intensity"].get(key) is not None for key in ("lower_hu", "upper_hu")),
        "patch_size_set": config.get("patch_size") is not None,
        "default_split_present": (ROOT / "artifacts/splits/msd_lung_5fold_seed43.json").is_file(),
    }
    print(json.dumps(checks, indent=2))
    ready = all(
        checks[key]
        for key in ("dataset_present", "target_spacing_set", "ct_window_set", "patch_size_set")
    )
    print(f"\nTraining readiness: {'READY' if ready else 'NOT READY — run inspect and configure preprocessing'}")
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    data_root = _resolve(args.data)
    output = _resolve(args.output)
    audits = [audit_case(case) for case in discover_cases(data_root)]
    rows = [audit.as_dict() for audit in audits]
    write_json(output / "cases.json", rows)
    write_json(output / "summary.json", summarize_audits(audits))
    _write_csv(output / "cases.csv", rows)
    print(f"Audited {len(audits)} labeled cases into {output}")
    return 0


def command_split(args: argparse.Namespace) -> int:
    data_root = _resolve(args.data)
    audits = [audit_case(case) for case in discover_cases(data_root)]
    destination = write_json(_resolve(args.output), make_folds(audits, args.folds, args.seed))
    print(f"Wrote {args.folds} patient-level folds to {destination}")
    return 0


def _run_directory(config: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return _resolve(override)
    experiment = config["experiment"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _resolve(Path(experiment["output_root"])) / f"{experiment['name']}_{timestamp}"


def command_train(args: argparse.Namespace) -> int:
    config = _load_configs(args)
    spacing, lower_hu, upper_hu, patch_size = _required_preprocessing(config)
    data = config["data"]
    experiment = config["experiment"]
    seed_everything(int(experiment["seed"]), bool(experiment.get("deterministic", True)))
    cases = discover_cases(_resolve(Path(data["root"])))
    train_cases, validation_cases = _load_split(args.split, int(experiment["fold"]), cases)

    train_transform = training_transforms(
        spacing,
        lower_hu,
        upper_hu,
        patch_size,
        samples_per_volume=int(data.get("samples_per_volume", 2)),
        positive_weight=float(data.get("positive_patch_weight", 1.0)),
        negative_weight=float(data.get("negative_patch_weight", 1.0)),
        orientation=str(data.get("canonical_orientation", "RAS")),
    )
    validation_transform = deterministic_transforms(
        spacing, lower_hu, upper_hu, str(data.get("canonical_orientation", "RAS"))
    )
    train_loader = make_loader(
        train_cases,
        train_transform,
        batch_size=int(experiment["batch_size"]),
        shuffle=True,
        num_workers=int(experiment["num_workers"]),
    )
    validation_loader = make_loader(
        validation_cases, validation_transform, batch_size=1, shuffle=False, num_workers=0
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    loss_function = build_loss()
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
    )
    output = _run_directory(config, args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(output / "environment.json", collect_environment(ROOT))
    shutil.copy2(_resolve(args.split), output / "split.json")

    start_epoch = 0
    best_dice = -1.0
    if args.resume:
        checkpoint = torch.load(_resolve(args.resume), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint.get("metrics", {}).get("mean_dice", -1.0))

    history: list[dict[str, Any]] = []
    epochs = int(experiment["epochs"])
    amp = "16" in str(experiment.get("precision", "32"))
    for epoch in range(start_epoch, epochs):
        loss = train_epoch(model, train_loader, loss_function, optimizer, device, amp=amp)
        rows = validate_cases(
            model,
            validation_loader,
            device,
            patch_size,
            overlap=float(config["inference"].get("overlap", 0.5)),
        )
        mean_dice = float(np.mean([row["dice"] for row in rows]))
        record = {"epoch": epoch, "train_loss": loss, "mean_validation_dice": mean_dice}
        history.append(record)
        _write_csv(output / "history.csv", history)
        _write_csv(output / "case_metrics.csv", rows)
        save_checkpoint(output / "checkpoints/last.pt", model, optimizer, epoch, {"mean_dice": mean_dice})
        if mean_dice > best_dice:
            best_dice = mean_dice
            save_checkpoint(output / "checkpoints/best.pt", model, optimizer, epoch, {"mean_dice": mean_dice})
        print(f"epoch={epoch + 1}/{epochs} loss={loss:.6f} val_dice={mean_dice:.6f}")
    print(f"Training complete: {output}")
    return 0


def _load_model_checkpoint(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(config).to(device)
    checkpoint = torch.load(_resolve(checkpoint_path), map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def command_evaluate(args: argparse.Namespace) -> int:
    config = _load_configs(args)
    spacing, lower_hu, upper_hu, patch_size = _required_preprocessing(config)
    data = config["data"]
    cases = discover_cases(_resolve(Path(data["root"])))
    _, validation_cases = _load_split(args.split, int(args.fold), cases)
    transform = deterministic_transforms(
        spacing, lower_hu, upper_hu, str(data.get("canonical_orientation", "RAS"))
    )
    loader = make_loader(validation_cases, transform, batch_size=1, num_workers=0)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model_checkpoint(config, args.checkpoint, device)
    rows = validate_cases(model, loader, device, patch_size, overlap=float(config["inference"]["overlap"]))
    output = _resolve(args.output)
    _write_csv(output, rows)
    summary = {
        "case_count": len(rows),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "median_dice": float(np.median([row["dice"] for row in rows])),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))
    return 0


def command_infer(args: argparse.Namespace) -> int:
    config = _load_configs(args)
    spacing, lower_hu, upper_hu, patch_size = _required_preprocessing(config)
    orientation = str(config["data"].get("canonical_orientation", "RAS"))
    preprocess = Compose(
        [
            LoadImaged(keys="image", image_only=False),
            EnsureChannelFirstd(keys="image"),
            Orientationd(
                keys="image", axcodes=orientation,
                labels=(("L", "R"), ("P", "A"), ("I", "S")),
            ),
            Spacingd(keys="image", pixdim=spacing, mode="bilinear"),
            ScaleIntensityRanged(
                keys="image", a_min=lower_hu, a_max=upper_hu, b_min=0.0, b_max=1.0, clip=True
            ),
            EnsureTyped(keys="image", track_meta=True),
        ]
    )
    item = preprocess({"image": str(_resolve(args.image))})
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _load_model_checkpoint(config, args.checkpoint, device)
    image = item["image"].unsqueeze(0).to(device=device, dtype=torch.float32)
    prediction = predict_volume(
        model,
        image,
        patch_size,
        overlap=float(config["inference"]["overlap"]),
        sliding_window_batch_size=int(config["inference"].get("sliding_window_batch_size", 1)),
    )[0]
    item["pred"] = prediction.cpu()
    inverted = Invertd(
        keys="pred",
        transform=preprocess,
        orig_keys="image",
        nearest_interp=True,
        to_tensor=True,
    )(item)
    array = np.asarray(inverted["pred"]).squeeze().astype(np.uint8)
    reference = nib.load(_resolve(args.image))
    if array.shape != reference.shape:
        raise RuntimeError(f"Inverted prediction shape {array.shape} != source shape {reference.shape}")
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(array, reference.affine, header), output)
    print(f"Saved original-geometry prediction to {output}")
    return 0


def command_test(args: argparse.Namespace) -> int:
    command = [sys.executable, "-m", "pytest"]
    if args.quiet:
        command.append("-q")
    command.extend(args.pytest_args)
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + existing if existing else "")
    return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Manage the 3D lung-tumor segmentation research pipeline.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    environment = commands.add_parser("environment", help="Report runtime, GPU, and Git provenance")
    environment.add_argument("--output", type=Path)
    environment.set_defaults(handler=command_environment)

    status = commands.add_parser("status", help="Show dataset and training readiness")
    status.add_argument("--data-config", type=Path, default=DEFAULT_DATA_CONFIG)
    status.set_defaults(handler=command_status)

    inspect = commands.add_parser("inspect", help="Audit labeled NIfTI cases")
    inspect.add_argument("--data", type=Path, default=Path("data/raw/Task06_Lung"))
    inspect.add_argument("--output", type=Path, default=Path("artifacts/audits/msd_lung"))
    inspect.set_defaults(handler=command_inspect)

    split = commands.add_parser("split", help="Generate deterministic patient-level folds")
    split.add_argument("--data", type=Path, default=Path("data/raw/Task06_Lung"))
    split.add_argument("--output", type=Path, default=Path("artifacts/splits/msd_lung_5fold_seed43.json"))
    split.add_argument("--folds", type=int, default=5)
    split.add_argument("--seed", type=int, default=43)
    split.set_defaults(handler=command_split)

    for name in ("train", "run"):
        train = commands.add_parser(name, help="Train one configured fold" if name == "train" else "Alias for train")
        _add_config_arguments(train)
        train.add_argument("--split", type=Path, default=Path("artifacts/splits/msd_lung_5fold_seed43.json"))
        train.add_argument("--output", type=Path)
        train.add_argument("--resume", type=Path)
        train.add_argument("--device", choices=("cpu", "cuda"))
        train.set_defaults(handler=command_train)

    evaluate = commands.add_parser("evaluate", help="Evaluate a checkpoint on one fold")
    _add_config_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", type=Path, default=Path("artifacts/splits/msd_lung_5fold_seed43.json"))
    evaluate.add_argument("--fold", type=int, default=0)
    evaluate.add_argument("--output", type=Path, default=Path("artifacts/evaluation/case_metrics.csv"))
    evaluate.add_argument("--device", choices=("cpu", "cuda"))
    evaluate.set_defaults(handler=command_evaluate)

    infer = commands.add_parser("infer", help="Run original-geometry NIfTI inference")
    _add_config_arguments(infer)
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--image", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--device", choices=("cpu", "cuda"))
    infer.set_defaults(handler=command_infer)

    test = commands.add_parser("test", help="Run the project test suite")
    test.add_argument("--quiet", action="store_true")
    test.add_argument("pytest_args", nargs=argparse.REMAINDER)
    test.set_defaults(handler=command_test)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
