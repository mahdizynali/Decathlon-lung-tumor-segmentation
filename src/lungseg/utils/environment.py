"""Capture enough runtime context to interpret an experiment artifact."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from lungseg.utils.io import write_json


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_output(repo_root: Path, *arguments: str) -> str | None:
    try:
        process = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return process.stdout.strip()


def collect_environment(repo_root: Path | None = None) -> dict[str, Any]:
    """Return software, hardware, and repository provenance information."""
    root = Path(repo_root or Path.cwd()).expanduser().resolve()
    report: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            name: _package_version(distribution)
            for name, distribution in {
                "lungseg": "lungseg",
                "monai": "monai",
                "nibabel": "nibabel",
                "numpy": "numpy",
                "pandas": "pandas",
                "scipy": "scipy",
                "torch": "torch",
            }.items()
        },
        "git": {
            "root": str(root),
            "commit": _git_output(root, "rev-parse", "HEAD"),
            "branch": _git_output(root, "branch", "--show-current"),
            "dirty": None,
        },
        "cuda": {
            "available": False,
            "runtime_version": None,
            "cudnn_version": None,
            "device_count": 0,
            "devices": [],
        },
    }

    porcelain = _git_output(root, "status", "--porcelain", "--untracked-files=normal")
    if porcelain is not None:
        report["git"]["dirty"] = bool(porcelain)

    try:
        import torch
    except ImportError:
        return report

    available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if available else 0
    report["cuda"] = {
        "available": available,
        "runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": device_count,
        "devices": [torch.cuda.get_device_name(index) for index in range(device_count)],
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON destination. The report is always printed.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository used to capture Git state (default: current directory).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = collect_environment(args.repo_root)
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
