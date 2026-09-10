#!/usr/bin/env python3
"""Single command-line entry point for the lung-segmentation project."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from lungseg.app import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
