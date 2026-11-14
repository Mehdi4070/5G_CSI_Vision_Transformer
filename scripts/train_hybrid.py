#!/usr/bin/env python3
"""Launch the shared transformer trainer with the hybrid configuration."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_vit import main  # noqa: E402


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv[1:1] = ["--config", str(PROJECT_ROOT / "configs" / "hybrid_baseline.yaml")]
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
