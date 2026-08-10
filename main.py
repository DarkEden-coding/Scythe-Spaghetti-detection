#!/usr/bin/env python3
"""Compatibility shim for ``python3 main.py``.

The application now lives in the ``src`` package. Prefer::

    scythe run          # installed console script
    python -m src    # without installing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["run"]))
