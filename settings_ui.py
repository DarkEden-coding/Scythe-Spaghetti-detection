#!/usr/bin/env python3
"""Compatibility shim for ``python3 settings_ui.py``.

Configuration now lives in ``settings.json``. Prefer::

    scythe configure
    python -m src configure
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["configure"]))
