"""Filesystem locations, resolved relative to the installation.

Nothing in Scythe depends on the current working directory. Every path used at
runtime is derived from this module, so ``python -m src`` behaves the same
whether it is launched from the repo root, from ``/``, or by systemd.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Repository / installation root (the directory containing the ``src`` package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Bundled model weights.
MODELS_DIR = PROJECT_ROOT / "models"

#: Default weights shipped with the project.
DEFAULT_MODEL = MODELS_DIR / "largeModel.pt"

#: Writable location for runtime artifacts (annotated frames, logs).
DATA_DIR = PROJECT_ROOT / "data"

#: Primary config file.
CONFIG_PATH = PROJECT_ROOT / "settings.json"

#: Pre-0.2 config format, read once for migration then left alone.
LEGACY_CONFIG_PATH = PROJECT_ROOT / "settings.py"

#: Default log file, served by the ``/get_log_file`` command.
LOG_FILE = DATA_DIR / "src.log"


def config_path() -> Path:
    """Return the config path, honouring the ``SCYTHE_CONFIG`` override."""
    override = os.environ.get("SCYTHE_CONFIG")
    return Path(override).expanduser().resolve() if override else CONFIG_PATH


def ensure_data_dir() -> Path:
    """Create and return the runtime data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
