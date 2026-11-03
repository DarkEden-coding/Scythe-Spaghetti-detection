"""Filesystem locations shared by every pipeline stage (see CONTRACTS.md §1)."""

import os
from pathlib import Path

#: Root of all generated data. Large; lives on /mnt/Primary.
DATA_ROOT = Path(os.environ.get("SCYTHE_SYNTH_ROOT", "/mnt/Primary/scythe-synth"))
CORPUS_DIR = DATA_ROOT / "corpus"
GCODE_CACHE = Path(os.environ.get("SCYTHE_GCODE_CACHE", str(DATA_ROOT / "gcode_cache")))
RUNS_DIR = DATA_ROOT / "runs"

#: Repository locations.
PIPELINE_DIR = Path(__file__).resolve().parents[1]
SYNTH_DIR = PIPELINE_DIR.parent
BLENDER_DIR = SYNTH_DIR / "blender"
REPO_ROOT = SYNTH_DIR.parent
REAL_DATASETS = REPO_ROOT / "datasets"

BLENDER_BIN = Path(os.environ.get(
    "SCYTHE_BLENDER", str(Path.home() / ".local/opt/blender-5.2.1-linux-x64/blender")))
