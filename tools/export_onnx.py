"""Export a ``.pt`` checkpoint to ONNX.

Usage::

    python -m tools.export_onnx models/largeModel.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.detection.export import export_onnx
from src.errors import DetectorError
from src.logging_setup import configure_logging
from src.paths import DEFAULT_MODEL


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "weights",
        nargs="?",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Checkpoint to export (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size (default: 640)."
    )
    args = parser.parse_args(argv)

    configure_logging("INFO", console=True)
    try:
        export_onnx(args.weights, args.imgsz)
    except DetectorError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
