"""Train a fail-detection model.

Every knob that used to be a "change this line" comment is now a flag::

    python -m tools.train --data "datasets/3d-printing-fail-detection/data.yaml" \\
        --weights models/largeModel.pt --name spaghetti-detection-L15
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.logging_setup import configure_logging
from src.paths import DEFAULT_MODEL

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", required=True, type=Path, help="Path to the dataset data.yaml."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Starting checkpoint (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument("--name", required=True, help="Run name for this training job.")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch", type=int, default=-1, help="-1 selects the largest batch that fits."
    )
    parser.add_argument(
        "--device",
        default="0",
        help="GPU index, a comma-separated list, or 'cpu' (default: 0).",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="Do not cache the dataset in RAM."
    )
    args = parser.parse_args(argv)

    configure_logging("INFO", console=True)

    if not args.data.exists():
        log.error("Dataset not found: %s", args.data)
        return 1

    from ultralytics import YOLO

    device = (
        args.device if args.device == "cpu" else [int(d) for d in args.device.split(",")]
    )

    log.info("Loading %s", args.weights)
    model = YOLO(str(args.weights))

    log.info("Training %r for %d epochs on device %s", args.name, args.epochs, device)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        cache=not args.no_cache,
        device=device,
    )
    log.info("Training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
