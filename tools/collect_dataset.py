"""Capture camera frames for labelling, optionally uploading to Roboflow.

    export ROBOFLOW_API_KEY=...          # only needed with --upload
    python -m tools.collect_dataset --interval 5 --output data/training

Ctrl-C stops the capture. The previous version hardcoded an API key in the
source, ran its loop at import time, and used the ``keyboard`` package to watch
for a keypress — which requires root on Linux.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_settings
from src.errors import ConfigError, ScytheError
from src.logging_setup import configure_logging
from src.printer.moonraker import MoonrakerClient

log = logging.getLogger(__name__)

ROBOFLOW_PROJECT = "3d-printing-fail-detection"


def _roboflow_project(project_name: str):
    """Connect to Roboflow using ``ROBOFLOW_API_KEY``."""
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ConfigError("ROBOFLOW_API_KEY is not set. Export it before using --upload.")
    from roboflow import Roboflow

    return Roboflow(api_key=api_key).project(project_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/training"), help="Where to save frames."
    )
    parser.add_argument(
        "--interval", type=float, default=5.0, help="Seconds between frames."
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Stop after N frames (0 = unlimited)."
    )
    parser.add_argument(
        "--upload", action="store_true", help="Also upload each frame to Roboflow."
    )
    parser.add_argument(
        "--project", default=ROBOFLOW_PROJECT, help="Roboflow project id."
    )
    args = parser.parse_args(argv)

    configure_logging("INFO", console=True)

    try:
        settings = load_settings(validate=False)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    project = _roboflow_project(args.project) if args.upload else None
    args.output.mkdir(parents=True, exist_ok=True)

    captured = 0
    log.info("Capturing every %.1fs to %s. Ctrl-C to stop.", args.interval, args.output)

    with MoonrakerClient(settings.printer) as printer:
        try:
            while not args.limit or captured < args.limit:
                try:
                    image = printer.get_snapshot()
                except ScytheError as exc:
                    log.error("%s", exc)
                    return 1

                if image is None:
                    log.warning("No frame available; retrying.")
                else:
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    target = args.output / f"{stamp}.jpg"
                    image.convert("RGB").save(target, quality=90)
                    captured += 1
                    log.info("Saved %s (%d captured)", target.name, captured)

                    if project is not None:
                        try:
                            project.upload(str(target))
                        except Exception:  # noqa: BLE001 - upload is best effort
                            log.exception("Roboflow upload failed for %s", target.name)

                time.sleep(args.interval)
        except KeyboardInterrupt:
            log.info("Stopped after %d frame(s).", captured)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
