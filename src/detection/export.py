"""ONNX export.

Kept beside the detector rather than in the training tools: the runtime needs
it when ``use_onnx`` is set and no ``.onnx`` file has been produced yet, and a
runtime package should not import from a dev-only one.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.errors import DetectorError

log = logging.getLogger(__name__)


def export_onnx(weights: Path, image_size: int = 640) -> Path:
    """Export ``weights`` (a ``.pt`` file) to ONNX beside itself.

    Returns the path to the exported model.
    """
    weights = Path(weights)
    if not weights.exists():
        raise DetectorError(f"Cannot export: weights not found at {weights}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DetectorError(
            "ultralytics is required to export a model. Install it with "
            "`pip install -r requirements.txt`."
        ) from exc

    log.info("Exporting %s to ONNX (imgsz=%d)...", weights, image_size)
    try:
        # CPU export keeps the graph portable across the machines people
        # actually run this on (Pi, mini PC, co-processor).
        exported = YOLO(str(weights)).export(
            format="onnx", imgsz=image_size, device="cpu"
        )
    except Exception as exc:  # ultralytics raises a wide variety of types
        raise DetectorError(f"ONNX export failed: {exc}") from exc

    result = Path(exported) if exported else weights.with_suffix(".onnx")
    if not result.exists():
        raise DetectorError(f"ONNX export reported success but {result} is missing.")

    log.info("Exported ONNX model to %s", result)
    return result
