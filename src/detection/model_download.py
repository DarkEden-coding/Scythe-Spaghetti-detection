"""Fetch the bundled ONNX model.

The exported model is 175 MB, over GitHub's 100 MB file limit, so it lives as
an asset on a dedicated GitHub release rather than in git. The installer calls
:func:`ensure_model`, and the detector calls it again if the file goes missing.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import requests

from src.errors import DetectorError

log = logging.getLogger(__name__)

MODEL_URL = (
    "https://github.com/DarkEden-coding/Scythe-Spaghetti-detection/releases/download/"
    "model-v1/largeModel.onnx"
)
MODEL_SHA256 = "cecf6922c556db402b37cd8a84d79c9678256f6e2c76ee6d4167dcf39f07e532"


def ensure_model(path: Path, url: str = MODEL_URL, sha256: str = MODEL_SHA256) -> Path:
    """Download the model to ``path`` unless it is already there."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    digest = hashlib.sha256()
    log.info("Downloading the detection model from %s ...", url)
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
                    digest.update(chunk)
    except requests.RequestException as exc:
        partial.unlink(missing_ok=True)
        raise DetectorError(f"Could not download the model from {url}: {exc}") from exc

    if sha256 and digest.hexdigest() != sha256:
        partial.unlink(missing_ok=True)
        raise DetectorError("The downloaded model is corrupt (checksum mismatch).")
    partial.replace(path)
    log.info("Model saved to %s", path)
    return path
