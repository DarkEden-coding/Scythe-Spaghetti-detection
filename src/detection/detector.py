"""Local spaghetti detector: the bundled YOLO model, run with ONNX Runtime.

The runtime needs only ``onnxruntime`` and ``numpy``. PyTorch and Ultralytics
are training tools (``pip install -e .[train]``); a printer install no longer
downloads a gigabyte of them to run one exported graph.

Constructing the detector is free; :meth:`SpaghettiDetector.load` does the
expensive work at a moment the caller chooses.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from src.config import DetectionSettings
from src.detection.results import DetectionBox, DetectionResult
from src.errors import DetectorError

log = logging.getLogger(__name__)

#: Ultralytics' default NMS overlap.
IOU_THRESHOLD = 0.7
#: Boxes kept in ``considered`` for threshold tuning, below the alert threshold.
CONSIDER_CONFIDENCE = 0.25


class SpaghettiDetector:
    """Runs the fail-detection model over camera frames.

    ``model`` is anything callable as ``model(input_array) -> output_array``
    with YOLOv8's shapes: input ``(1, 3, S, S)`` floats in 0..1, output
    ``(1, 4 + classes, anchors)``. Tests pass a fake; :meth:`load` builds an
    ONNX Runtime session.
    """

    def __init__(self, settings: DetectionSettings, model=None):
        self._settings = settings
        self._model = model
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> Path:
        return self._settings.model_path

    @property
    def model_name(self) -> str:
        return self.model_path.name

    # -- lifecycle --------------------------------------------------------- #

    def load(self) -> None:
        """Load the model, downloading it first if it is missing.

        Idempotent and safe to call from any thread.
        """
        with self._lock:
            if self._model is not None:
                return

            path = self.model_path
            if not path.exists():
                from src.detection.model_download import ensure_model

                ensure_model(path)

            try:
                import onnxruntime
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise DetectorError(
                    "onnxruntime is required to run the local model. Install it with "
                    "`pip install -r requirements.txt`."
                ) from exc

            providers = ["CPUExecutionProvider"]
            if self._settings.use_cuda:
                providers.insert(0, "CUDAExecutionProvider")
            log.info("Loading model from %s...", path)
            try:
                session = onnxruntime.InferenceSession(str(path), providers=providers)
            except Exception as exc:
                raise DetectorError(f"Failed to load model {path}: {exc}") from exc
            input_name = session.get_inputs()[0].name
            self._model = lambda array: session.run(None, {input_name: array})[0]
            log.info("Model loaded.")

    # -- inference --------------------------------------------------------- #

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Match the preprocessing the model was trained with."""
        size = self._settings.image_size
        return image.convert("L").resize((size, size))

    def detect(self, image: Image.Image) -> DetectionResult:
        """Run inference over a single frame.

        Blocking and CPU bound — callers on an event loop should dispatch this
        to a worker thread.
        """
        if self._model is None:
            self.load()

        settings = self._settings
        started = perf_counter()
        frame = self.preprocess(image)
        gray = np.asarray(frame, dtype=np.float32) / 255.0
        array = np.broadcast_to(gray, (1, 3, *gray.shape)).copy()

        try:
            output = np.asarray(self._model(array))
        except Exception as exc:
            raise DetectorError(f"Inference failed: {exc}") from exc

        x_scale = image.width / frame.width
        y_scale = image.height / frame.height
        considered = tuple(
            self._iter_boxes(output, x_scale, y_scale, image.width, image.height)
        )
        matches = tuple(
            box
            for box in considered
            if box.class_id == settings.spaghetti_class_id
            and box.confidence >= settings.min_confidence
        )

        result = DetectionResult(
            boxes=matches,
            image=image.copy(),
            duration_seconds=perf_counter() - started,
            considered=considered,
        )
        log.debug("Detection: %s", result.summary())
        return result

    def _iter_boxes(
        self,
        output: Any,
        x_scale: float,
        y_scale: float,
        width: int,
        height: int,
    ) -> Iterator[DetectionBox]:
        """Decode YOLOv8 output, apply per-class NMS, and map onto the frame."""
        rows = output[0].T  # (anchors, 4 + classes)
        scores = rows[:, 4:]
        class_ids = scores.argmax(axis=1)
        confidences = scores.max(axis=1)
        floor = min(CONSIDER_CONFIDENCE, self._settings.min_confidence)
        keep = confidences >= floor
        rows, class_ids, confidences = rows[keep], class_ids[keep], confidences[keep]

        cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

        for index in _nms(xyxy, confidences, class_ids, IOU_THRESHOLD):
            x1, y1, x2, y2 = xyxy[index]
            yield DetectionBox(
                x1=max(0, min(width, round(float(x1) * x_scale))),
                y1=max(0, min(height, round(float(y1) * y_scale))),
                x2=max(0, min(width, round(float(x2) * x_scale))),
                y2=max(0, min(height, round(float(y2) * y_scale))),
                confidence=round(float(confidences[index]), 4),
                class_id=int(class_ids[index]),
            )


def _nms(boxes: Any, scores: Any, classes: Any, iou: float) -> list[int]:
    """Greedy non-maximum suppression within each class, best first."""
    kept: list[int] = []
    for index in np.argsort(-scores):
        x1, y1, x2, y2 = boxes[index]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        suppressed = False
        for other in kept:
            if classes[other] != classes[index]:
                continue
            ox1, oy1, ox2, oy2 = boxes[other]
            inter = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(
                0.0, min(y2, oy2) - max(y1, oy1)
            )
            other_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
            union = area + other_area - inter
            if union > 0 and inter / union > iou:
                suppressed = True
                break
        if not suppressed:
            kept.append(int(index))
    return kept
