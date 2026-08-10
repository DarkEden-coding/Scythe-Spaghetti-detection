"""YOLO-backed spaghetti detector.

Constructing the detector is free; :meth:`SpaghettiDetector.load` does the
expensive work (and any ONNX export) at a moment the caller chooses. Nothing
happens at import, so this module can be imported in tests without Torch
present.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from time import perf_counter

from PIL import Image

from src.config import DetectionSettings
from src.detection.export import export_onnx
from src.detection.results import DetectionBox, DetectionResult
from src.errors import DetectorError

log = logging.getLogger(__name__)


class SpaghettiDetector:
    """Runs the fail-detection model over camera frames.

    The detector never touches the filesystem for its results and never draws
    on the image — it returns a :class:`DetectionResult` and lets the caller
    decide what to persist or send. The old implementation wrote
    ``fail_img.jpg`` on every call and the loop re-read it from disk, coupling
    two modules through a file.
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
        return self._settings.active_model_path

    # -- lifecycle --------------------------------------------------------- #

    def load(self) -> None:
        """Load the model, exporting to ONNX first if required.

        Idempotent and safe to call from any thread.
        """
        with self._lock:
            if self._model is not None:
                return

            settings = self._settings
            if settings.use_onnx and not settings.onnx_path.exists():
                log.info("No ONNX model at %s; building one.", settings.onnx_path)
                export_onnx(settings.model_path, settings.image_size)

            path = settings.active_model_path
            if not path.exists():
                raise DetectorError(f"Model file not found: {path}")

            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise DetectorError(
                    "ultralytics is required to run detection. Install it with "
                    "`pip install -r requirements.txt`."
                ) from exc

            log.info("Loading YOLO model from %s (device=%s)...", path, settings.device)
            try:
                self._model = YOLO(str(path), task="detect")
            except Exception as exc:
                raise DetectorError(f"Failed to load model {path}: {exc}") from exc
            log.info("Model loaded.")

    # -- inference --------------------------------------------------------- #

    def preprocess(self, image: Image.Image) -> Image.Image:
        """Match the preprocessing the model was trained with."""
        size = self._settings.image_size
        return image.convert("L").resize((size, size))

    def detect(self, image: Image.Image) -> DetectionResult:
        """Run inference over a single frame.

        Blocking and CPU/GPU bound — callers on an event loop should dispatch
        this to a worker thread.
        """
        if self._model is None:
            self.load()

        settings = self._settings
        started = perf_counter()
        frame = self.preprocess(image)

        try:
            raw_results = self._model(
                source=frame,
                save=False,
                save_conf=False,
                show=False,
                conf=settings.min_confidence,
                device=settings.device,
                verbose=False,
            )
        except Exception as exc:
            raise DetectorError(f"Inference failed: {exc}") from exc

        considered = tuple(self._iter_boxes(raw_results))
        matches = tuple(
            box
            for box in considered
            if box.class_id == settings.spaghetti_class_id
            and box.confidence >= settings.min_confidence
        )

        result = DetectionResult(
            boxes=matches,
            image=frame,
            duration_seconds=perf_counter() - started,
            considered=considered,
        )
        log.debug("Detection: %s", result.summary())
        return result

    @staticmethod
    def _iter_boxes(raw_results):
        """Flatten Ultralytics' nested result objects into value objects."""
        for result in raw_results:
            for box in getattr(result, "boxes", []) or []:
                xyxy = box.xyxy[0]
                yield DetectionBox(
                    x1=int(xyxy[0].item()),
                    y1=int(xyxy[1].item()),
                    x2=int(xyxy[2].item()),
                    y2=int(xyxy[3].item()),
                    confidence=round(float(box.conf[0].item()), 4),
                    class_id=int(box.cls[0].item()),
                )
