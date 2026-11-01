"""A detector that can be replaced while the monitor is running.

The dashboard's settings panel can change the backend, model or crop. Rather
than restarting the process, the new detector is built and loaded off to the
side and then swapped in; a detection already in flight finishes on the old one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PIL import Image

from src.config import DetectionSettings
from src.detection.results import DetectionResult


class SwitchableDetector:
    def __init__(
        self, factory: Callable[[DetectionSettings], Any], settings: DetectionSettings
    ):
        self._factory = factory
        self._current = factory(settings)

    @property
    def current(self) -> Any:
        return self._current

    @property
    def model_name(self) -> str:
        return self._current.model_name

    @property
    def average_cost(self) -> float | None:
        """Measured cost per request, for backends that report one."""
        return getattr(self._current, "average_cost", None)

    def build(self, settings: DetectionSettings) -> Any:
        """Create and load a detector for ``settings`` without swapping it in.

        Blocking (YOLO loads a model), so call it from a worker thread.
        """
        detector = self._factory(settings)
        detector.load()
        return detector

    def replace(self, detector: Any) -> None:
        self._current = detector

    def load(self) -> None:
        self._current.load()

    def detect(self, image: Image.Image) -> DetectionResult:
        return self._current.detect(image)
