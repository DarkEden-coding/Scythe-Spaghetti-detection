"""Spaghetti detection: model loading, inference, and annotation."""

from src.detection.annotate import draw_boxes
from src.detection.detector import SpaghettiDetector
from src.detection.results import DetectionBox, DetectionResult

__all__ = [
    "DetectionBox",
    "DetectionResult",
    "SpaghettiDetector",
    "draw_boxes",
]
