"""Events emitted by the monitor loop.

The loop knows nothing about Discord, email, or a web UI — it describes what
happened and hands the event to a :class:`~src.notify.base.Notifier`.
Adding a channel means implementing the notifier, not editing the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from src.detection.results import DetectionResult
from src.printer.base import PrintState


@dataclass(frozen=True)
class Event:
    """Base class for everything the monitor emits."""


@dataclass(frozen=True)
class MonitoringStarted(Event):
    """The loop has come up and is about to take its first frame."""

    target_loop_time: float
    printer_url: str
    model_name: str


@dataclass(frozen=True)
class MonitoringStopped(Event):
    """The loop is shutting down, cleanly or otherwise."""

    reason: str
    uptime_seconds: float


@dataclass(frozen=True)
class StatusUpdate(Event):
    """A routine heartbeat with the current camera view."""

    uptime_seconds: float
    state: PrintState
    image: Image.Image | None = None
    detail: str = ""
    captured_at: float | None = None


@dataclass(frozen=True)
class DebugDetection(Event):
    """A detection produced while idle debug mode is enabled.

    Debug detections are for the local web UI only: they never pause the
    printer, notify Discord, or wait for operator acknowledgement.
    """

    result: DetectionResult
    state: PrintState
    uptime_seconds: float
    captured_at: float | None = None


@dataclass(frozen=True)
class SpaghettiDetected(Event):
    """A print failure was detected.

    ``paused`` records what actually happened, not what was configured — a
    pause request can fail.
    """

    result: DetectionResult
    annotated_image: Image.Image | None
    paused: bool
    pause_requested: bool
    saved_frame: str = ""
    captured_at: float | None = None


@dataclass(frozen=True)
class ImageUnavailable(Event):
    """A camera frame could not be retrieved."""

    reason: str
    consecutive_failures: int


@dataclass(frozen=True)
class MonitorError(Event):
    """An unexpected error occurred; the loop caught it and will continue."""

    message: str
    consecutive_failures: int


__all__ = [
    "DebugDetection",
    "Event",
    "ImageUnavailable",
    "MonitorError",
    "MonitoringStarted",
    "MonitoringStopped",
    "SpaghettiDetected",
    "StatusUpdate",
]
