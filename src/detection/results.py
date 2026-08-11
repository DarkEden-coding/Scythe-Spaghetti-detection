"""Value objects returned by the detector.

These carry no PIL/Torch behaviour of their own, which keeps them cheap to
construct in tests and safe to log.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image


@dataclass(frozen=True)
class DetectionBox:
    """One detected region, in pixels of the preprocessed image."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of a single inference pass.

    ``image`` is the original camera frame and boxes use that frame's pixel
    coordinates. The detector handles mapping from its square model input.
    """

    boxes: tuple[DetectionBox, ...] = ()
    image: Image.Image | None = None
    duration_seconds: float = 0.0
    #: Every box the model returned, including classes we ignore — useful when
    #: tuning thresholds from the logs.
    considered: tuple[DetectionBox, ...] = field(default=(), repr=False)

    def __bool__(self) -> bool:
        """True when spaghetti was found.

        Replaces the old ``list | False`` return, which only worked because
        every call site happened to treat it as a boolean.
        """
        return bool(self.boxes)

    @property
    def count(self) -> int:
        return len(self.boxes)

    @property
    def max_confidence(self) -> float:
        return max((b.confidence for b in self.boxes), default=0.0)

    def summary(self) -> str:
        if not self.boxes:
            return f"no spaghetti ({self.duration_seconds:.2f}s)"
        return (
            f"{self.count} detection(s), max confidence "
            f"{self.max_confidence:.2f} ({self.duration_seconds:.2f}s)"
        )
