"""The interface the monitor loop depends on.

The loop is written against :class:`PrinterClient`, not against Moonraker, so
support for another firmware (OctoPrint, Duet, …) means adding a module here
rather than editing the loop.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

from PIL import Image


class PrintState(str, Enum):
    """Normalised print state, independent of any firmware's vocabulary."""

    STANDBY = "standby"
    PRINTING = "printing"
    PAUSED = "paused"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    ERROR = "error"
    UNKNOWN = "unknown"

    @property
    def is_active(self) -> bool:
        """True when a print is in progress and worth inspecting."""
        return self is PrintState.PRINTING

    @classmethod
    def parse(cls, raw: str) -> PrintState:
        try:
            return cls(str(raw).strip().lower())
        except ValueError:
            return cls.UNKNOWN


@runtime_checkable
class PrinterClient(Protocol):
    """Everything the monitor needs from a printer.

    All methods are synchronous and may block; the monitor runs them off the
    event loop.
    """

    def get_snapshot(self) -> Image.Image | None:
        """Return the current camera frame, or ``None`` if unavailable."""
        ...

    def get_state(self) -> PrintState:
        """Return the current print state."""
        ...

    def is_printing(self) -> bool:
        """Convenience wrapper around :meth:`get_state`."""
        ...

    def pause(self) -> bool:
        """Pause the running print. Returns True if a pause was issued."""
        ...

    def resume(self) -> bool:
        """Resume a paused print. Returns True if a resume was issued."""
        ...

    def close(self) -> None:
        """Release any held resources."""
        ...
