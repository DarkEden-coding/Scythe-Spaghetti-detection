"""Exception hierarchy for Scythe.

Every failure mode the application knows how to talk about derives from
:class:`ScytheError`, so callers can distinguish "something we anticipated"
from a genuine bug.
"""


class ScytheError(Exception):
    """Base class for all Scythe errors."""


class ConfigError(ScytheError):
    """Configuration is missing, malformed, or invalid."""


class PrinterError(ScytheError):
    """Base class for printer communication problems."""


class PrinterUnavailable(PrinterError):
    """The printer could not be reached, or returned an unusable response."""


class WebcamNotFound(PrinterError):
    """The configured webcam name does not exist on the printer."""

    def __init__(self, requested: str, available: list[str]) -> None:
        self.requested = requested
        self.available = available
        names = ", ".join(available) or "<none>"
        super().__init__(
            f"Webcam {requested!r} not found (names are case sensitive). "
            f"Available webcams: {names}"
        )


class DetectorError(ScytheError):
    """The detection model could not be loaded or run."""


class NotifierError(ScytheError):
    """A notification channel failed in a way the loop should know about."""
