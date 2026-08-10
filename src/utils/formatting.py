"""Human-readable formatting helpers."""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Render a duration as ``2 days, 3 hours, 4 minutes, 5 seconds``.

    Leading zero-valued units are dropped, so a fresh start reads
    ``12 seconds`` rather than ``0 days, 0 hours, 0 minutes, 12 seconds``.
    """
    remaining = max(0, int(seconds))
    units = (("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1))

    parts: list[str] = []
    for name, size in units:
        value, remaining = divmod(remaining, size)
        if value or parts or name == "second":
            parts.append(f"{value} {name}{'' if value == 1 else 's'}")
    return ", ".join(parts)
