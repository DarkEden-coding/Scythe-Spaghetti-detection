"""Runtime-editable detection settings, for the dashboard's settings panel.

Only detection and pacing settings are editable here. Each change is validated,
applied to the running monitor (swapping the detector if needed), and merged
into the config file so it survives a restart.
"""

from __future__ import annotations

import asyncio
import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.config import (
    DECISION_PROVIDERS,
    ConfigError,
    Settings,
    env_var_names,
    parse_bool,
    save_overrides,
)
from src.detection import pricing
from src.detection.switchable import SwitchableDetector

#: Dashboard field -> (dotted setting path, type).
FIELDS: dict[str, tuple[str, type]] = {
    "capture_interval": ("target_loop_time", float),
    "detection_interval": ("detection.interval", float),
    "backend": ("detection.backend", str),
    "provider": ("detection.decision.provider", str),
    "base_url": ("detection.decision.base_url", str),
    "api_key": ("detection.decision.api_key", str),
    "model": ("detection.decision.model", str),
    "threshold": ("detection.decision.threshold", float),
    "crop": ("detection.decision.crop", str),
    "image_max_side": ("detection.decision.image_max_side", int),
    "yolo_confidence": ("detection.min_confidence", float),
    "pause_on_failure": ("pause_on_spaghetti", bool),
}

#: Below this the camera and the API are polled faster than is useful.
MIN_INTERVAL = 5.0


def _get(settings: Settings, dotted: str) -> Any:
    target: Any = settings
    for part in dotted.split("."):
        target = getattr(target, part)
    return target


def _set(settings: Settings, dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    target: Any = settings
    for part in parents:
        target = getattr(target, part)
    setattr(target, leaf, value)


def _coerce(value: Any, kind: type) -> Any:
    if kind is bool:
        return parse_bool(value)
    if kind in (int, float) and isinstance(value, bool):
        raise ValueError("expected a number")
    return kind(value.strip() if isinstance(value, str) else value)


def _nested(dotted: str, value: Any) -> dict[str, Any]:
    *parents, leaf = dotted.split(".")
    out: dict[str, Any] = {leaf: value}
    for part in reversed(parents):
        out = {part: out}
    return out


def _merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, Mapping):
            _merge(base.setdefault(key, {}), value)
        else:
            base[key] = value
    return base


class SettingsControl:
    def __init__(
        self,
        settings: Settings,
        detector: SwitchableDetector,
        config_path: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._settings = settings
        self._detector = detector
        self._config_path = config_path
        self._env = os.environ if env is None else env
        self._lock = asyncio.Lock()

    @property
    def settings(self) -> Settings:
        return self._settings

    def estimate(self, settings: Settings | None = None) -> pricing.CostEstimate | None:
        """Cost per print-hour for the decision backend, or ``None`` for YOLO."""
        settings = settings or self._settings
        detection = settings.detection
        if detection.backend != "decision":
            return None
        measured = None
        if detection is self._settings.detection:
            measured = self._detector.average_cost
        return pricing.estimate(
            detection.decision.provider,
            detection.decision.model,
            settings.target_loop_time,
            detection.interval,
            measured,
        )

    def public(self) -> dict[str, Any]:
        """Current values for the dashboard. The API key is never returned."""
        values = {
            name: _get(self._settings, path) for name, (path, _kind) in FIELDS.items()
        }
        key = values.pop("api_key")
        env_names = env_var_names()
        estimate = self.estimate()
        return {
            "values": values,
            "api_key_set": bool(key),
            "api_key_hint": f"…{key[-4:]}" if len(key) >= 8 else "",
            "env_locked": sorted(
                name
                for name, (path, _kind) in FIELDS.items()
                if env_names[path] in self._env
            ),
            "providers": DECISION_PROVIDERS,
            "models": [
                {"id": model, "label": label, "cost_per_request": cost}
                for model, label, cost in pricing.KNOWN_MODELS
            ],
            "estimate": None
            if estimate is None
            else {
                "detections_per_hour": estimate.detections_per_hour,
                "cost_per_request": estimate.cost_per_request,
                "cost_per_hour": estimate.cost_per_hour,
                "source": estimate.source,
            },
            "min_interval": MIN_INTERVAL,
        }

    async def update(self, changes: Mapping[str, Any]) -> dict[str, Any]:
        """Validate, apply and persist dashboard changes.

        Raises :class:`ConfigError` with a user-facing message on bad input or
        when the new detector cannot be built.
        """
        unknown = sorted(set(changes) - set(FIELDS))
        if unknown:
            raise ConfigError(f"Unknown settings: {', '.join(unknown)}.")

        async with self._lock:
            candidate = copy.deepcopy(self._settings)
            persisted: dict[str, Any] = {}
            for name, raw in changes.items():
                path, kind = FIELDS[name]
                if name == "api_key" and not str(raw).strip():
                    continue  # blank means "keep the saved key"
                try:
                    value = _coerce(raw, kind)
                except (TypeError, ValueError) as exc:
                    raise ConfigError(f"Invalid value for {name}: {raw!r}.") from exc
                _set(candidate, path, value)
                _merge(persisted, _nested(path, value))

            if (
                "capture_interval" in changes
                and candidate.target_loop_time < MIN_INTERVAL
            ):
                raise ConfigError(
                    f"Capture interval must be at least {MIN_INTERVAL:g} seconds."
                )
            if 0 < candidate.detection.interval < candidate.target_loop_time:
                raise ConfigError(
                    "Detection interval can't be shorter than the capture interval. "
                    "Use 0 to detect on every captured frame."
                )
            candidate.detection.validate()
            # Decision settings can be prepared before switching backends, so
            # check them even while YOLO is active. The key can come later.
            candidate.detection.decision.validate(require_key=False)

            if candidate.detection != self._settings.detection:
                try:
                    detector = await asyncio.to_thread(
                        self._detector.build, candidate.detection
                    )
                except Exception as exc:  # noqa: BLE001 - report any build failure
                    raise ConfigError(f"Could not start the new detector: {exc}") from exc
                self._detector.replace(detector)

            self._settings.detection = candidate.detection
            self._settings.target_loop_time = candidate.target_loop_time
            self._settings.pause_on_spaghetti = candidate.pause_on_spaghetti
            if persisted:
                await asyncio.to_thread(save_overrides, persisted, self._config_path)
        return self.public()


__all__ = ["FIELDS", "SettingsControl"]
