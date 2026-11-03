"""Typed, validated application settings.

Settings live in a JSON file (``settings.json`` by default) and may be
overridden per-field by environment variables, which is what makes the project
usable under systemd, Docker, or a secrets manager without editing files.

Precedence, lowest to highest: dataclass defaults, config file, environment.

The pre-0.2 format was a generated ``settings.py`` that every module imported.
:func:`read_legacy_settings` parses that file *without executing it* so existing
installs can be migrated automatically.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from functools import cache
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from src import paths
from src.errors import ConfigError

ENV_PREFIX = "SCYTHE"

#: Placeholder values treated as "not configured yet".
PLACEHOLDERS = frozenset({"", "YOUR DISCORD BOT TOKEN", "CHANGEME", "changeme", "TODO"})

STATUS_UPDATE_MODES = frozenset({"edit", "message", "silent"})

TRUE_WORDS = frozenset({"1", "true", "t", "yes", "y", "on"})
FALSE_WORDS = frozenset({"0", "false", "f", "no", "n", "off"})


def parse_bool(value: Any) -> bool:
    """Coerce a config value to ``bool``.

    ``bool("False")`` is ``True``, which silently defeated every boolean in the
    old prompt-driven config. Word forms are parsed explicitly instead.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in TRUE_WORDS:
        return True
    if text in FALSE_WORDS:
        return False
    raise ConfigError(
        f"Cannot interpret {value!r} as a yes/no value. "
        f"Use one of: {', '.join(sorted(TRUE_WORDS | FALSE_WORDS))}."
    )


@dataclass
class DiscordSettings:
    """Credentials and behaviour for the Discord notification channel."""

    bot_token: str = ""
    ping_user_id: int = 0
    log_channel_id: int = 0
    #: ``edit`` keeps a single status message up to date instead of posting one
    #: per loop (~5,700 messages/day at the default interval).
    status_update_mode: str = "edit"
    #: Seconds to wait for a human acknowledgement before resuming detection.
    #: ``0`` waits forever.
    acknowledge_timeout: float = 0.0

    @property
    def enabled(self) -> bool:
        """An empty token turns Discord off; the dashboard still works."""
        return bool(self.bot_token)

    def validate(self) -> None:
        if self.bot_token in PLACEHOLDERS:
            raise ConfigError(
                "discord.bot_token is not set. Add it in the dashboard settings or set "
                f"{ENV_PREFIX}_DISCORD_BOT_TOKEN."
            )
        if self.log_channel_id <= 0:
            raise ConfigError("discord.log_channel_id must be a valid channel id.")
        if self.ping_user_id <= 0:
            raise ConfigError("discord.ping_user_id must be a valid user id.")
        if self.status_update_mode not in STATUS_UPDATE_MODES:
            raise ConfigError(
                f"discord.status_update_mode must be one of "
                f"{sorted(STATUS_UPDATE_MODES)}, got {self.status_update_mode!r}."
            )
        if self.acknowledge_timeout < 0:
            raise ConfigError("discord.acknowledge_timeout cannot be negative.")


@dataclass
class PrinterSettings:
    """How to reach Moonraker."""

    #: Moonraker, or the Mainsail/Fluidd address that proxies it. The web
    #: setup finds this automatically on the printer's own host.
    url: str = "http://localhost/"
    webcam_name: str = "Bed"
    request_timeout: float = 10.0
    max_retries: int = 3

    def validate(self) -> None:
        if not self.url:
            raise ConfigError("printer.url is required.")
        if not self.url.startswith(("http://", "https://")):
            raise ConfigError(
                f"printer.url must start with http:// or https://, got {self.url!r}."
            )
        if self.request_timeout <= 0:
            raise ConfigError("printer.request_timeout must be positive.")
        if self.max_retries < 0:
            raise ConfigError("printer.max_retries cannot be negative.")

    @property
    def base_url(self) -> str:
        """The printer URL with exactly one trailing slash."""
        return self.url.rstrip("/") + "/"


@dataclass
class WebSettings:
    """Local operator dashboard bind settings."""

    enabled: bool = True
    host: str = "0.0.0.0"
    #: 8080 is often taken by the camera streamer on Klipper hosts.
    port: int = 8090
    #: PBKDF2 hash of the dashboard password; empty means no login.
    password_hash: str = ""
    #: Signs login cookies. Generated when a password is first set.
    session_secret: str = ""
    #: Address that reaches the dashboard from elsewhere (e.g. a Tailscale
    #: URL). Discord alerts link to it when set.
    public_url: str = ""

    def validate(self) -> None:
        """Validate the dashboard listener address."""
        if not self.host.strip():
            raise ConfigError("web.host is required.")
        if not 1 <= self.port <= 65535:
            raise ConfigError("web.port must be between 1 and 65535.")
        if self.public_url and not self.public_url.startswith(("http://", "https://")):
            raise ConfigError("web.public_url must start with http:// or https://.")


DETECTION_BACKENDS = frozenset({"yolo", "decision"})

#: Decision-model providers and the endpoint each one uses unless
#: ``detection.decision.base_url`` overrides it.
DECISION_PROVIDERS = {
    "openrouter": "https://openrouter.ai/api/alpha/decisions",
    "ollama": "http://localhost:11434/v1/systemone",
}


@dataclass
class DecisionSettings:
    """Decision-model backend (``detection.backend = "decision"``).

    Defaults come from the print-failure evaluation (see
    ``PRINT_FAILURE_EVAL_FINDINGS.md``): a crop around the part and a
    layer-shift-specific question mattered more than the choice of model.
    """

    #: ``openrouter`` for the hosted API, ``ollama`` for a self-hosted server.
    provider: str = "openrouter"
    #: Endpoint override; empty uses the provider's default.
    base_url: str = ""
    api_key: str = ""
    #: d1-ultra matched Jev in good light and was the best cloud model in the
    #: dark, at about half Jev's price.
    model: str = "~liquid/d1-ultra-latest"
    #: Flag a failure when any question's P(yes) reaches this. Decision models
    #: are poorly calibrated, so tune it per model on your own frames.
    threshold: float = 0.6
    #: Optional crop around the part as ``left,top,right,bottom`` fractions of
    #: the frame, e.g. ``0.32,0.21,0.93,0.61``. Empty sends the full frame.
    crop: str = ""
    #: Longest side of the image sent to the model, in pixels.
    image_max_side: int = 1240
    request_timeout: float = 30.0

    def validate(self, require_key: bool = True) -> None:
        if self.provider not in DECISION_PROVIDERS:
            raise ConfigError(
                f"detection.decision.provider must be one of "
                f"{sorted(DECISION_PROVIDERS)}, got {self.provider!r}."
            )
        if self.base_url and not self.base_url.startswith(("http://", "https://")):
            raise ConfigError(
                "detection.decision.base_url must start with http:// or https://."
            )
        if require_key and self.provider == "openrouter" and self.api_key in PLACEHOLDERS:
            raise ConfigError(
                "detection.decision.api_key is not set. Enter it in the dashboard "
                f"settings or set {ENV_PREFIX}_DETECTION_DECISION_API_KEY."
            )
        if not self.model.strip():
            raise ConfigError("detection.decision.model is required.")
        if not 0.0 < self.threshold <= 1.0:
            raise ConfigError(
                "detection.decision.threshold must be between 0 (exclusive) and 1."
            )
        self.crop_box()
        if self.image_max_side < 256:
            raise ConfigError("detection.decision.image_max_side must be at least 256.")
        if self.request_timeout <= 0:
            raise ConfigError("detection.decision.request_timeout must be positive.")

    @property
    def endpoint(self) -> str:
        return self.base_url or DECISION_PROVIDERS.get(self.provider, "")

    def crop_box(self) -> tuple[float, float, float, float] | None:
        """Parse :attr:`crop` into fractions, or ``None`` for the full frame."""
        if not self.crop.strip():
            return None
        try:
            left, top, right, bottom = (float(part) for part in self.crop.split(","))
        except ValueError as exc:
            raise ConfigError(
                "detection.decision.crop must be four comma-separated fractions: "
                f"left,top,right,bottom. Got {self.crop!r}."
            ) from exc
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise ConfigError(
                "detection.decision.crop fractions must satisfy "
                f"0 <= left < right <= 1 and 0 <= top < bottom <= 1. Got {self.crop!r}."
            )
        return left, top, right, bottom


@dataclass
class DetectionSettings:
    """Model selection and inference thresholds."""

    #: ``yolo`` runs the bundled local model; ``decision`` asks a decision
    #: model (OpenRouter or self-hosted) instead.
    backend: str = "yolo"
    decision: DecisionSettings = field(default_factory=DecisionSettings)
    #: Seconds between detections. Frames are still captured every
    #: ``target_loop_time``; ``0`` runs detection on every captured frame.
    interval: float = 0.0
    #: ONNX model run by the ``yolo`` backend.
    model_path: Path = paths.DEFAULT_MODEL
    #: Use ONNX Runtime's CUDA provider when it is installed.
    use_cuda: bool = False
    #: Ignored. Kept so pre-0.3 config files still load; ONNX is always used.
    use_onnx: bool = True
    min_confidence: float = 0.6
    #: Class index that represents spaghetti in the shipped model.
    spaghetti_class_id: int = 1
    image_size: int = 640
    #: Persist the annotated frame to ``data/`` for post-mortem debugging.
    save_annotated_frames: bool = True

    def validate(self) -> None:
        if self.backend not in DETECTION_BACKENDS:
            raise ConfigError(
                f"detection.backend must be one of {sorted(DETECTION_BACKENDS)}, "
                f"got {self.backend!r}."
            )
        if self.interval < 0:
            raise ConfigError("detection.interval cannot be negative.")
        if self.backend == "decision":
            self.decision.validate()
            return
        if not 0.0 < self.min_confidence <= 1.0:
            raise ConfigError(
                "detection.min_confidence must be between 0 (exclusive) and 1."
            )
        if self.image_size <= 0:
            raise ConfigError("detection.image_size must be positive.")

    def __post_init__(self) -> None:
        # Pre-0.3 configs point at the PyTorch checkpoint; the runtime only
        # reads its exported sibling.
        if self.model_path.suffix == ".pt":
            self.model_path = self.model_path.with_suffix(".onnx")


@dataclass
class Settings:
    """Root settings object, injected into every component."""

    discord: DiscordSettings = field(default_factory=DiscordSettings)
    printer: PrinterSettings = field(default_factory=PrinterSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    web: WebSettings = field(default_factory=WebSettings)

    #: Target seconds between monitor iterations.
    target_loop_time: float = 30.0
    #: False until the web setup is finished. Monitoring waits for it.
    setup_complete: bool = False
    pause_on_spaghetti: bool = True
    enable_auto_update: bool = False
    log_level: str = "INFO"

    def validate(self) -> Settings:
        if self.target_loop_time <= 0:
            raise ConfigError("target_loop_time must be positive.")
        if self.log_level.upper() not in {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }:
            raise ConfigError(f"Unknown log_level {self.log_level!r}.")
        # Discord is optional when the dashboard is on, so there is always
        # somewhere for alerts to go.
        if self.discord.enabled or not self.web.enabled:
            self.discord.validate()
        self.printer.validate()
        self.detection.validate()
        self.web.validate()
        return self

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(self)

    def save(self, path: Path | None = None) -> Path:
        """Write settings to disk with owner-only permissions."""
        target = path or paths.config_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        _restrict_permissions(target)
        return target


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def _coerce(value: Any, annotation: Any) -> Any:
    """Convert a raw config/env value to the field's declared type."""
    origin = get_origin(annotation)
    if origin is not None:
        # e.g. `int | str` — take the first arm that accepts the value.
        for arm in get_args(annotation):
            if arm is type(None):
                continue
            try:
                return _coerce(value, arm)
            except (ConfigError, TypeError, ValueError):
                continue
        raise ConfigError(f"Cannot coerce {value!r} to {annotation}.")
    if annotation is bool:
        return parse_bool(value)
    if annotation is int:
        return int(str(value).strip())
    if annotation is float:
        return float(str(value).strip())
    if annotation is Path:
        return _resolve_path(value)
    if annotation is str:
        return str(value)
    return value


@cache
def _hints(cls: type) -> dict[str, Any]:
    """Resolved field annotations.

    ``dataclasses.fields()`` hands back annotation *strings* under
    ``from __future__ import annotations``, so they have to be resolved before
    they can be compared against real types.
    """
    return get_type_hints(cls)


def _build(cls: type, raw: Mapping[str, Any], where: str = "") -> Any:
    """Recursively instantiate a settings dataclass from a mapping."""
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Expected an object for {where or cls.__name__}.")

    hints = _hints(cls)
    known = {f.name for f in fields(cls)}
    for key in raw:
        if key not in known:
            raise ConfigError(
                f"Unknown setting {where + key!r}. Known keys: {sorted(known)}"
            )

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        annotation = hints[f.name]
        if is_dataclass(annotation):
            kwargs[f.name] = _build(annotation, value, f"{where}{f.name}.")
        else:
            try:
                kwargs[f.name] = _coerce(value, annotation)
            except ConfigError as exc:
                raise ConfigError(f"Invalid value for {where}{f.name}: {exc}") from exc
            except (TypeError, ValueError) as exc:
                raise ConfigError(
                    f"Invalid value for {where}{f.name}: {value!r} ({exc})"
                ) from exc
    return cls(**kwargs)


def _env_overrides(cls: type, env: Mapping[str, str], prefix: str) -> dict[str, Any]:
    """Collect ``SCYTHE_SECTION_FIELD`` overrides into a nested mapping."""
    hints = _hints(cls)
    out: dict[str, Any] = {}
    for f in fields(cls):
        annotation = hints[f.name]
        if is_dataclass(annotation):
            nested = _env_overrides(annotation, env, f"{prefix}{f.name.upper()}_")
            if nested:
                out[f.name] = nested
        else:
            name = f"{prefix}{f.name.upper()}"
            if name in env:
                out[f.name] = env[name]
    return out


def env_var_names() -> dict[str, str]:
    """Map dotted setting paths to their environment variable names (for docs)."""

    def walk(cls: type, dotted: str, prefix: str) -> dict[str, str]:
        hints = _hints(cls)
        out: dict[str, str] = {}
        for f in fields(cls):
            annotation = hints[f.name]
            if is_dataclass(annotation):
                out.update(
                    walk(
                        annotation,
                        f"{dotted}{f.name}.",
                        f"{prefix}{f.name.upper()}_",
                    )
                )
            else:
                out[f"{dotted}{f.name}"] = f"{prefix}{f.name.upper()}"
        return out

    return walk(Settings, "", f"{ENV_PREFIX}_")


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _resolve_path(value: Any) -> Path:
    """Interpret a configured path.

    Relative paths resolve against the installation root, never the current
    working directory — the same config must work from systemd, from a shell in
    the repo, and from anywhere else.
    """
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (paths.PROJECT_ROOT / path)


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        # Store paths inside the install as relative, so a config file stays
        # portable between machines and reads cleanly in a diff.
        try:
            return obj.relative_to(paths.PROJECT_ROOT).as_posix()
        except ValueError:
            return str(obj)
    return obj


def _restrict_permissions(path: Path) -> None:
    """Best-effort chmod 600 — the file holds a bot token."""
    # Windows and some network filesystems have no POSIX mode bits; a config
    # file that cannot be locked down is still better than no config file.
    with contextlib.suppress(OSError):
        path.chmod(0o600)


# --------------------------------------------------------------------------- #
# Legacy migration
# --------------------------------------------------------------------------- #

#: Flat pre-0.2 key -> dotted path in the new schema.
LEGACY_KEY_MAP = {
    "discord_bot_token": "discord.bot_token",
    "discord_ping_userid": "discord.ping_user_id",
    "discord_log_channel_id": "discord.log_channel_id",
    "target_loop_time": "target_loop_time",
    "printer_url": "printer.url",
    "webcam_name": "printer.webcam_name",
    "pause_on_spaghetti": "pause_on_spaghetti",
    "use_cuda": "detection.use_cuda",
    "enable_auto_update": "enable_auto_update",
    "use_onnx": "detection.use_onnx",
}


def read_legacy_settings(path: Path) -> dict[str, Any]:
    """Parse a pre-0.2 ``settings.py`` into the nested schema.

    The file is parsed with :mod:`ast`, never imported, so a corrupted or
    hostile config cannot execute code just by being present.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise ConfigError(f"Could not read legacy settings from {path}: {exc}") from exc

    flat: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            flat[target.id] = ast.literal_eval(node.value)
        except ValueError:
            continue  # non-literal assignment: not a setting we understand

    nested: dict[str, Any] = {}
    for legacy_key, dotted in LEGACY_KEY_MAP.items():
        if legacy_key not in flat:
            continue
        cursor = nested
        *parents, leaf = dotted.split(".")
        for part in parents:
            cursor = cursor.setdefault(part, {})
        cursor[leaf] = flat[legacy_key]
    return nested


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def load_settings(
    path: Path | None = None,
    env: Mapping[str, str] | None = None,
    *,
    validate: bool = True,
) -> Settings:
    """Load settings from disk plus environment overrides.

    With no config at all the defaults are returned, and the dashboard's
    setup takes over. Raises :class:`ConfigError` for invalid settings.
    """
    config_file = path or paths.config_path()
    env = os.environ if env is None else env

    raw: dict[str, Any] = {}
    if config_file.exists():
        try:
            raw = json.loads(config_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{config_file} is not valid JSON: {exc}") from exc
    elif paths.LEGACY_CONFIG_PATH.exists():
        raw = read_legacy_settings(paths.LEGACY_CONFIG_PATH)

    overrides = _env_overrides(Settings, env, f"{ENV_PREFIX}_")
    # Configs written before the web setup existed were set up by hand.
    if raw and "setup_complete" not in raw and "printer" in raw:
        raw["setup_complete"] = True

    settings = _build(Settings, _deep_merge(raw, overrides))
    return settings.validate() if validate else settings


def save_overrides(changes: Mapping[str, Any], path: Path | None = None) -> Path:
    """Merge ``changes`` into the config file without touching other keys.

    Saving the whole :class:`Settings` would also write values that came from
    environment variables, such as a bot token kept out of the file on purpose.
    """
    target = path or paths.config_path()
    raw: dict[str, Any] = {}
    if target.exists():
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{target} is not valid JSON: {exc}") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    merged = _deep_merge(raw, _to_jsonable_mapping(changes))
    target.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    _restrict_permissions(target)
    return target


def _to_jsonable_mapping(changes: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _to_jsonable_mapping(value)
        if isinstance(value, Mapping)
        else _to_jsonable(value)
        for key, value in changes.items()
    }


def settings_exist(path: Path | None = None) -> bool:
    """True if either the current or the legacy config file is present."""
    return (path or paths.config_path()).exists() or paths.LEGACY_CONFIG_PATH.exists()


__all__ = [
    "DetectionSettings",
    "DiscordSettings",
    "DecisionSettings",
    "PrinterSettings",
    "Settings",
    "WebSettings",
    "env_var_names",
    "load_settings",
    "parse_bool",
    "read_legacy_settings",
    "save_overrides",
    "settings_exist",
]
