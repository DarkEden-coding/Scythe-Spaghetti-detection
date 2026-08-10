"""Interactive configuration wizard.

Writes ``settings.json`` rather than generating a Python module. The old
approach produced a ``settings.py`` that every module imported at startup,
which meant config values could execute code, could not be overridden per
environment, and broke existing installs with ``ImportError`` whenever a new
key was added.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import paths
from src.config import (
    Settings,
    _build,
    load_settings,
    parse_bool,
    read_legacy_settings,
)
from src.errors import ConfigError


@dataclass(frozen=True)
class Prompt:
    """One question the wizard asks."""

    path: str
    label: str
    parse: Callable[[str], Any]
    help: str = ""
    secret: bool = False


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise ValueError("must be greater than zero")
    return value


def _confidence(text: str) -> float:
    value = float(text)
    if not 0 < value <= 1:
        raise ValueError("must be between 0 and 1")
    return value


def _snowflake(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise ValueError("must be a positive Discord id")
    return value


def _mode(text: str) -> str:
    value = text.strip().lower()
    if value not in {"edit", "message", "silent"}:
        raise ValueError("must be one of: edit, message, silent")
    return value


PROMPTS: tuple[tuple[str, tuple[Prompt, ...]], ...] = (
    (
        "Discord",
        (
            Prompt(
                "discord.bot_token",
                "Bot token",
                str,
                "From the Discord developer portal.",
                secret=True,
            ),
            Prompt(
                "discord.log_channel_id",
                "Log channel id",
                _snowflake,
                "Enable Developer Mode, right-click the channel, Copy ID.",
            ),
            Prompt(
                "discord.ping_user_id",
                "User id to ping on failure",
                _snowflake,
            ),
            Prompt(
                "discord.status_update_mode",
                "Status updates (edit / message / silent)",
                _mode,
                "'edit' keeps one message current instead of posting every loop.",
            ),
        ),
    ),
    (
        "Printer",
        (
            Prompt("printer.url", "Moonraker URL", str),
            Prompt(
                "printer.webcam_name",
                "Webcam name",
                str,
                "Exactly as it appears in Mainsail/Fluidd — case sensitive.",
            ),
        ),
    ),
    (
        "Detection",
        (
            Prompt(
                "detection.use_cuda",
                "Use CUDA (yes/no)",
                parse_bool,
                "Only if you have an NVIDIA GPU with CUDA installed.",
            ),
            Prompt(
                "detection.use_onnx",
                "Use ONNX runtime (yes/no)",
                parse_bool,
                "Faster on CPU-only machines such as a Raspberry Pi.",
            ),
            Prompt(
                "detection.min_confidence",
                "Minimum confidence (0-1)",
                _confidence,
                "Raise it if you get false positives, lower it for false negatives.",
            ),
        ),
    ),
    (
        "Behaviour",
        (
            Prompt("target_loop_time", "Seconds between checks", _positive_float),
            Prompt(
                "pause_on_spaghetti",
                "Pause the print on detection (yes/no)",
                parse_bool,
            ),
            Prompt(
                "enable_auto_update",
                "Automatically pull updates on start (yes/no)",
                parse_bool,
            ),
            Prompt("log_level", "Log level", str.upper),
        ),
    ),
)


def _get(data: dict, dotted: str) -> Any:
    cursor: Any = data
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _set(data: dict, dotted: str, value: Any) -> None:
    cursor = data
    *parents, leaf = dotted.split(".")
    for part in parents:
        cursor = cursor.setdefault(part, {})
    cursor[leaf] = value


def _mask(value: Any) -> str:
    text = str(value)
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}…{text[-4:]}"


def _existing_values(config_path: Path) -> dict[str, Any]:
    """Current settings as a plain dict, migrating a legacy file if needed."""
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"! {config_path} could not be read; starting from defaults.")
            return {}
    if paths.LEGACY_CONFIG_PATH.exists():
        print(f"Found a legacy {paths.LEGACY_CONFIG_PATH.name}; importing its values.")
        try:
            return read_legacy_settings(paths.LEGACY_CONFIG_PATH)
        except ConfigError as exc:
            print(f"! Could not import legacy settings: {exc}")
    return {}


def run_wizard(config_path: Path | None = None, input_fn=input, output_fn=print) -> Path:
    """Prompt for each setting and write the config file."""
    target = config_path or paths.config_path()
    values = _existing_values(target)
    defaults = Settings().to_dict()

    output_fn("\nScythe configuration")
    output_fn(f"Writing to: {target}")
    output_fn("Press Enter to keep the value shown in brackets.\n")

    for section, prompts in PROMPTS:
        output_fn(f"--- {section} ---")
        for prompt in prompts:
            current = _get(values, prompt.path)
            if current is None:
                current = _get(defaults, prompt.path)

            shown = _mask(current) if prompt.secret and current else current
            if prompt.help:
                output_fn(f"  ({prompt.help})")

            while True:
                raw = input_fn(f"{prompt.label} [{shown}]: ").strip()
                if not raw:
                    _set(values, prompt.path, current)
                    break
                try:
                    _set(values, prompt.path, prompt.parse(raw))
                    break
                except (ValueError, ConfigError) as exc:
                    output_fn(f"  ! Invalid value: {exc}")
        output_fn("")

    settings = _build(Settings, values)
    try:
        settings.validate()
    except ConfigError as exc:
        output_fn(f"! Settings saved, but they are not yet usable: {exc}")

    written = settings.save(target)
    output_fn(f"Saved to {written}")

    if paths.LEGACY_CONFIG_PATH.exists():
        output_fn(
            f"\nNote: {paths.LEGACY_CONFIG_PATH.name} is no longer used and can be "
            f"deleted. It may still contain your bot token — remove it so the "
            f"token cannot be committed."
        )
    return written


def migrate_legacy(config_path: Path | None = None) -> Path | None:
    """Convert a legacy ``settings.py`` to JSON without prompting.

    Returns the written path, or ``None`` if there was nothing to migrate.
    """
    target = config_path or paths.config_path()
    if target.exists() or not paths.LEGACY_CONFIG_PATH.exists():
        return None
    values = read_legacy_settings(paths.LEGACY_CONFIG_PATH)
    if not values:
        return None
    return _build(Settings, values).save(target)


__all__ = ["Prompt", "load_settings", "migrate_legacy", "run_wizard"]
