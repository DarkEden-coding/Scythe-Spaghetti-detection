"""Command line interface.

``scythe run`` replaces ``python main.py``; ``scythe configure`` replaces
``python settings_ui.py``. Both legacy entry points still work as thin shims.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from src import __version__, paths
from src.config import ConfigError, env_var_names, load_settings
from src.logging_setup import configure_logging

log = logging.getLogger("src.cli")

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_RUNTIME_ERROR = 3


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scythe",
        description="Self-hosted 3D printer spaghetti detection.",
    )
    parser.add_argument("--version", action="version", version=f"scythe {__version__}")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Config file to use (default: settings.json beside the package).",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override the configured log level.",
    )

    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="Start monitoring (default).")
    sub.add_parser("configure", help="Interactively create or update the config.")

    check = sub.add_parser("check", help="Validate config and printer connectivity.")
    check.add_argument(
        "--snapshot",
        type=Path,
        metavar="PATH",
        help="Save a test camera frame to PATH.",
    )

    sub.add_parser("env", help="List the environment variable for every setting.")
    sub.add_parser("update", help="Check for and apply updates, then exit.")
    return parser


def _load(args: argparse.Namespace):
    settings = load_settings(args.config)
    if args.log_level:
        settings.log_level = args.log_level
    return settings


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_configure(args: argparse.Namespace) -> int:
    from src.wizard import run_wizard

    configure_logging("WARNING", console=False)
    try:
        run_wizard(args.config)
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled; nothing was written.")
        return EXIT_OK
    return EXIT_OK


def cmd_env(_args: argparse.Namespace) -> int:
    width = max(len(k) for k in env_var_names())
    print("Every setting can be overridden by an environment variable:\n")
    for dotted, var in sorted(env_var_names().items()):
        print(f"  {dotted:<{width}}  {var}")
    return EXIT_OK


def cmd_update(args: argparse.Namespace) -> int:
    from src.updater import Updater

    configure_logging(args.log_level or "INFO")
    # Run regardless of the setting: an explicit `scythe update` is consent.
    result = Updater(paths.PROJECT_ROOT, enabled=True).check_and_apply()
    log.info(result.message or "Nothing to do.")
    for name in result.changed_files:
        log.info("  changed: %s", name)
    return EXIT_OK


def cmd_check(args: argparse.Namespace) -> int:
    from src.printer.moonraker import MoonrakerClient

    settings = _load(args)
    configure_logging(settings.log_level)

    ok = True
    print(f"Config              : OK ({args.config or paths.config_path()})")

    model = settings.detection.active_model_path
    if model.exists():
        print(f"Model               : OK ({model.name})")
    elif settings.detection.use_onnx:
        print(f"Model               : will be exported to {model.name} on first run")
    else:
        print(f"Model               : MISSING ({model})")
        ok = False

    with MoonrakerClient(settings.printer) as printer:
        state = printer.get_state()
        print(f"Printer state       : {state.value}")
        if state.value == "unknown":
            print("                      (could not read print_stats — check the URL)")
            ok = False

        try:
            url = printer.snapshot_url
            print(f"Webcam              : OK ({url})")
        except Exception as exc:  # noqa: BLE001 - this command reports, not raises
            print(f"Webcam              : FAILED ({exc})")
            return EXIT_RUNTIME_ERROR

        image = printer.get_snapshot()
        if image is None:
            print("Snapshot            : FAILED")
            ok = False
        else:
            print(f"Snapshot            : OK ({image.width}x{image.height} {image.mode})")
            if args.snapshot:
                image.convert("RGB").save(args.snapshot)
                print(f"                      saved to {args.snapshot}")

    print("\nResult              :", "OK" if ok else "PROBLEMS FOUND")
    return EXIT_OK if ok else EXIT_RUNTIME_ERROR


def cmd_run(args: argparse.Namespace) -> int:
    from src.app import Application
    from src.updater import Updater

    settings = _load(args)
    log_file = configure_logging(settings.log_level)
    log.info("Scythe %s starting (log file: %s)", __version__, log_file)

    if settings.enable_auto_update:
        result = Updater(paths.PROJECT_ROOT, enabled=True).check_and_apply()
        if result.message:
            log.info("Auto-update: %s", result.message)
        if result.restart_required:
            log.info("Code changed; restarting into the new version.")
            _restart()

    return Application(settings).run()


def _restart() -> None:
    """Replace this process with a fresh one.

    ``os.execv`` rather than ``subprocess.run``: the previous implementation
    spawned a child and waited on it, so every update left another parent
    process sitting in the tree.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, "-m", "src", *sys.argv[1:]])


COMMANDS = {
    "run": cmd_run,
    "configure": cmd_configure,
    "check": cmd_check,
    "env": cmd_env,
    "update": cmd_update,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    handler = COMMANDS[args.command or "run"]

    try:
        return handler(args)
    except ConfigError as exc:
        configure_logging("ERROR", console=False)
        print(f"Configuration error: {exc}", file=sys.stderr)
        print("\nRun `scythe configure` to set things up.", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_OK
    except Exception as exc:  # noqa: BLE001 - top level: report, never traceback-dump
        log.exception("Fatal error")
        print(f"Fatal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
