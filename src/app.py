"""Composition root.

The only place that knows which concrete implementations are wired together.
Everything below it depends on interfaces, which is what makes the roadmap
items (email notifications, a web UI, slash commands) additive rather than a
rewrite of the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

from src.config import Settings
from src.detection.detector import SpaghettiDetector
from src.monitor import MonitorLoop
from src.notify.base import CompositeNotifier, Notifier
from src.printer.base import PrinterClient
from src.printer.moonraker import MoonrakerClient

log = logging.getLogger(__name__)


def build_printer(settings: Settings) -> PrinterClient:
    return MoonrakerClient(settings.printer)


def build_detector(settings: Settings) -> SpaghettiDetector:
    return SpaghettiDetector(settings.detection)


def build_notifier(settings: Settings, printer: PrinterClient) -> Notifier:
    """Assemble the configured notification channels.

    Imported lazily so that ``scythe check`` and the test suite do not need the
    channel libraries merely to construct settings.
    """
    from src.notify.discord_notifier import DiscordNotifier

    channels: list[Notifier] = [DiscordNotifier(settings.discord, printer)]
    if settings.web.enabled:
        from src.notify.web_notifier import WebNotifier

        channels.append(WebNotifier(settings.web, printer, settings.target_loop_time))
    return channels[0] if len(channels) == 1 else CompositeNotifier(channels)


class Application:
    """Owns the lifecycle of every long-lived component."""

    def __init__(
        self,
        settings: Settings,
        printer: PrinterClient | None = None,
        detector: SpaghettiDetector | None = None,
        notifier: Notifier | None = None,
    ):
        self._settings = settings
        self._printer = printer or build_printer(settings)
        self._detector = detector or build_detector(settings)
        self._notifier = notifier or build_notifier(settings, self._printer)
        self._loop: MonitorLoop | None = None

    async def run_async(self) -> None:
        """Load the model, connect, and monitor until interrupted."""
        # Load before connecting: a bad model path should fail fast rather than
        # after announcing itself in the channel.
        log.info("Loading detection model...")
        await asyncio.to_thread(self._detector.load)

        self._install_signal_handlers()

        try:
            await self._notifier.start()
            self._loop = MonitorLoop(
                self._printer, self._detector, self._notifier, self._settings
            )
            await self._loop.run()
        finally:
            with contextlib.suppress(Exception):
                await self._notifier.close()
            await asyncio.to_thread(self._printer.close)

    def run(self) -> int:
        """Synchronous entry point. Returns a process exit code."""
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            log.info("Interrupted; shutting down.")
        except asyncio.CancelledError:
            log.info("Shutdown complete.")
        return 0

    def request_stop(self) -> None:
        if self._loop is not None:
            self._loop.request_stop()

    def _install_signal_handlers(self) -> None:
        """Stop the loop cleanly on SIGINT/SIGTERM where the platform allows.

        systemd sends SIGTERM on ``stop``; without this the process is killed
        mid-iteration and never posts its shutdown message.
        """
        loop = asyncio.get_running_loop()
        for name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except (NotImplementedError, RuntimeError):
                # Windows proactor loops do not support this; KeyboardInterrupt
                # still unwinds through run().
                log.debug("Signal handler for %s unavailable on this platform.", name)


__all__ = ["Application", "build_detector", "build_notifier", "build_printer"]
