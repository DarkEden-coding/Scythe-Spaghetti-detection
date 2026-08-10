"""The monitoring loop.

Depends only on the :class:`~src.printer.base.PrinterClient`,
:class:`~src.detection.detector.SpaghettiDetector`, and
:class:`~src.notify.base.Notifier` abstractions, so it can be exercised in
tests with three small fakes and no network, model, or Discord token.

Every blocking call is dispatched to a worker thread. Running YOLO inference
directly on the event loop — as the original did — blocks the Discord
heartbeat for however long the model takes, which on a Pi is long enough to be
disconnected mid-print.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timezone
from time import monotonic

from PIL import Image

from src import paths
from src.config import Settings
from src.detection.annotate import draw_boxes
from src.detection.detector import SpaghettiDetector
from src.detection.results import DetectionResult
from src.errors import ScytheError
from src.events import (
    DebugDetection,
    ImageUnavailable,
    MonitorError,
    MonitoringStarted,
    MonitoringStopped,
    SpaghettiDetected,
    StatusUpdate,
)
from src.notify.base import Notifier
from src.printer.base import PrinterClient, PrintState

log = logging.getLogger(__name__)

#: After the first failure, only re-notify every Nth consecutive one so a
#: printer that is simply switched off does not flood the channel.
FAILURE_NOTIFY_INTERVAL = 10


class DebugDetectionControl:
    """Runtime-only switch for running detection while the printer is idle."""

    def __init__(self) -> None:
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Return whether idle debug detection is enabled."""
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable idle debug detection until process restart."""
        self._enabled = enabled


class MonitorLoop:
    """Polls the printer, runs detection, and emits events."""

    def __init__(
        self,
        printer: PrinterClient,
        detector: SpaghettiDetector,
        notifier: Notifier,
        settings: Settings,
        debug_detection: DebugDetectionControl | None = None,
    ):
        self._printer = printer
        self._detector = detector
        self._notifier = notifier
        self._settings = settings
        self._debug_detection = debug_detection or DebugDetectionControl()

        self._started_at = monotonic()
        self._stop = asyncio.Event()
        self._image_failures = 0
        self._error_failures = 0

    # -- control ----------------------------------------------------------- #

    @property
    def uptime(self) -> float:
        return monotonic() - self._started_at

    def request_stop(self) -> None:
        """Ask the loop to finish the current iteration and exit."""
        self._stop.set()

    async def run(self) -> None:
        """Run until :meth:`request_stop` is called or the task is cancelled."""
        self._started_at = monotonic()
        reason = "Stopped."

        await self._notifier.notify(
            MonitoringStarted(
                target_loop_time=self._settings.target_loop_time,
                printer_url=self._settings.printer.base_url,
                model_name=self._detector.model_path.name,
            )
        )

        try:
            while not self._stop.is_set():
                started = monotonic()
                try:
                    await self._tick()
                    self._error_failures = 0
                except asyncio.CancelledError:
                    raise
                except ScytheError as exc:
                    await self._report_error(str(exc))
                except Exception as exc:  # noqa: BLE001 - the loop must survive
                    log.exception("Unexpected error in monitor loop")
                    await self._report_error(f"{type(exc).__name__}: {exc}")

                await self._sleep_remaining(monotonic() - started)
        except asyncio.CancelledError:
            reason = "Monitoring cancelled."
            raise
        finally:
            await self._safe_notify(
                MonitoringStopped(reason=reason, uptime_seconds=self.uptime)
            )

    async def _sleep_remaining(self, elapsed: float) -> None:
        """Pace the loop, tolerating iterations that overrun the target.

        ``target - elapsed`` goes negative whenever inference is slower than
        the interval; ``asyncio.sleep`` on a negative delay yields immediately
        and the loop spins. Clamping at zero keeps it honest.
        """
        remaining = max(0.0, self._settings.target_loop_time - elapsed)
        if remaining == 0.0:
            log.debug(
                "Iteration took %.1fs, over the %.1fs target; not sleeping.",
                elapsed,
                self._settings.target_loop_time,
            )
        # Waiting on the stop event rather than sleeping means shutdown is
        # immediate instead of up to one interval late. Timing out is the
        # normal path: it means the interval simply elapsed.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._stop.wait(), timeout=remaining)

    # -- one iteration ----------------------------------------------------- #

    async def _tick(self) -> None:
        image = await asyncio.to_thread(self._printer.get_snapshot)
        if image is None:
            await self._report_image_failure("Failed to fetch a camera frame.")
            return
        self._image_failures = 0

        state = await asyncio.to_thread(self._printer.get_state)
        if not state.is_active and not self._debug_detection.enabled:
            await self._notifier.notify(
                StatusUpdate(
                    uptime_seconds=self.uptime,
                    state=state,
                    image=image,
                    detail="Idle — detection skipped.",
                )
            )
            return

        result = await asyncio.to_thread(self._detector.detect, image)
        if not state.is_active:
            if result:
                await self._notifier.notify(
                    DebugDetection(
                        result=result,
                        state=state,
                        uptime_seconds=self.uptime,
                    )
                )
            else:
                await self._notifier.notify(
                    StatusUpdate(
                        uptime_seconds=self.uptime,
                        state=state,
                        image=image,
                        detail=(
                            "Idle debug detection — no spaghetti "
                            f"({result.duration_seconds:.1f}s inference)."
                        ),
                    )
                )
            return

        if not result:
            await self._notifier.notify(
                StatusUpdate(
                    uptime_seconds=self.uptime,
                    state=state,
                    image=image,
                    detail=f"No spaghetti ({result.duration_seconds:.1f}s inference).",
                )
            )
            return

        await self._handle_detection(result)

    async def _handle_detection(self, result: DetectionResult) -> None:
        log.warning("Spaghetti detected: %s", result.summary())

        pause_requested = self._settings.pause_on_spaghetti
        paused = False
        if pause_requested:
            paused = await asyncio.to_thread(self._printer.pause)

        annotated = None
        if result.image is not None:
            annotated = await asyncio.to_thread(draw_boxes, result.image, result.boxes)

        saved = ""
        if annotated is not None and self._settings.detection.save_annotated_frames:
            saved = await asyncio.to_thread(self._save_frame, annotated)

        ack = await self._notifier.notify(
            SpaghettiDetected(
                result=result,
                annotated_image=annotated,
                paused=paused,
                pause_requested=pause_requested,
                saved_frame=saved,
            )
        )

        if ack is None:
            log.warning(
                "No channel can acknowledge this failure; pausing detection is "
                "the operator's call. Continuing."
            )
            return

        timeout = self._settings.discord.acknowledge_timeout or None
        await ack.wait(timeout)

    @staticmethod
    def _save_frame(image: Image.Image) -> str:
        """Persist an annotated failure frame for post-mortem debugging."""
        directory = paths.ensure_data_dir() / "detections"
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = directory / f"spaghetti-{stamp}.jpg"
        image.convert("RGB").save(target, format="JPEG", quality=90)
        log.info("Saved annotated frame to %s", target)
        return str(target)

    # -- failure reporting ------------------------------------------------- #

    async def _report_image_failure(self, reason: str) -> None:
        self._image_failures += 1
        log.warning("%s (consecutive: %d)", reason, self._image_failures)
        if self._should_notify(self._image_failures):
            await self._safe_notify(
                ImageUnavailable(reason=reason, consecutive_failures=self._image_failures)
            )

    async def _report_error(self, message: str) -> None:
        self._error_failures += 1
        log.error("%s (consecutive: %d)", message, self._error_failures)
        if self._should_notify(self._error_failures):
            await self._safe_notify(
                MonitorError(message=message, consecutive_failures=self._error_failures)
            )

    @staticmethod
    def _should_notify(count: int) -> bool:
        return count == 1 or count % FAILURE_NOTIFY_INTERVAL == 0

    async def _safe_notify(self, event) -> None:
        """Notify without letting a broken channel take down the loop."""
        try:
            await self._notifier.notify(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Failed to deliver %s", type(event).__name__)


__all__ = ["DebugDetectionControl", "MonitorLoop", "PrintState"]
