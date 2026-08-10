"""Local web dashboard notification channel.

The dashboard consumes the same monitor events as Discord, so it never creates
another camera or inference loop. It keeps only the current frame and latest
detection in memory.
"""

from __future__ import annotations

import asyncio
import logging
from io import BytesIO
from pathlib import Path
from time import time
from typing import Any

from aiohttp import web
from PIL import Image

from src.config import WebSettings
from src.events import (
    DebugDetection,
    Event,
    ImageUnavailable,
    MonitorError,
    MonitoringStarted,
    MonitoringStopped,
    SpaghettiDetected,
    StatusUpdate,
)
from src.monitor import DebugDetectionControl
from src.notify.base import BaseNotifier
from src.printer.base import PrinterClient

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"


class WebAcknowledgement:
    """An operator acknowledgement delivered through the dashboard."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def acknowledge(self) -> None:
        """Release the monitor's pending detection wait."""
        self._event.set()

    async def wait(self, timeout: float | None = None) -> bool:
        """Wait until acknowledged, or return false after a configured timeout."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return True


class WebNotifier(BaseNotifier):
    """Serve the latest monitor state and narrow printer controls over HTTP."""

    def __init__(
        self,
        settings: WebSettings,
        printer: PrinterClient,
        loop_interval: float,
        debug_detection: DebugDetectionControl,
    ) -> None:
        self._settings = settings
        self._printer = printer
        self._loop_interval = loop_interval
        self._debug_detection = debug_detection
        self._runner: web.AppRunner | None = None
        self._revision = 0
        self._frame: bytes | None = None
        self._frame_meta: dict[str, Any] | None = None
        self._pending_ack: WebAcknowledgement | None = None
        self._state: dict[str, Any] = {
            "connection": "starting",
            "message": "Waiting for the monitor to start.",
            "consecutive_failures": 0,
            "printer": {"state": "unknown", "detail": "", "uptime_seconds": 0.0},
            "current_detection": None,
            "last_detection": None,
        }

        app = web.Application(middlewares=[self._security_headers])
        app.add_routes(
            [
                web.get("/", self._index),
                web.get("/style.css", self._static),
                web.get("/app.js", self._static),
                web.get("/api/state", self._get_state),
                web.get("/api/frame.jpg", self._get_frame),
                web.post("/api/pause", self._pause),
                web.post("/api/resume", self._resume),
                web.post("/api/acknowledge", self._acknowledge),
                web.post("/api/debug-detection", self._set_debug_detection),
                web.get("/api/healthz", self._health),
            ]
        )
        self._app = app

    @web.middleware
    async def _security_headers(
        self, request: web.Request, handler: Any
    ) -> web.StreamResponse:
        """Apply browser protections and reject cross-site control requests."""
        if request.method == "POST":
            if request.headers.get("X-Scythe-Request") != "1":
                raise web.HTTPForbidden(text="Missing Scythe request header.")
            origin = request.headers.get("Origin")
            expected = f"{request.scheme}://{request.host}"
            if origin and origin != expected:
                raise web.HTTPForbidden(text="Cross-origin control request rejected.")

        response = await handler(request)
        response.headers.update(
            {
                "Content-Security-Policy": (
                    "default-src 'self'; img-src 'self'; style-src 'self'; "
                    "script-src 'self'; connect-src 'self'"
                ),
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
            }
        )
        return response

    async def start(self) -> None:
        """Start the dashboard server once."""
        if self._runner is not None:
            return
        self._runner = web.AppRunner(self._app, access_log=None)
        try:
            await self._runner.setup()
            site = web.TCPSite(
                self._runner,
                host=self._settings.host,
                port=self._settings.port,
            )
            await site.start()
        except Exception:
            await self._runner.cleanup()
            self._runner = None
            raise
        log.info(
            "Web dashboard listening on http://%s:%d",
            self._settings.host,
            self._settings.port,
        )

    async def close(self) -> None:
        """Stop the dashboard server if it was started."""
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def notify(self, event: Event) -> WebAcknowledgement | None:
        """Publish a monitor event to browser clients."""
        self._revision += 1
        if isinstance(event, MonitoringStarted):
            self._state.update(
                connection="online",
                message="Monitoring started.",
                consecutive_failures=0,
            )
        elif isinstance(event, MonitoringStopped):
            self._state.update(
                connection="stopped",
                message=event.reason,
            )
            self._state["printer"]["uptime_seconds"] = event.uptime_seconds
        elif isinstance(event, StatusUpdate):
            self._pending_ack = None
            self._state.update(
                connection="online",
                message="",
                consecutive_failures=0,
                current_detection=None,
            )
            self._state["printer"] = {
                "state": event.state.value,
                "detail": event.detail,
                "uptime_seconds": event.uptime_seconds,
            }
            await self._store_frame(event.image)
        elif isinstance(event, (SpaghettiDetected, DebugDetection)):
            is_debug = isinstance(event, DebugDetection)
            result = event.result
            boxes = [
                {
                    "x1": box.x1,
                    "y1": box.y1,
                    "x2": box.x2,
                    "y2": box.y2,
                    "confidence": box.confidence,
                    "class_id": box.class_id,
                }
                for box in result.boxes
            ]
            detection = {
                "detected_at": time(),
                "count": result.count,
                "max_confidence": result.max_confidence,
                "duration_seconds": result.duration_seconds,
                "debug": is_debug,
                "paused": False if is_debug else event.paused,
                "pause_requested": False if is_debug else event.pause_requested,
                "boxes": boxes,
            }
            self._state.update(
                connection="online",
                message=("Idle debug detection." if is_debug else "Spaghetti detected."),
                current_detection=detection,
                last_detection=detection,
            )
            if is_debug:
                self._state["printer"] = {
                    "state": event.state.value,
                    "detail": "Idle debug detection found spaghetti.",
                    "uptime_seconds": event.uptime_seconds,
                }
                await self._store_frame(result.image)
            else:
                self._state["printer"]["state"] = "paused" if event.paused else "printing"
                await self._store_frame(result.image or event.annotated_image)
                self._pending_ack = WebAcknowledgement()
                return self._pending_ack
        elif isinstance(event, ImageUnavailable):
            self._state.update(
                connection="camera_unavailable",
                message=event.reason,
                consecutive_failures=event.consecutive_failures,
            )
        elif isinstance(event, MonitorError):
            self._state.update(
                connection="error",
                message=event.message,
                consecutive_failures=event.consecutive_failures,
            )
        return None

    async def _store_frame(self, image: Image.Image | None) -> None:
        """Encode and retain one browser-ready JPEG."""
        if image is None:
            return

        def encode() -> bytes:
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="JPEG", quality=88)
            return buffer.getvalue()

        self._frame = await asyncio.to_thread(encode)
        self._frame_meta = {
            "url": f"/api/frame.jpg?v={self._revision}",
            "captured_at": time(),
            "width": image.width,
            "height": image.height,
        }

    def state_payload(self) -> dict[str, Any]:
        """Return the JSON-ready state used by the API and focused tests."""
        pending = self._pending_ack is not None
        return {
            "revision": self._revision,
            "server_time": time(),
            "loop_interval": self._loop_interval,
            "connection": self._state["connection"],
            "message": self._state["message"],
            "consecutive_failures": self._state["consecutive_failures"],
            "printer": self._state["printer"],
            "frame": self._frame_meta,
            "current_detection": self._state["current_detection"],
            "last_detection": self._state["last_detection"],
            "pending_acknowledgement": pending,
            "debug_detection_enabled": self._debug_detection.enabled,
        }

    async def _index(self, _request: web.Request) -> web.FileResponse:
        """Serve the dashboard shell."""
        return web.FileResponse(STATIC_DIR / "index.html")

    async def _static(self, request: web.Request) -> web.FileResponse:
        """Serve one allow-listed static asset."""
        return web.FileResponse(STATIC_DIR / request.path.lstrip("/"))

    async def _get_state(self, _request: web.Request) -> web.Response:
        """Return the latest event snapshot."""
        return web.json_response(
            self.state_payload(), headers={"Cache-Control": "no-store"}
        )

    async def _get_frame(self, request: web.Request) -> web.Response:
        """Return the latest frame with revision-based caching."""
        if self._frame is None:
            raise web.HTTPNotFound(text="No camera frame is available yet.")
        etag = str(self._revision)
        if request.headers.get("If-None-Match") == f'"{etag}"':
            return web.Response(status=304)
        return web.Response(
            body=self._frame,
            content_type="image/jpeg",
            headers={"Cache-Control": "no-cache", "ETag": f'"{etag}"'},
        )

    async def _pause(self, _request: web.Request) -> web.Response:
        """Pause an active print."""
        return web.json_response({"ok": await asyncio.to_thread(self._printer.pause)})

    async def _resume(self, _request: web.Request) -> web.Response:
        """Resume a paused print."""
        return web.json_response({"ok": await asyncio.to_thread(self._printer.resume)})

    async def _acknowledge(self, _request: web.Request) -> web.Response:
        """Acknowledge the active detection, if one is waiting."""
        if self._pending_ack is None:
            raise web.HTTPConflict(text="No detection is waiting for acknowledgement.")
        self._pending_ack.acknowledge()
        self._pending_ack = None
        self._revision += 1
        return web.json_response({"ok": True})

    async def _set_debug_detection(self, request: web.Request) -> web.Response:
        """Enable or disable runtime-only detection while the printer is idle."""
        try:
            payload = await request.json()
        except ValueError as exc:
            raise web.HTTPBadRequest(text="Expected a JSON request body.") from exc
        enabled = payload.get("enabled") if isinstance(payload, dict) else None
        if not isinstance(enabled, bool):
            raise web.HTTPBadRequest(text="enabled must be true or false.")

        self._debug_detection.set_enabled(enabled)
        current = self._state["current_detection"]
        if not enabled and current and current.get("debug"):
            self._state.update(current_detection=None, message="")
        self._revision += 1
        return web.json_response({"ok": True, "enabled": enabled})

    async def _health(self, _request: web.Request) -> web.Response:
        """Return a cheap process health response."""
        return web.json_response({"status": "ok"})


__all__ = ["WebAcknowledgement", "WebNotifier"]
