"""Focused checks for the event-backed web dashboard."""

from __future__ import annotations

import asyncio

from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from src.config import WebSettings
from src.detection.results import DetectionBox, DetectionResult
from src.events import DebugDetection, SpaghettiDetected, StatusUpdate
from src.monitor import DebugDetectionControl
from src.notify.base import Acknowledgement
from src.notify.web_notifier import WebNotifier
from src.printer.base import PrintState
from tests.conftest import FakePrinter


async def test_status_event_updates_frame_without_another_camera_read() -> None:
    """The dashboard must reuse the monitor frame rather than polling Moonraker."""
    printer = FakePrinter()
    notifier = WebNotifier(
        WebSettings(), printer, loop_interval=30, debug_detection=DebugDetectionControl()
    )

    await notifier.notify(
        StatusUpdate(
            uptime_seconds=90,
            state=PrintState.PRINTING,
            image=Image.new("RGB", (320, 240), "black"),
            detail="No spaghetti (0.4s inference).",
        )
    )

    state = notifier.state_payload()
    assert state["printer"]["state"] == "printing"
    assert state["frame"]["width"] == 320
    assert state["current_detection"] is None
    assert printer.snapshot_calls == 0


async def test_detection_exposes_boxes_and_web_acknowledgement() -> None:
    """A browser acknowledgement should satisfy the monitor protocol."""
    notifier = WebNotifier(
        WebSettings(),
        FakePrinter(),
        loop_interval=30,
        debug_detection=DebugDetectionControl(),
    )
    result = DetectionResult(
        boxes=(DetectionBox(10, 20, 110, 140, 0.91, 1),),
        image=Image.new("RGB", (320, 240), "black"),
        duration_seconds=0.3,
    )

    ack = await notifier.notify(
        SpaghettiDetected(
            result=result,
            annotated_image=None,
            paused=True,
            pause_requested=True,
        )
    )

    assert isinstance(ack, Acknowledgement)
    state = notifier.state_payload()
    assert state["pending_acknowledgement"] is True
    assert state["current_detection"]["boxes"][0]["confidence"] == 0.91

    ack.acknowledge()
    assert await asyncio.wait_for(ack.wait(), timeout=0.1) is True


async def test_debug_detection_is_display_only() -> None:
    """Idle debug detections should publish without an acknowledgement."""
    control = DebugDetectionControl()
    control.set_enabled(True)
    notifier = WebNotifier(WebSettings(), FakePrinter(), 30, control)
    result = DetectionResult(
        boxes=(DetectionBox(10, 20, 110, 140, 0.91, 1),),
        image=Image.new("RGB", (320, 240), "black"),
        duration_seconds=0.3,
    )

    ack = await notifier.notify(
        DebugDetection(result, PrintState.STANDBY, uptime_seconds=5)
    )

    state = notifier.state_payload()
    assert ack is None
    assert state["current_detection"]["debug"] is True
    assert state["printer"]["state"] == "standby"
    assert state["pending_acknowledgement"] is False
    assert state["debug_detection_enabled"] is True


async def test_debug_detection_api_updates_shared_control() -> None:
    """The dashboard toggle must change the control read by the monitor."""
    control = DebugDetectionControl()
    notifier = WebNotifier(WebSettings(), FakePrinter(), 30, control)

    async with TestClient(TestServer(notifier._app)) as client:
        response = await client.post(
            "/api/debug-detection",
            json={"enabled": True},
            headers={"X-Scythe-Request": "1"},
        )

    assert response.status == 200
    assert control.enabled is True
    assert notifier.state_payload()["debug_detection_enabled"] is True


def test_web_settings_reject_invalid_port() -> None:
    """Invalid listener ports should fail during config validation."""
    settings = WebSettings(port=70000)

    try:
        settings.validate()
    except Exception as exc:
        assert "between 1 and 65535" in str(exc)
    else:
        raise AssertionError("invalid web port was accepted")
