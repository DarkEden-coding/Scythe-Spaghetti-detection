"""Shared fixtures and fakes.

Nothing here touches the network, a model, or Discord — the whole point of the
refactor is that the monitor can be exercised with three small stand-ins.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    DetectionSettings,
    DiscordSettings,
    PrinterSettings,
    Settings,
)
from src.detection.results import DetectionBox, DetectionResult  # noqa: E402
from src.notify.base import BaseNotifier  # noqa: E402
from src.printer.base import PrintState  # noqa: E402


@pytest.fixture
def frame() -> Image.Image:
    return Image.new("RGB", (640, 640), color=(30, 30, 30))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"not-a-real-model")
    return Settings(
        discord=DiscordSettings(bot_token="token", ping_user_id=1, log_channel_id=2),
        printer=PrinterSettings(url="http://printer.local/", webcam_name="Bed"),
        detection=DetectionSettings(
            model_path=weights, use_onnx=False, save_annotated_frames=False
        ),
        target_loop_time=0.01,
    )


class FakePrinter:
    """A :class:`~src.printer.base.PrinterClient` with scripted responses."""

    def __init__(self, image=None, state=PrintState.PRINTING, pause_succeeds=True):
        self.image = image
        self.state = state
        self.pause_succeeds = pause_succeeds
        self.pause_calls = 0
        self.snapshot_calls = 0
        self.closed = False

    def get_snapshot(self):
        self.snapshot_calls += 1
        return self.image

    def get_state(self) -> PrintState:
        return self.state

    def is_printing(self) -> bool:
        return self.state.is_active

    def pause(self) -> bool:
        self.pause_calls += 1
        return self.pause_succeeds

    def close(self) -> None:
        self.closed = True


class FakeDetector:
    """Returns a canned :class:`DetectionResult` without loading anything."""

    def __init__(
        self, result: DetectionResult | None = None, error: Exception | None = None
    ):
        self.result = result if result is not None else DetectionResult()
        self.error = error
        self.calls = 0
        self.model_path = Path("fake-model.pt")

    def load(self) -> None:
        return None

    def detect(self, image) -> DetectionResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


class RecordingAck:
    def __init__(self, result: bool = True):
        self.result = result
        self.waited = False
        self.timeout = None

    async def wait(self, timeout=None) -> bool:
        self.waited = True
        self.timeout = timeout
        return self.result


class RecordingNotifier(BaseNotifier):
    """Captures every event, optionally handing back an acknowledgement."""

    def __init__(self, ack: RecordingAck | None = None, fail_on=()):
        self.events = []
        self.ack = ack
        self.fail_on = tuple(fail_on)
        self.started = False
        self.closed = False

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True

    async def notify(self, event):
        if isinstance(event, self.fail_on):
            raise RuntimeError("channel is down")
        self.events.append(event)
        from src.events import SpaghettiDetected

        if isinstance(event, SpaghettiDetected):
            return self.ack
        return None

    def of_type(self, event_type):
        return [e for e in self.events if isinstance(e, event_type)]


@pytest.fixture
def detection_result(frame) -> DetectionResult:
    return DetectionResult(
        boxes=(DetectionBox(10, 20, 110, 140, 0.91, 1),),
        image=frame.convert("L"),
        duration_seconds=0.2,
    )


async def run_briefly(loop, seconds: float = 0.2) -> None:
    """Run a monitor loop, then stop it."""
    task = asyncio.ensure_future(loop.run())
    await asyncio.sleep(seconds)
    loop.request_stop()
    await asyncio.wait_for(task, timeout=5)
