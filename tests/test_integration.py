"""End-to-end wiring against a real HTTP server.

Uses the real :class:`MoonrakerClient` (real ``requests``, real urllib3 retries)
and the real :class:`SpaghettiDetector` with a stand-in model, so the seams
between the packages are exercised rather than mocked away.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

import pytest
from PIL import Image

from src.config import PrinterSettings, Settings
from src.detection.detector import SpaghettiDetector
from src.events import SpaghettiDetected, StatusUpdate
from src.monitor import MonitorLoop
from src.printer.base import PrintState
from src.printer.moonraker import MoonrakerClient
from tests.conftest import RecordingAck, RecordingNotifier, run_briefly
from tests.test_detection import FakeBox, FakeModel


def _jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (1280, 720), (90, 90, 90)).save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeMoonraker(BaseHTTPRequestHandler):
    state = "printing"
    pauses: list[str] = []

    def _send(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler's API
        if "webcams/list" in self.path:
            body = {
                "result": {
                    "webcams": [
                        {"name": "Bed", "snapshot_url": "webcam/?action=snapshot"}
                    ]
                }
            }
            self._send(json.dumps(body).encode(), "application/json")
        elif "print_stats" in self.path:
            body = {"result": {"status": {"print_stats": {"state": type(self).state}}}}
            self._send(json.dumps(body).encode(), "application/json")
        elif "action=snapshot" in self.path:
            self._send(_jpeg(), "image/jpeg")
        else:
            self.send_error(404)

    def do_POST(self):  # noqa: N802
        type(self).pauses.append(self.path)
        self._send(b"{}", "application/json")

    def log_message(self, *_args):
        pass  # keep the test output clean


@pytest.fixture
def moonraker():
    FakeMoonraker.state = "printing"
    FakeMoonraker.pauses = []
    server = HTTPServer(("127.0.0.1", 0), FakeMoonraker)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def printer(moonraker):
    with MoonrakerClient(
        PrinterSettings(url=moonraker, webcam_name="Bed", max_retries=0)
    ) as client:
        yield client


def build_settings(tmp_path, printer_url) -> Settings:
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"x")
    settings = Settings()
    settings.printer.url = printer_url
    settings.detection.model_path = weights
    settings.detection.use_onnx = False
    settings.detection.save_annotated_frames = False
    settings.target_loop_time = 0.01
    return settings


class TestPrinterOverHttp:
    def test_reads_state_and_snapshot(self, printer):
        assert printer.get_state() is PrintState.PRINTING
        image = printer.get_snapshot()
        assert image is not None
        assert image.size == (1280, 720)

    def test_pause_hits_the_right_endpoint(self, printer):
        assert printer.pause() is True
        assert FakeMoonraker.pauses == ["/printer/print/pause"]

    def test_idle_printer_is_not_paused(self, printer):
        FakeMoonraker.state = "standby"
        assert printer.pause() is False
        assert FakeMoonraker.pauses == []


class TestFullLoop:
    async def test_clean_print_reports_status(self, tmp_path, printer, moonraker):
        settings = build_settings(tmp_path, moonraker)
        detector = SpaghettiDetector(settings.detection, model=FakeModel([]))
        notifier = RecordingNotifier()

        await run_briefly(MonitorLoop(printer, detector, notifier, settings), 0.3)

        statuses = notifier.of_type(StatusUpdate)
        assert statuses
        assert statuses[0].state is PrintState.PRINTING
        assert not notifier.of_type(SpaghettiDetected)

    async def test_detection_pauses_the_real_endpoint(self, tmp_path, printer, moonraker):
        settings = build_settings(tmp_path, moonraker)
        model = FakeModel([FakeBox((100, 100, 300, 300), 0.93, 1)])
        detector = SpaghettiDetector(settings.detection, model=model)
        notifier = RecordingNotifier(ack=RecordingAck())

        await run_briefly(MonitorLoop(printer, detector, notifier, settings), 0.3)

        events = notifier.of_type(SpaghettiDetected)
        assert events
        assert events[0].paused is True
        assert events[0].annotated_image is not None
        assert "/printer/print/pause" in FakeMoonraker.pauses

    async def test_printer_going_away_does_not_kill_the_loop(self, tmp_path, moonraker):
        settings = build_settings(tmp_path, moonraker)
        settings.printer.url = "http://127.0.0.1:1/"  # nothing listening
        settings.printer.max_retries = 0
        printer = MoonrakerClient(settings.printer)
        detector = SpaghettiDetector(settings.detection, model=FakeModel([]))
        notifier = RecordingNotifier()

        await run_briefly(MonitorLoop(printer, detector, notifier, settings), 0.3)
        printer.close()
        # No exception, and the loop reported the outage rather than dying.
        assert notifier.events
