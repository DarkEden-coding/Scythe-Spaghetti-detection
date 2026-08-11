from __future__ import annotations

from io import BytesIO

import pytest
import requests
from PIL import Image

from src.config import PrinterSettings
from src.errors import WebcamNotFound
from src.printer.base import PrintState
from src.printer.moonraker import MoonrakerClient


class FakeResponse:
    def __init__(self, payload=None, content=b"", status_code=200):
        self._payload = payload
        self.content = content
        self.status_code = status_code

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    """Routes requests by URL substring."""

    def __init__(self, routes: dict, post_status=200):
        self.routes = routes
        self.requests: list[str] = []
        self.posts: list[str] = []
        self.post_status = post_status
        self.closed = False

    def _match(self, url):
        for fragment, response in self.routes.items():
            if fragment in url:
                return response
        raise requests.ConnectionError(f"no route for {url}")

    def get(self, url, timeout=None):
        self.requests.append(url)
        response = self._match(url)
        if isinstance(response, Exception):
            raise response
        return response

    def post(self, url, timeout=None):
        self.posts.append(url)
        return FakeResponse(status_code=self.post_status)

    def close(self):
        self.closed = True


def jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 24), (10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


WEBCAMS = FakeResponse(
    {"result": {"webcams": [{"name": "Bed", "snapshot_url": "webcam/?action=snapshot"}]}}
)


def printing_state(state="printing"):
    return FakeResponse({"result": {"status": {"print_stats": {"state": state}}}})


@pytest.fixture
def settings():
    return PrinterSettings(url="http://printer.local/", webcam_name="Bed", max_retries=0)


class TestConstruction:
    def test_no_io_at_construction(self, settings):
        """Importing/constructing must not require the printer to be online."""
        session = FakeSession({})
        MoonrakerClient(settings, session=session)
        assert session.requests == []

    def test_url_joining_is_slash_safe(self, settings):
        settings.url = "http://printer.local"
        client = MoonrakerClient(settings, session=FakeSession({}))
        assert client._url("printer/print/pause") == (
            "http://printer.local/printer/print/pause"
        )


class TestWebcam:
    def test_snapshot_url_is_resolved_once(self, settings):
        session = FakeSession(
            {
                "server/webcams/list": WEBCAMS,
                "action=snapshot": FakeResponse(content=jpeg_bytes()),
            }
        )
        client = MoonrakerClient(settings, session=session)

        assert client.get_snapshot() is not None
        assert client.get_snapshot() is not None
        assert sum("webcams/list" in u for u in session.requests) == 1
        snapshots = [url for url in session.requests if "action=snapshot" in url]
        assert snapshots[0] != snapshots[1]
        assert all("_scythe=" in url for url in snapshots)

    def test_unknown_webcam_name_raises(self, settings):
        settings.webcam_name = "Nozzle"
        client = MoonrakerClient(
            settings, session=FakeSession({"server/webcams/list": WEBCAMS})
        )
        with pytest.raises(WebcamNotFound) as excinfo:
            client.get_snapshot()
        assert "Bed" in str(excinfo.value)

    def test_snapshot_failure_returns_none_and_forgets_the_url(self, settings):
        session = FakeSession(
            {
                "server/webcams/list": WEBCAMS,
                "action=snapshot": requests.ConnectionError("down"),
            }
        )
        client = MoonrakerClient(settings, session=session)

        assert client.get_snapshot() is None
        assert client._snapshot_url is None  # rediscovered next time

    def test_undecodable_snapshot_returns_none(self, settings):
        session = FakeSession(
            {
                "server/webcams/list": WEBCAMS,
                "action=snapshot": FakeResponse(content=b"<html>error</html>"),
            }
        )
        assert MoonrakerClient(settings, session=session).get_snapshot() is None


class TestPrintState:
    @pytest.mark.parametrize(
        ("raw", "expected", "active"),
        [
            ("printing", PrintState.PRINTING, True),
            ("standby", PrintState.STANDBY, False),
            ("paused", PrintState.PAUSED, False),
            ("complete", PrintState.COMPLETE, False),
            ("something-new", PrintState.UNKNOWN, False),
        ],
    )
    def test_states(self, settings, raw, expected, active):
        client = MoonrakerClient(
            settings, session=FakeSession({"print_stats": printing_state(raw)})
        )
        assert client.get_state() is expected
        assert client.is_printing() is active

    def test_unreachable_printer_is_unknown_not_an_exception(self, settings):
        client = MoonrakerClient(
            settings,
            session=FakeSession({"print_stats": requests.ConnectionError("down")}),
        )
        assert client.get_state() is PrintState.UNKNOWN

    def test_malformed_payload_is_unknown(self, settings):
        client = MoonrakerClient(
            settings, session=FakeSession({"print_stats": FakeResponse({"result": {}})})
        )
        assert client.get_state() is PrintState.UNKNOWN


class TestPauseAndResume:
    def test_pauses_when_printing(self, settings):
        session = FakeSession({"print_stats": printing_state("printing")})
        assert MoonrakerClient(settings, session=session).pause() is True
        assert session.posts == ["http://printer.local/printer/print/pause"]

    def test_does_not_pause_when_idle(self, settings):
        session = FakeSession({"print_stats": printing_state("standby")})
        assert MoonrakerClient(settings, session=session).pause() is False
        assert session.posts == []

    def test_resumes_only_when_paused(self, settings):
        session = FakeSession({"print_stats": printing_state("paused")})
        assert MoonrakerClient(settings, session=session).resume() is True
        assert session.posts == ["http://printer.local/printer/print/resume"]

    def test_does_not_resume_unknown_or_idle_state(self, settings):
        session = FakeSession({"print_stats": printing_state("standby")})
        assert MoonrakerClient(settings, session=session).resume() is False
        assert session.posts == []

    def test_reports_failure_rather_than_raising(self, settings):
        session = FakeSession(
            {"print_stats": printing_state("printing")}, post_status=500
        )
        assert MoonrakerClient(settings, session=session).pause() is False


def test_context_manager_closes_owned_session(settings):
    session = FakeSession({})
    with MoonrakerClient(settings, session=session):
        pass
    assert session.closed is False, "a caller-supplied session is not ours to close"
