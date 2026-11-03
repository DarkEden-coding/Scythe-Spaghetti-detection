"""Web setup: login, setup routes, and the system helper wrapper."""

from __future__ import annotations

import asyncio
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.detection.switchable import SwitchableDetector
from src.monitor import DebugDetectionControl
from src.notify.web_notifier import WebNotifier
from src.settings_control import SettingsControl
from src.setup_service import System, check_password, hash_password
from tests.conftest import FakeDetector, FakePrinter

HEADERS = {"X-Scythe-Request": "1"}


class FakeSystem(System):
    def __init__(self, tmp_path):
        super().__init__(
            helper=tmp_path / "helper",
            nginx_sites=(tmp_path / "mainsail",),
            navi_file=tmp_path / "config" / ".theme" / "navi.json",
        )
        self.ran = []

    def helper_available(self):
        return True

    def run_helper(self, *args, timeout=300):
        self.ran.append(args)
        if args[0] == "mainsail-link":
            self.nginx_sites[0].write_text("location /scythe/ {}")
        return ""

    def tailscale(self):
        return {"installed": True, "state": "Running", "auth_url": "",
                "dns_name": "printer.tailnet.ts.net"}


@pytest.fixture
def config_file(tmp_path):
    return tmp_path / "settings.json"


@pytest.fixture
async def client(settings, config_file, tmp_path):
    settings.setup_complete = False
    control = SettingsControl(
        settings, SwitchableDetector(lambda _d: FakeDetector(), settings.detection),
        config_file, env={},
    )
    notifier = WebNotifier(settings.web, FakePrinter(), 30, DebugDetectionControl(), control)
    restarts = []
    notifier._setup._restart = lambda: restarts.append(True)
    notifier._setup._system = FakeSystem(tmp_path)
    (tmp_path / "mainsail").write_text("server {}")
    (tmp_path / "config").mkdir()
    client = TestClient(TestServer(notifier._app))
    await client.start_server()
    client.restarts = restarts
    client.system = notifier._setup._system
    yield client
    await client.close()


def saved(config_file):
    return json.loads(config_file.read_text())


async def wait_for_task(client):
    for _ in range(50):
        status = await (await client.get("/api/setup")).json()
        if not status["task"]["running"]:
            return status
        await asyncio.sleep(0.01)
    raise AssertionError("task never finished")


class TestPassword:
    def test_hash_round_trip(self):
        stored = hash_password("correct horse")
        assert check_password("correct horse", stored)
        assert not check_password("wrong", stored)
        assert not check_password("x", "garbage")

    async def test_open_until_a_password_is_set(self, client, config_file):
        assert (await client.get("/api/state")).status == 200

        response = await client.post(
            "/api/setup/password", json={"password": "hunter22"}, headers=HEADERS
        )
        assert response.status == 200
        assert saved(config_file)["web"]["password_hash"].startswith("pbkdf2_sha256$")
        # The browser that set it stays logged in.
        assert (await client.get("/api/state")).status == 200

        client.session.cookie_jar.clear()
        assert (await client.get("/api/state")).status == 401
        page = await client.get("/", allow_redirects=False)
        assert page.status == 302 and page.headers["Location"] == "login"
        assert (await client.get("/login")).status == 200

        wrong = await client.post("/api/login", json={"password": "nope"}, headers=HEADERS)
        assert wrong.status == 401
        right = await client.post(
            "/api/login", json={"password": "hunter22"}, headers=HEADERS
        )
        assert right.status == 200
        assert (await client.get("/api/state")).status == 200

    async def test_short_password_rejected(self, client):
        response = await client.post(
            "/api/setup/password", json={"password": "short"}, headers=HEADERS
        )
        assert response.status == 400


class TestSetup:
    async def test_finish_saves_and_restarts(self, client, config_file, settings):
        response = await client.post("/api/setup/finish", json={}, headers=HEADERS)
        assert response.status == 200
        assert saved(config_file)["setup_complete"] is True
        await asyncio.sleep(0.6)
        assert client.restarts == [True]

    async def test_printer_needs_a_camera(self, client):
        response = await client.post(
            "/api/setup/printer", json={"url": "http://p.local/"}, headers=HEADERS
        )
        assert response.status == 400

    async def test_printer_is_saved(self, client, config_file):
        response = await client.post(
            "/api/setup/printer",
            json={"url": "http://p.local/", "webcam": "Nozzle"},
            headers=HEADERS,
        )
        assert response.status == 200
        assert saved(config_file)["printer"] == {
            "url": "http://p.local/", "webcam_name": "Nozzle"
        }

    async def test_serve_sets_the_public_url(self, client, config_file):
        response = await client.post(
            "/api/setup/system", json={"action": "tailscale-serve"}, headers=HEADERS
        )
        assert response.status == 200
        status = await wait_for_task(client)
        assert client.system.ran == [("tailscale-serve", "8090")]
        assert status["public_url"] == "https://printer.tailnet.ts.net/"
        assert saved(config_file)["web"]["public_url"] == status["public_url"]

    async def test_mainsail_link_adds_sidebar_entry(self, client):
        await client.post(
            "/api/setup/system", json={"action": "mainsail-link"}, headers=HEADERS
        )
        status = await wait_for_task(client)
        mainsail = status["system"]["mainsail"]
        assert mainsail["linked"] and mainsail["sidebar"]
        assert status["system"]["tailnet_url"].endswith("/scythe/")

        await client.post(
            "/api/setup/system", json={"action": "mainsail-unlink"}, headers=HEADERS
        )
        await wait_for_task(client)
        assert not client.system.navi_file.exists()

    async def test_unknown_action_rejected(self, client):
        response = await client.post(
            "/api/setup/system", json={"action": "rm -rf"}, headers=HEADERS
        )
        assert response.status == 400
