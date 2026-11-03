"""Dashboard-editable settings: validation, hot swap, persistence, pricing."""

from __future__ import annotations

import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.config import ConfigError
from src.detection import pricing
from src.detection.switchable import SwitchableDetector
from src.events import StatusUpdate
from src.monitor import DebugDetectionControl, MonitorLoop
from src.notify.web_notifier import WebNotifier
from src.settings_control import SettingsControl
from tests.conftest import FakeDetector, FakePrinter, RecordingNotifier, run_briefly


class NamedDetector(FakeDetector):
    def __init__(self, detection):
        super().__init__()
        self.detection = detection
        self.model_name = detection.backend
        self.loads = 0

    def load(self):
        self.loads += 1


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"discord": {"bot_token": "from-file"}}))
    return path


@pytest.fixture
def control(settings, config_file):
    settings.target_loop_time = 30.0
    detector = SwitchableDetector(NamedDetector, settings.detection)
    return SettingsControl(settings, detector, config_file, env={})


DECISION = {"backend": "decision", "api_key": "sk-or-v1-abcdef123456"}


class TestUpdate:
    async def test_switches_backend_and_saves_only_changed_keys(
        self, control, settings, config_file
    ):
        await control.update(DECISION)

        assert settings.detection.backend == "decision"
        assert control._detector.current.detection is settings.detection
        assert control._detector.current.loads == 1
        saved = json.loads(config_file.read_text())
        assert saved["discord"] == {"bot_token": "from-file"}
        assert saved["detection"]["decision"]["api_key"] == "sk-or-v1-abcdef123456"
        assert "printer" not in saved

    async def test_never_returns_the_key(self, control):
        public = await control.update(DECISION)

        assert "api_key" not in public["values"]
        assert public["api_key_set"] is True
        assert public["api_key_hint"] == "…3456"
        assert "sk-or" not in json.dumps(public)

    async def test_blank_key_keeps_the_saved_one(self, control, settings):
        await control.update(DECISION)
        await control.update({"api_key": "", "model": "typesafe/jev-1.14"})

        assert settings.detection.decision.api_key == "sk-or-v1-abcdef123456"
        assert settings.detection.decision.model == "typesafe/jev-1.14"

    async def test_discord_change_is_saved_and_needs_restart(
        self, control, settings, config_file
    ):
        settings.discord.bot_token = ""
        settings.discord.log_channel_id = 0
        with pytest.raises(ConfigError, match="channel"):
            await control.update({"discord_token": "MTIz.abc.secret-token"})
        assert control.restart_required is False

        public = await control.update(
            {
                "discord_token": "MTIz.abc.secret-token",
                "discord_channel_id": "123",
                "discord_user_id": "456",
            }
        )

        assert public["restart_required"] is True
        assert public["discord_token_hint"] == "…oken"
        assert "secret-token" not in json.dumps(public)
        assert json.loads(config_file.read_text())["discord"]["log_channel_id"] == 123

    async def test_interval_change_does_not_rebuild_detector(self, control, settings):
        before = control._detector.current
        await control.update({"capture_interval": 20})

        assert settings.target_loop_time == 20
        assert control._detector.current is before

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"backend": "decision"}, "api_key"),
            ({"capture_interval": 1}, "at least"),
            ({"capture_interval": 30, "detection_interval": 10}, "shorter"),
            ({"threshold": "high"}, "Invalid value"),
            ({"bot_token": "x"}, "Unknown"),
        ],
    )
    async def test_rejects_bad_changes_without_applying(
        self, control, settings, config_file, changes, message
    ):
        before = config_file.read_text()
        with pytest.raises(ConfigError, match=message):
            await control.update(changes)

        assert settings.detection.backend == "yolo"
        assert settings.target_loop_time == 30.0
        assert config_file.read_text() == before

    async def test_failed_detector_build_is_reported(self, settings, config_file):
        def broken(detection):
            if detection.backend == "decision":
                raise RuntimeError("no network")
            return NamedDetector(detection)

        control = SettingsControl(
            settings, SwitchableDetector(broken, settings.detection), config_file, env={}
        )
        with pytest.raises(ConfigError, match="no network"):
            await control.update(DECISION)
        assert settings.detection.backend == "yolo"

    def test_reports_env_locked_fields(self, settings, config_file):
        control = SettingsControl(
            settings,
            SwitchableDetector(NamedDetector, settings.detection),
            config_file,
            env={"SCYTHE_DETECTION_DECISION_API_KEY": "x"},
        )
        assert control.public()["env_locked"] == ["api_key"]


class TestPricing:
    def test_detection_rate_is_capped_by_capture_rate(self):
        assert pricing.detections_per_hour(60, 0) == 60
        assert pricing.detections_per_hour(30, 120) == 30

    def test_estimate_sources(self):
        table = pricing.estimate("openrouter", "~liquid/d1-ultra-latest", 60, 0)
        assert table.source == "table"
        assert table.cost_per_hour == pytest.approx(60 * 0.0000454)

        measured = pricing.estimate("openrouter", "x/unknown", 60, 0, measured_cost=0.001)
        assert (measured.source, measured.cost_per_hour) == (
            "measured",
            pytest.approx(0.06),
        )

        assert pricing.estimate("openrouter", "x/unknown", 60, 0).cost_per_hour is None
        assert pricing.estimate("ollama", "d3:4b", 60, 0).cost_per_hour == 0.0

    async def test_public_estimate_follows_settings(self, control):
        assert control.public()["estimate"] is None
        public = await control.update({**DECISION, "detection_interval": 120})
        assert public["estimate"]["detections_per_hour"] == 30


class TestDetectionInterval:
    async def test_skips_detection_until_due(self, settings, frame):
        settings.target_loop_time = 0.01
        settings.detection.interval = 60
        notifier, detector = RecordingNotifier(), FakeDetector()
        loop = MonitorLoop(FakePrinter(frame), detector, notifier, settings)
        await run_briefly(loop, 0.05)

        assert detector.calls == 1
        assert any("next detection" in s.detail for s in notifier.of_type(StatusUpdate))


class TestRoutes:
    async def test_get_and_post_settings(self, control):
        notifier = WebNotifier(
            control.settings.web, FakePrinter(), 30, DebugDetectionControl(), control
        )
        client = TestClient(TestServer(notifier._app))
        await client.start_server()
        try:
            response = await client.get("/api/settings")
            assert (await response.json())["values"]["backend"] == "yolo"

            headers = {"X-Scythe-Request": "1"}
            response = await client.post(
                "/api/settings", json={"threshold": 2}, headers=headers
            )
            assert response.status == 400
            assert "threshold" in (await response.json())["error"]

            response = await client.post("/api/settings", json=DECISION, headers=headers)
            assert (await response.json())["settings"]["values"]["backend"] == "decision"
        finally:
            await client.close()
