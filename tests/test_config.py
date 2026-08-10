from __future__ import annotations

import json

import pytest

from src.config import (
    Settings,
    env_var_names,
    load_settings,
    parse_bool,
    read_legacy_settings,
)
from src.errors import ConfigError


class TestParseBool:
    @pytest.mark.parametrize("value", ["true", "True", "yes", "Y", "on", "1", 1, True])
    def test_truthy(self, value):
        assert parse_bool(value) is True

    @pytest.mark.parametrize("value", ["false", "False", "no", "N", "off", "0", 0, False])
    def test_falsy(self, value):
        assert parse_bool(value) is False

    def test_the_old_bug(self):
        """`bool("False")` is True, which silently defeated every toggle."""
        assert bool("False") is True
        assert parse_bool("False") is False

    def test_rejects_nonsense(self):
        with pytest.raises(ConfigError):
            parse_bool("maybe")


class TestLoading:
    def _write(self, tmp_path, payload):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _minimal(self, model_path):
        return {
            "discord": {
                "bot_token": "abc",
                "ping_user_id": 111,
                "log_channel_id": 222,
            },
            "detection": {"model_path": str(model_path), "use_onnx": False},
        }

    @pytest.fixture
    def model(self, tmp_path):
        path = tmp_path / "model.pt"
        path.write_bytes(b"x")
        return path

    def test_round_trip(self, tmp_path, model):
        path = self._write(tmp_path, self._minimal(model))
        settings = load_settings(path, env={})
        assert settings.discord.bot_token == "abc"
        assert settings.printer.webcam_name == "Bed"  # default preserved

    def test_missing_file_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.paths.LEGACY_CONFIG_PATH", tmp_path / "nothing.py")
        with pytest.raises(ConfigError, match="No configuration found"):
            load_settings(tmp_path / "absent.json", env={})

    def test_unknown_key_is_rejected(self, tmp_path, model):
        payload = self._minimal(model) | {"targetlooptime": 5}
        path = self._write(tmp_path, payload)
        with pytest.raises(ConfigError, match="Unknown setting"):
            load_settings(path, env={})

    def test_bad_json(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("{nope", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid JSON"):
            load_settings(path, env={})

    def test_env_overrides_file(self, tmp_path, model):
        path = self._write(tmp_path, self._minimal(model))
        settings = load_settings(
            path,
            env={
                "SCYTHE_DISCORD_BOT_TOKEN": "from-env",
                "SCYTHE_PRINTER_URL": "http://other.local",
                "SCYTHE_PAUSE_ON_SPAGHETTI": "no",
            },
        )
        assert settings.discord.bot_token == "from-env"
        assert settings.printer.url == "http://other.local"
        assert settings.pause_on_spaghetti is False

    def test_env_alone_is_enough(self, tmp_path, model, monkeypatch):
        monkeypatch.setattr("src.paths.LEGACY_CONFIG_PATH", tmp_path / "nothing.py")
        settings = load_settings(
            tmp_path / "absent.json",
            env={
                "SCYTHE_DISCORD_BOT_TOKEN": "t",
                "SCYTHE_DISCORD_PING_USER_ID": "1",
                "SCYTHE_DISCORD_LOG_CHANNEL_ID": "2",
                "SCYTHE_DETECTION_MODEL_PATH": str(model),
                "SCYTHE_DETECTION_USE_ONNX": "false",
            },
        )
        assert settings.discord.log_channel_id == 2

    def test_saved_file_reloads(self, tmp_path, model):
        original = load_settings(self._write(tmp_path, self._minimal(model)), env={})
        target = original.save(tmp_path / "out.json")
        assert load_settings(target, env={}).to_dict() == original.to_dict()


class TestValidation:
    def test_placeholder_token_rejected(self):
        settings = Settings()
        settings.discord.bot_token = "YOUR DISCORD BOT TOKEN"
        with pytest.raises(ConfigError, match="bot_token"):
            settings.discord.validate()

    def test_url_must_have_a_scheme(self):
        settings = Settings()
        settings.printer.url = "mainsailos.local"
        with pytest.raises(ConfigError, match="http"):
            settings.printer.validate()

    def test_base_url_normalises_slashes(self):
        settings = Settings()
        settings.printer.url = "http://printer.local///"
        assert settings.printer.base_url == "http://printer.local/"

    def test_confidence_bounds(self):
        settings = Settings()
        settings.detection.min_confidence = 1.5
        with pytest.raises(ConfigError, match="min_confidence"):
            settings.detection.validate()

    def test_status_mode_bounds(self):
        settings = Settings()
        settings.discord.bot_token = "real-token"
        settings.discord.ping_user_id = 1
        settings.discord.log_channel_id = 2
        settings.discord.status_update_mode = "shout"
        with pytest.raises(ConfigError, match="status_update_mode"):
            settings.discord.validate()

    def test_onnx_path_is_a_sibling(self, tmp_path):
        settings = Settings()
        settings.detection.model_path = tmp_path / "large.pt"
        assert settings.detection.onnx_path == tmp_path / "large.onnx"
        assert settings.detection.active_model_path == settings.detection.onnx_path


class TestLegacyMigration:
    def test_reads_without_executing(self, tmp_path):
        legacy = tmp_path / "settings.py"
        legacy.write_text(
            "\n".join(
                [
                    "discord_bot_token = 'legacy-token'",
                    "discord_ping_userid = 123",
                    "discord_log_channel_id = 456",
                    "target_loop_time = 45",
                    "printer_url = 'http://legacy.local/'",
                    "webcam_name = 'Nozzle'",
                    "pause_on_spaghetti = False",
                    "use_cuda = True",
                    "enable_auto_update = False",
                    "use_onnx = True",
                ]
            ),
            encoding="utf-8",
        )
        values = read_legacy_settings(legacy)
        assert values["discord"]["bot_token"] == "legacy-token"
        assert values["printer"]["webcam_name"] == "Nozzle"
        assert values["detection"]["use_cuda"] is True
        assert values["pause_on_spaghetti"] is False

    def test_side_effects_are_not_executed(self, tmp_path):
        marker = tmp_path / "executed.txt"
        legacy = tmp_path / "settings.py"
        legacy.write_text(
            f"import pathlib\n"
            f"pathlib.Path({str(marker)!r}).write_text('boom')\n"
            f"printer_url = 'http://safe.local/'\n",
            encoding="utf-8",
        )
        values = read_legacy_settings(legacy)
        assert not marker.exists()
        assert values["printer"]["url"] == "http://safe.local/"


def test_env_var_names_cover_nested_fields():
    names = env_var_names()
    assert names["discord.bot_token"] == "SCYTHE_DISCORD_BOT_TOKEN"
    assert names["detection.min_confidence"] == "SCYTHE_DETECTION_MIN_CONFIDENCE"
    assert names["target_loop_time"] == "SCYTHE_TARGET_LOOP_TIME"


class TestPathHandling:
    def test_relative_paths_resolve_against_the_install_not_cwd(
        self, tmp_path, monkeypatch
    ):
        """A config must behave identically under systemd and in a shell."""
        from src import paths
        from src.config import _build

        monkeypatch.chdir(tmp_path)
        settings = _build(Settings, {"detection": {"model_path": "models/x.pt"}})
        assert settings.detection.model_path == paths.PROJECT_ROOT / "models" / "x.pt"

    def test_absolute_paths_are_left_alone(self, tmp_path):
        from src.config import _build

        absolute = tmp_path / "weights.pt"
        settings = _build(Settings, {"detection": {"model_path": str(absolute)}})
        assert settings.detection.model_path == absolute

    def test_install_relative_paths_serialise_as_relative(self):
        from src import paths

        settings = Settings()
        settings.detection.model_path = paths.MODELS_DIR / "largeModel.pt"
        assert settings.to_dict()["detection"]["model_path"] == "models/largeModel.pt"

    def test_external_paths_serialise_absolute(self, tmp_path):
        settings = Settings()
        settings.detection.model_path = tmp_path / "elsewhere.pt"
        assert settings.to_dict()["detection"]["model_path"] == str(
            tmp_path / "elsewhere.pt"
        )
