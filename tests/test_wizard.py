from __future__ import annotations

import json

import pytest

from src.config import load_settings
from src.wizard import PROMPTS, migrate_legacy, run_wizard


def answers(**overrides) -> list[str]:
    """One answer per prompt, blank (keep current) unless overridden."""
    return [
        overrides.get(prompt.path, "")
        for _section, prompts in PROMPTS
        for prompt in prompts
    ]


def scripted(values: list[str]):
    queue = list(values)
    return lambda _prompt: queue.pop(0) if queue else ""


@pytest.fixture
def no_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr("src.paths.LEGACY_CONFIG_PATH", tmp_path / "absent.py")


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"x")
    return path


class TestWizard:
    def test_writes_valid_json(self, tmp_path, no_legacy):
        target = tmp_path / "settings.json"
        run_wizard(
            target,
            input_fn=scripted(
                answers(
                    **{
                        "discord.bot_token": "tok",
                        "discord.log_channel_id": "123",
                        "discord.ping_user_id": "456",
                        "printer.url": "http://p.local/",
                    }
                )
            ),
            output_fn=lambda *_: None,
        )
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["discord"]["bot_token"] == "tok"
        assert data["discord"]["log_channel_id"] == 123
        assert data["printer"]["url"] == "http://p.local/"

    def test_blank_answer_keeps_the_default(self, tmp_path, no_legacy):
        target = tmp_path / "settings.json"
        run_wizard(target, input_fn=scripted(answers()), output_fn=lambda *_: None)
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["printer"]["webcam_name"] == "Bed"
        assert data["target_loop_time"] == 30.0

    def test_no_for_a_boolean_actually_means_no(self, tmp_path, no_legacy):
        """The old prompt ran `bool("False")`, which is True."""
        target = tmp_path / "settings.json"
        run_wizard(
            target,
            input_fn=scripted(answers(**{"pause_on_spaghetti": "False"})),
            output_fn=lambda *_: None,
        )
        assert json.loads(target.read_text())["pause_on_spaghetti"] is False

    def test_reprompts_on_invalid_input(self, tmp_path, no_legacy):
        target = tmp_path / "settings.json"
        queue = answers(**{"discord.log_channel_id": "not-a-number"})
        # Insert the retry answer directly after the bad one.
        index = queue.index("not-a-number")
        queue.insert(index + 1, "777")

        prompts_seen: list[str] = []

        def record(text):
            prompts_seen.append(text)
            return queue.pop(0) if queue else ""

        run_wizard(target, input_fn=record, output_fn=lambda *_: None)
        assert json.loads(target.read_text())["discord"]["log_channel_id"] == 777
        assert sum("Log channel id" in p for p in prompts_seen) == 2

    def test_second_run_preserves_existing_values(self, tmp_path, no_legacy):
        target = tmp_path / "settings.json"
        run_wizard(
            target,
            input_fn=scripted(answers(**{"discord.bot_token": "first"})),
            output_fn=lambda *_: None,
        )
        run_wizard(target, input_fn=scripted(answers()), output_fn=lambda *_: None)
        assert json.loads(target.read_text())["discord"]["bot_token"] == "first"

    def test_invalid_config_is_still_written_with_a_warning(self, tmp_path, no_legacy):
        target = tmp_path / "settings.json"
        messages: list[str] = []
        run_wizard(target, input_fn=scripted(answers()), output_fn=messages.append)
        assert target.exists()
        assert any("not yet usable" in m for m in messages)


class TestMigration:
    def test_legacy_settings_become_json(self, tmp_path, monkeypatch, model):
        legacy = tmp_path / "settings.py"
        legacy.write_text(
            "\n".join(
                [
                    "discord_bot_token = 'legacy'",
                    "discord_ping_userid = 1",
                    "discord_log_channel_id = 2",
                    "printer_url = 'http://legacy.local/'",
                    "use_onnx = False",
                    "pause_on_spaghetti = False",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr("src.paths.LEGACY_CONFIG_PATH", legacy)

        target = migrate_legacy(tmp_path / "settings.json")
        assert target is not None

        monkeypatch.setattr("src.config.DetectionSettings.validate", lambda self: None)
        settings = load_settings(target, env={})
        assert settings.discord.bot_token == "legacy"
        assert settings.pause_on_spaghetti is False

    def test_does_not_clobber_an_existing_config(self, tmp_path, monkeypatch):
        legacy = tmp_path / "settings.py"
        legacy.write_text("printer_url = 'http://legacy.local/'", encoding="utf-8")
        monkeypatch.setattr("src.paths.LEGACY_CONFIG_PATH", legacy)

        existing = tmp_path / "settings.json"
        existing.write_text("{}", encoding="utf-8")
        assert migrate_legacy(existing) is None

    def test_nothing_to_migrate(self, tmp_path, no_legacy):
        assert migrate_legacy(tmp_path / "settings.json") is None
