from __future__ import annotations

import subprocess

import pytest

from src.updater import Updater


class FakeGit:
    """Records git invocations and replays scripted output."""

    def __init__(self, responses: dict[str, tuple[int, str, str] | list]):
        self.responses = responses
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, args, **kwargs):
        command = tuple(args[1:])  # drop "git"
        self.calls.append(command)
        for prefix, response in self.responses.items():
            if not " ".join(command).startswith(prefix):
                continue
            # A list scripts successive calls (e.g. HEAD before and after a pull).
            if isinstance(response, list):
                response = response.pop(0) if len(response) > 1 else response[0]
            code, out, err = response
            return subprocess.CompletedProcess(args, code, out, err)
        return subprocess.CompletedProcess(args, 0, "", "")

    def ran(self, prefix: str) -> bool:
        return any(" ".join(c).startswith(prefix) for c in self.calls)


CLEAN_REPO = {
    "rev-parse --git-dir": (0, ".git", ""),
    "status --porcelain": (0, "", ""),
    "fetch": (0, "", ""),
    "tag --list": (0, "v0.4.0\nv0.3.1\nnightly\n", ""),
    "rev-parse HEAD": (0, "aaaa", ""),
    "rev-parse v0.4.0": (0, "bbbb", ""),
    "merge-base --is-ancestor": (0, "", ""),
}


@pytest.fixture
def git(monkeypatch):
    def install(responses):
        fake = FakeGit(responses)
        monkeypatch.setattr(subprocess, "run", fake)
        return fake

    return install


def update(tmp_path):
    return Updater(tmp_path, enabled=True).check_and_apply()


class TestGating:
    def test_disabled_by_default_does_nothing(self, git, tmp_path):
        fake = git(CLEAN_REPO)
        result = Updater(tmp_path, enabled=False).check_and_apply()
        assert result.updated is False
        assert fake.calls == [], "the setting must actually be honoured"

    def test_not_a_git_checkout(self, git, tmp_path):
        git({"rev-parse --git-dir": (128, "", "not a repository")})
        assert "not a git checkout" in update(tmp_path).message

    def test_no_release_yet(self, git, tmp_path):
        git(CLEAN_REPO | {"tag --list": (0, "nightly\n", "")})
        assert "no release" in update(tmp_path).message


class TestSafety:
    def test_local_changes_abort_the_update(self, git, tmp_path):
        fake = git(CLEAN_REPO | {"status --porcelain": (0, " M src/app.py\n", "")})
        result = update(tmp_path)

        assert result.updated is False
        assert "local changes" in result.message
        assert not fake.ran("reset")

    def test_never_escalates_to_sudo(self, git, tmp_path):
        fake = git(CLEAN_REPO)
        update(tmp_path)
        assert all("sudo" not in " ".join(c) for c in fake.calls)

    def test_never_moves_backwards(self, git, tmp_path):
        fake = git(CLEAN_REPO | {"merge-base --is-ancestor": (1, "", "")})
        result = update(tmp_path)
        assert result.updated is False
        assert not fake.ran("reset")


class TestOutcome:
    def test_up_to_date(self, git, tmp_path):
        git(CLEAN_REPO | {"rev-parse HEAD": (0, "bbbb", "")})
        result = update(tmp_path)
        assert result.updated is False
        assert "v0.4.0" in result.message

    def test_moves_to_newest_release_tag(self, git, tmp_path):
        fake = git(CLEAN_REPO | {"diff --name-only": (0, "src/monitor.py\n", "")})
        result = update(tmp_path)
        assert fake.ran("reset --hard --quiet v0.4.0")
        assert result.updated is True
        assert result.restart_required is True

    def test_docs_only_change_does_not_restart(self, git, tmp_path):
        git(CLEAN_REPO | {"diff --name-only": (0, "readme.md\n", "")})
        result = update(tmp_path)
        assert result.updated is True
        assert result.restart_required is False

    def test_new_dependencies_are_installed(self, git, tmp_path):
        fake = git(CLEAN_REPO | {"diff --name-only": (0, "requirements.txt\n", "")})
        assert update(tmp_path).restart_required is True
        assert any("pip" in call for call in fake.calls)

    def test_failed_reset_is_reported(self, git, tmp_path):
        git(CLEAN_REPO | {"reset": (1, "", "boom")})
        result = update(tmp_path)
        assert result.updated is False
        assert "failed" in result.message
