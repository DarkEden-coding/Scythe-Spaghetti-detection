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
    "rev-parse --abbrev-ref --symbolic-full-name @{u}": (0, "origin/master", ""),
    "fetch": (0, "", ""),
}


@pytest.fixture
def git(monkeypatch):
    def install(responses):
        fake = FakeGit(responses)
        monkeypatch.setattr(subprocess, "run", fake)
        return fake

    return install


class TestGating:
    def test_disabled_by_default_does_nothing(self, git, tmp_path):
        fake = git(CLEAN_REPO)
        result = Updater(tmp_path, enabled=False).check_and_apply()
        assert result.updated is False
        assert fake.calls == [], "the setting must actually be honoured"

    def test_not_a_git_checkout(self, git, tmp_path):
        git({"rev-parse --git-dir": (128, "", "not a repository")})
        result = Updater(tmp_path, enabled=True).check_and_apply()
        assert result.updated is False
        assert "not a git checkout" in result.message


class TestSafety:
    def test_local_changes_abort_the_update(self, git, tmp_path):
        fake = git(CLEAN_REPO | {"status --porcelain": (0, " M settings.json\n", "")})
        result = Updater(tmp_path, enabled=True).check_and_apply()

        assert result.updated is False
        assert "local changes" in result.message
        assert not fake.ran("stash"), "the old updater discarded local work"
        assert not fake.ran("pull")

    def test_never_escalates_to_sudo(self, git, tmp_path):
        fake = git(CLEAN_REPO | {"rev-list --count": (0, "1", "")})
        Updater(tmp_path, enabled=True).check_and_apply()
        assert all("sudo" not in " ".join(c) for c in fake.calls)

    def test_only_fast_forwards(self, git, tmp_path):
        fake = git(CLEAN_REPO | {"rev-list --count": (0, "2", "")})
        Updater(tmp_path, enabled=True).check_and_apply()
        assert fake.ran("pull --ff-only")

    def test_no_upstream_is_skipped(self, git, tmp_path):
        git(
            CLEAN_REPO
            | {
                "rev-parse --abbrev-ref --symbolic-full-name @{u}": (
                    128,
                    "",
                    "no upstream",
                )
            }
        )
        result = Updater(tmp_path, enabled=True).check_and_apply()
        assert result.updated is False
        assert "upstream" in result.message


class TestOutcome:
    def test_up_to_date(self, git, tmp_path):
        git(CLEAN_REPO | {"rev-list --count": (0, "0", "")})
        result = Updater(tmp_path, enabled=True).check_and_apply()
        assert result.updated is False
        assert result.restart_required is False

    def test_python_change_requires_restart(self, git, tmp_path):
        git(
            CLEAN_REPO
            | {
                "rev-list --count": (0, "1", ""),
                "rev-parse HEAD": [(0, "aaaa", ""), (0, "bbbb", "")],
                "diff --name-only": (0, "src/monitor.py\nreadme.md\n", ""),
            }
        )
        result = Updater(tmp_path, enabled=True).check_and_apply()
        assert result.updated is True
        assert result.restart_required is True
        assert "src/monitor.py" in result.changed_files

    def test_docs_only_change_does_not_restart(self, git, tmp_path):
        git(
            CLEAN_REPO
            | {
                "rev-list --count": (0, "1", ""),
                "rev-parse HEAD": [(0, "aaaa", ""), (0, "bbbb", "")],
                "diff --name-only": (0, "readme.md\n", ""),
            }
        )
        result = Updater(tmp_path, enabled=True).check_and_apply()
        assert result.updated is True
        assert result.restart_required is False

    def test_failed_pull_is_reported(self, git, tmp_path):
        git(
            CLEAN_REPO
            | {
                "rev-list --count": (0, "1", ""),
                "pull --ff-only": (1, "", "diverged"),
            }
        )
        result = Updater(tmp_path, enabled=True).check_and_apply()
        assert result.updated is False
        assert "failed" in result.message
