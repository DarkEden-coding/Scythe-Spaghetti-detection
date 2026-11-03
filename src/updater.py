"""Optional self-update to the newest release tag.

Installs on a Klipper host are updated by Moonraker's update manager instead
(the installer registers Scythe there on its ``stable`` channel, which also
follows release tags). This updater covers every other machine.

Conservative by design: the setting is honoured, a dirty tree aborts the
update instead of destroying it, Scythe only ever moves forward to a newer
``vX.Y.Z`` tag, and privilege escalation is never attempted.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

GIT_TIMEOUT = 120
PIP_TIMEOUT = 1800
RELEASE_TAG = re.compile(r"^v\d+\.\d+\.\d+$")


@dataclass(frozen=True)
class UpdateResult:
    """What the updater did."""

    updated: bool = False
    restart_required: bool = False
    message: str = ""
    changed_files: tuple[str, ...] = ()


class Updater:
    """Moves the checkout to the newest release tag."""

    def __init__(self, repo_root: Path, enabled: bool = False):
        self._root = Path(repo_root)
        self._enabled = enabled

    # -- git plumbing ------------------------------------------------------ #

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self._root,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
            check=False,
        )

    def _git_ok(self, *args: str) -> str | None:
        """Run a git command, returning stripped stdout or ``None`` on failure."""
        result = self._git(*args)
        if result.returncode != 0:
            log.debug("git %s failed: %s", " ".join(args), result.stderr.strip())
            return None
        return result.stdout.strip()

    @property
    def is_git_repo(self) -> bool:
        return self._git_ok("rev-parse", "--git-dir") is not None

    def has_local_changes(self) -> bool:
        # Untracked files (settings.json, data/, the model) are ignored.
        return bool(self._git_ok("status", "--porcelain", "--untracked-files=no"))

    def latest_tag(self) -> str | None:
        tags = self._git_ok("tag", "--list", "v*", "--sort=-v:refname") or ""
        return next((t for t in tags.splitlines() if RELEASE_TAG.match(t)), None)

    # -- public API -------------------------------------------------------- #

    def check_and_apply(self) -> UpdateResult:
        """Update the checkout if configured, safe, and a newer release exists."""
        if not self._enabled:
            log.debug("Auto-update disabled; skipping.")
            return UpdateResult(message="Auto-update is disabled.")

        if not self.is_git_repo:
            return UpdateResult(
                message=f"{self._root} is not a git checkout; cannot auto-update."
            )

        if self.has_local_changes():
            log.warning(
                "Local modifications present — skipping auto-update rather than "
                "discarding them. Commit or stash them to resume updates."
            )
            return UpdateResult(message="Skipped: the working tree has local changes.")

        if self._git("fetch", "--quiet", "--tags", "origin").returncode != 0:
            return UpdateResult(message="Skipped: `git fetch` failed.")

        tag = self.latest_tag()
        if tag is None:
            return UpdateResult(message="Skipped: no release has been published yet.")

        before = self._git_ok("rev-parse", "HEAD")
        target = self._git_ok("rev-parse", f"{tag}^{{commit}}")
        if before == target:
            log.info("Already on the newest release (%s).", tag)
            return UpdateResult(message=f"Already up to date ({tag}).")
        # Never move backwards, e.g. from a development commit past the tag.
        if self._git("merge-base", "--is-ancestor", "HEAD", tag).returncode != 0:
            return UpdateResult(message=f"Skipped: this checkout is not behind {tag}.")

        log.info("Updating to release %s.", tag)
        reset = self._git("reset", "--hard", "--quiet", tag)
        if reset.returncode != 0:
            log.error("Update failed: %s", reset.stderr.strip())
            return UpdateResult(message=f"`git reset` failed: {reset.stderr.strip()}")

        changed = self._changed_files(before, target)
        if "requirements.txt" in changed:
            self._install_requirements()
        restart = any(
            name.endswith(".py") or name == "requirements.txt" for name in changed
        )
        log.info("Updated to %s (%d file(s) changed).", tag, len(changed))
        return UpdateResult(
            updated=True,
            restart_required=restart,
            message=f"Updated to {tag}.",
            changed_files=changed,
        )

    def _install_requirements(self) -> None:
        """Install new dependencies into the running interpreter's environment."""
        log.info("Dependencies changed; installing them.")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            cwd=self._root,
            capture_output=True,
            text=True,
            timeout=PIP_TIMEOUT,
            check=False,
        )
        if result.returncode != 0:
            log.error("Installing dependencies failed: %s", result.stderr.strip())

    def _changed_files(self, before: str | None, after: str | None) -> tuple[str, ...]:
        if not before or not after or before == after:
            return ()
        diff = self._git_ok("diff", "--name-only", before, after)
        return tuple(line for line in (diff or "").splitlines() if line)


__all__ = ["UpdateResult", "Updater"]
