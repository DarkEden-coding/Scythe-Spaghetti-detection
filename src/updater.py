"""Optional git self-update.

Deliberately conservative compared to the previous implementation, which ran
unconditionally (the ``enable_auto_update`` setting was defined but never read),
escalated to ``sudo`` on permission errors, and ran ``git stash`` — silently
discarding whatever the operator had edited locally.

Here: the setting is honoured, a dirty tree aborts the update instead of
destroying it, only fast-forwards are accepted, and privilege escalation is
never attempted.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

GIT_TIMEOUT = 120


@dataclass(frozen=True)
class UpdateResult:
    """What the updater did."""

    updated: bool = False
    restart_required: bool = False
    message: str = ""
    changed_files: tuple[str, ...] = ()


class Updater:
    """Fast-forwards the working tree to the tracked upstream branch."""

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
        status = self._git_ok("status", "--porcelain")
        return bool(status)

    def upstream_branch(self) -> str | None:
        return self._git_ok("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")

    def commits_behind(self) -> int:
        count = self._git_ok("rev-list", "--count", "HEAD..@{u}")
        try:
            return int(count) if count else 0
        except ValueError:
            return 0

    # -- public API -------------------------------------------------------- #

    def check_and_apply(self) -> UpdateResult:
        """Update the checkout if configured, safe, and behind upstream."""
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

        if self._git("fetch", "--quiet").returncode != 0:
            return UpdateResult(message="Skipped: `git fetch` failed.")

        if self.upstream_branch() is None:
            return UpdateResult(message="Skipped: no upstream branch is tracked.")

        behind = self.commits_behind()
        if behind == 0:
            log.info("Already up to date.")
            return UpdateResult(message="Already up to date.")

        before = self._git_ok("rev-parse", "HEAD")
        log.info("%d new commit(s) upstream; fast-forwarding.", behind)

        pull = self._git("pull", "--ff-only")
        if pull.returncode != 0:
            log.error("Update failed: %s", pull.stderr.strip())
            return UpdateResult(
                message=f"`git pull --ff-only` failed: {pull.stderr.strip()}"
            )

        after = self._git_ok("rev-parse", "HEAD")
        changed = self._changed_files(before, after)
        restart = any(name.endswith(".py") for name in changed)

        log.info("Updated to %s (%d file(s) changed).", (after or "?")[:8], len(changed))
        return UpdateResult(
            updated=True,
            restart_required=restart,
            message=f"Updated {behind} commit(s).",
            changed_files=changed,
        )

    def _changed_files(self, before: str | None, after: str | None) -> tuple[str, ...]:
        if not before or not after or before == after:
            return ()
        diff = self._git_ok("diff", "--name-only", before, after)
        return tuple(line for line in (diff or "").splitlines() if line)


__all__ = ["UpdateResult", "Updater"]
