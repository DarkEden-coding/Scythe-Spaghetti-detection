"""Notification interfaces.

A notifier turns :mod:`src.events` into messages somewhere. It may also
return an :class:`Acknowledgement`, which is how a human tells the monitor it
is safe to resume after a failure — expressed generically so that a Discord
reaction, an email reply, or a web button all fit the same shape.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

from src.events import Event

log = logging.getLogger(__name__)


@runtime_checkable
class Acknowledgement(Protocol):
    """A pending "a human has seen this" signal."""

    async def wait(self, timeout: float | None = None) -> bool:
        """Block until acknowledged. Returns False if the timeout elapsed."""
        ...


class Notifier(Protocol):
    """Delivers events to a channel."""

    async def notify(self, event: Event) -> Acknowledgement | None:
        """Deliver ``event``.

        Returns an :class:`Acknowledgement` when the channel offers a way for a
        human to respond, otherwise ``None``.
        """
        ...

    async def start(self) -> None:
        """Connect / authenticate. Must be idempotent."""
        ...

    async def close(self) -> None:
        """Tear down. Must be safe to call without a successful start."""
        ...


class BaseNotifier:
    """Convenience base providing async-context-manager behaviour."""

    async def start(self) -> None:  # pragma: no cover - trivial
        return None

    async def close(self) -> None:  # pragma: no cover - trivial
        return None

    async def notify(self, event: Event) -> Acknowledgement | None:
        raise NotImplementedError

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.close()


class NullNotifier(BaseNotifier):
    """Logs events and acknowledges nothing.

    Used by ``scythe check`` and by tests, and as a safe fallback so the
    monitor can run with no channels configured.
    """

    async def notify(self, event: Event) -> Acknowledgement | None:
        log.info("event: %s", type(event).__name__)
        return None


class _AnyAcknowledgement:
    """Resolves as soon as any underlying acknowledgement resolves."""

    def __init__(self, acks: list[Acknowledgement]):
        self._acks = acks

    async def wait(self, timeout: float | None = None) -> bool:
        if not self._acks:
            return True
        tasks = [asyncio.ensure_future(ack.wait(timeout)) for ack in self._acks]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            return any(task.result() for task in done if not task.cancelled())
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()


class CompositeNotifier(BaseNotifier):
    """Fans events out to several notifiers.

    One failing channel must not take down the others or the monitor, so
    per-notifier errors are logged and swallowed.
    """

    def __init__(self, notifiers: list[Notifier]):
        self._notifiers = list(notifiers)

    def __len__(self) -> int:
        return len(self._notifiers)

    async def start(self) -> None:
        """Start every available channel, isolating startup failures."""
        active: list[Notifier] = []
        for notifier in self._notifiers:
            try:
                await notifier.start()
            except Exception:
                log.exception("Error starting %s", type(notifier).__name__)
                try:
                    await notifier.close()
                except Exception:
                    log.exception("Error cleaning up %s", type(notifier).__name__)
            else:
                active.append(notifier)
        self._notifiers = active

    async def close(self) -> None:
        for notifier in self._notifiers:
            try:
                await notifier.close()
            except Exception:
                log.exception("Error closing %s", type(notifier).__name__)

    async def notify(self, event: Event) -> Acknowledgement | None:
        acks: list[Acknowledgement] = []
        for notifier in self._notifiers:
            try:
                ack = await notifier.notify(event)
            except Exception:
                log.exception(
                    "%s failed to deliver %s",
                    type(notifier).__name__,
                    type(event).__name__,
                )
                continue
            if ack is not None:
                acks.append(ack)
        if not acks:
            return None
        return acks[0] if len(acks) == 1 else _AnyAcknowledgement(acks)


__all__ = [
    "Acknowledgement",
    "BaseNotifier",
    "CompositeNotifier",
    "Notifier",
    "NullNotifier",
]
