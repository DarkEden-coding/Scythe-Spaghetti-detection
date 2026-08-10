"""Discord notification channel.

Owns the ``discord.Client`` and its lifecycle. The monitor loop used to live
inside ``on_ready``, which fires again on every reconnect and therefore started
a second concurrent loop each time the network hiccupped. Here the client runs
as a background task and the caller awaits :meth:`DiscordNotifier.start` once.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import discord

from src.config import DiscordSettings
from src.errors import NotifierError
from src.events import (
    Event,
    ImageUnavailable,
    MonitorError,
    MonitoringStarted,
    MonitoringStopped,
    SpaghettiDetected,
    StatusUpdate,
)
from src.notify.base import BaseNotifier
from src.utils.formatting import format_duration
from src.utils.images import to_jpeg_bytes

log = logging.getLogger(__name__)

ACK_EMOJI = "👍"
STATUS_FILENAME = "current_view.jpg"
FAIL_FILENAME = "spaghetti.jpg"


class DiscordAcknowledgement:
    """Waits for a human to react to the failure message."""

    def __init__(self, client: discord.Client, message: discord.Message):
        self._client = client
        self._message = message

    def _check(self, reaction: discord.Reaction, user: discord.abc.User) -> bool:
        return reaction.message.id == self._message.id and user.id != getattr(
            self._client.user, "id", None
        )

    async def wait(self, timeout: float | None = None) -> bool:
        """Block until someone reacts.

        Uses the gateway event rather than re-fetching the message once a
        second, which is what the previous implementation did.
        """
        log.info("Waiting for acknowledgement on message %s", self._message.id)
        try:
            await self._client.wait_for(
                "reaction_add", check=self._check, timeout=timeout or None
            )
        except asyncio.TimeoutError:
            log.warning("Acknowledgement timed out after %ss; resuming.", timeout)
            return False
        log.info("Acknowledged; resuming detection.")
        return True


class DiscordNotifier(BaseNotifier):
    """Posts status and failure messages to a Discord channel."""

    def __init__(self, settings: DiscordSettings, client: discord.Client | None = None):
        self._settings = settings
        self._client = client or self._build_client()
        self._channel: discord.abc.Messageable | None = None
        self._runner: asyncio.Task | None = None
        self._status_message: discord.Message | None = None
        self._started = False

    @staticmethod
    def _build_client() -> discord.Client:
        intents = discord.Intents.default()
        intents.typing = False
        intents.presences = False
        # Reaction acknowledgement needs message content off but reactions on;
        # the default intents already include guild reactions.
        return discord.Client(intents=intents)

    # -- lifecycle --------------------------------------------------------- #

    async def start(self) -> None:
        if self._started:
            return

        self._runner = asyncio.create_task(
            self._client.start(self._settings.bot_token), name="discord-client"
        )
        ready = asyncio.create_task(self._client.wait_until_ready())
        done, _ = await asyncio.wait(
            {self._runner, ready}, return_when=asyncio.FIRST_COMPLETED
        )
        if self._runner in done:
            # The client stopped before becoming ready: surface the real cause.
            ready.cancel()
            exc = self._runner.exception()
            raise NotifierError(f"Discord login failed: {exc}") from exc

        channel = self._client.get_channel(self._settings.log_channel_id)
        if channel is None:
            try:
                channel = await self._client.fetch_channel(self._settings.log_channel_id)
            except discord.DiscordException as exc:
                raise NotifierError(
                    f"Could not access channel {self._settings.log_channel_id}. "
                    f"Check the id and that the bot has been invited: {exc}"
                ) from exc
        if not isinstance(channel, discord.abc.Messageable):
            raise NotifierError(
                f"Channel {self._settings.log_channel_id} is not a text channel."
            )

        self._channel = channel
        self._started = True
        log.info("Connected to Discord as %s", self._client.user)

    async def close(self) -> None:
        self._started = False
        if not self._client.is_closed():
            await self._client.close()
        if self._runner is not None:
            self._runner.cancel()
            # Shutdown must not raise: whatever the client task was doing when
            # it was cancelled, we are on our way out either way.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._runner
            self._runner = None

    @property
    def channel(self) -> discord.abc.Messageable:
        if self._channel is None:
            raise NotifierError("DiscordNotifier.start() has not completed.")
        return self._channel

    # -- dispatch ---------------------------------------------------------- #

    async def notify(self, event: Event) -> DiscordAcknowledgement | None:
        handlers = {
            MonitoringStarted: self._on_started,
            MonitoringStopped: self._on_stopped,
            StatusUpdate: self._on_status,
            SpaghettiDetected: self._on_spaghetti,
            ImageUnavailable: self._on_image_unavailable,
            MonitorError: self._on_error,
        }
        handler = handlers.get(type(event))
        if handler is None:
            log.debug("No Discord handler for %s", type(event).__name__)
            return None
        return await handler(event)

    # -- handlers ---------------------------------------------------------- #

    async def _on_started(self, event: MonitoringStarted) -> None:
        await self.channel.send(
            embed=discord.Embed(
                title="Monitoring started",
                description=(
                    f"Watching `{event.printer_url}` every "
                    f"{event.target_loop_time:g}s using `{event.model_name}`."
                ),
                color=discord.Color.green(),
            )
        )

    async def _on_stopped(self, event: MonitoringStopped) -> None:
        await self.channel.send(
            embed=discord.Embed(
                title="Monitoring stopped",
                description=(
                    f"{event.reason} (ran for {format_duration(event.uptime_seconds)})"
                ),
                color=discord.Color.dark_grey(),
            )
        )

    async def _on_status(self, event: StatusUpdate) -> None:
        mode = self._settings.status_update_mode
        if mode == "silent":
            return None

        embed = discord.Embed(
            title="Status",
            description=(
                f"State: **{event.state.value}**\n"
                f"Uptime: {format_duration(event.uptime_seconds)}"
                + (f"\n{event.detail}" if event.detail else "")
            ),
            color=discord.Color.green(),
        )

        def build_file() -> discord.File | None:
            if event.image is None:
                return None
            return discord.File(to_jpeg_bytes(event.image), filename=STATUS_FILENAME)

        # `edit` keeps one message current instead of posting ~2,880/day.
        if mode == "edit" and self._status_message is not None:
            try:
                attachments = [f] if (f := build_file()) else []
                await self._status_message.edit(embed=embed, attachments=attachments)
                return None
            except discord.DiscordException as exc:
                log.warning("Could not edit status message (%s); posting a new one.", exc)
                self._status_message = None

        file = build_file()
        message = await self.channel.send(embed=embed, file=file or discord.utils.MISSING)
        if mode == "edit":
            self._status_message = message
        return None

    async def _on_spaghetti(
        self, event: SpaghettiDetected
    ) -> DiscordAcknowledgement | None:
        if event.paused:
            description = "Print **paused automatically**. Please check the printer."
        elif event.pause_requested:
            description = (
                "Auto-pause was requested but **failed** — check the printer now."
            )
        else:
            description = (
                "Print **not** paused (auto-pause is disabled). Please check the printer."
            )

        embed = discord.Embed(
            title="Spaghetti detected!",
            description=f"{description}\n<@{self._settings.ping_user_id}>",
            color=discord.Color.red(),
        )
        embed.add_field(name="Detections", value=str(event.result.count))
        embed.add_field(name="Confidence", value=f"{event.result.max_confidence:.0%}")
        embed.set_footer(text=f"React with {ACK_EMOJI} to resume monitoring.")

        file = discord.utils.MISSING
        if event.annotated_image is not None:
            file = discord.File(
                to_jpeg_bytes(event.annotated_image), filename=FAIL_FILENAME
            )
            embed.set_image(url=f"attachment://{FAIL_FILENAME}")

        message = await self.channel.send(
            content=f"<@{self._settings.ping_user_id}>", embed=embed, file=file
        )
        try:
            await message.add_reaction(ACK_EMOJI)
        except discord.DiscordException as exc:
            log.warning("Could not add the acknowledgement reaction: %s", exc)

        # A failure supersedes the status message; start a fresh one next loop.
        self._status_message = None
        return DiscordAcknowledgement(self._client, message)

    async def _on_image_unavailable(self, event: ImageUnavailable) -> None:
        await self.channel.send(
            embed=discord.Embed(
                title="Camera unavailable",
                description=(
                    f"{event.reason}\nConsecutive failures: {event.consecutive_failures}"
                ),
                color=discord.Color.orange(),
            )
        )

    async def _on_error(self, event: MonitorError) -> None:
        await self.channel.send(
            embed=discord.Embed(
                title="Monitor error",
                description=(
                    f"{event.message}\n"
                    f"Consecutive failures: {event.consecutive_failures}\n"
                    "Monitoring will continue."
                ),
                color=discord.Color.orange(),
            )
        )
