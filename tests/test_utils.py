from __future__ import annotations

import pytest
from PIL import Image

from src.notify.base import CompositeNotifier, NullNotifier
from src.utils.formatting import format_duration
from src.utils.images import to_jpeg_bytes


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0 seconds"),
            (1, "1 second"),
            (59, "59 seconds"),
            (60, "1 minute, 0 seconds"),
            (3600, "1 hour, 0 minutes, 0 seconds"),
            (86400, "1 day, 0 hours, 0 minutes, 0 seconds"),
            (90061, "1 day, 1 hour, 1 minute, 1 second"),
        ],
    )
    def test_formats(self, seconds, expected):
        assert format_duration(seconds) == expected

    def test_negative_is_clamped(self):
        assert format_duration(-5) == "0 seconds"

    def test_drops_leading_zero_units(self):
        assert format_duration(75) == "1 minute, 15 seconds"


class TestJpegBytes:
    def test_rewound_and_readable(self):
        buffer = to_jpeg_bytes(Image.new("RGB", (10, 10), (1, 2, 3)))
        assert buffer.tell() == 0
        assert Image.open(buffer).size == (10, 10)

    def test_grayscale_passes_through(self):
        buffer = to_jpeg_bytes(Image.new("L", (10, 10), 128))
        assert Image.open(buffer).mode == "L"

    def test_rgba_is_converted(self):
        """JPEG cannot store an alpha channel."""
        buffer = to_jpeg_bytes(Image.new("RGBA", (10, 10), (1, 2, 3, 4)))
        assert Image.open(buffer).mode == "RGB"


class BrokenNotifier(NullNotifier):
    async def notify(self, event):
        raise RuntimeError("down")


class CountingNotifier(NullNotifier):
    def __init__(self):
        self.count = 0

    async def notify(self, event):
        self.count += 1
        return None


class StartBrokenNotifier(NullNotifier):
    def __init__(self):
        self.closed = False

    async def start(self):
        raise RuntimeError("login failed")

    async def close(self):
        self.closed = True


class TestCompositeNotifier:
    async def test_one_broken_channel_does_not_stop_the_others_at_startup(self):
        broken = StartBrokenNotifier()
        composite = CompositeNotifier([broken, CountingNotifier()])

        await composite.start()

        assert len(composite) == 1
        assert broken.closed is True

    async def test_one_broken_channel_does_not_stop_the_others(self):
        healthy = CountingNotifier()
        composite = CompositeNotifier([BrokenNotifier(), healthy])
        await composite.notify(object())
        assert healthy.count == 1

    async def test_fans_out(self):
        a, b = CountingNotifier(), CountingNotifier()
        await CompositeNotifier([a, b]).notify(object())
        assert (a.count, b.count) == (1, 1)

    async def test_returns_none_without_acknowledgements(self):
        assert await CompositeNotifier([CountingNotifier()]).notify(object()) is None
