from __future__ import annotations

import asyncio

import pytest

from src.detection.results import DetectionResult
from src.errors import PrinterUnavailable
from src.events import (
    ImageUnavailable,
    MonitorError,
    MonitoringStarted,
    MonitoringStopped,
    SpaghettiDetected,
    StatusUpdate,
)
from src.monitor import FAILURE_NOTIFY_INTERVAL, MonitorLoop
from src.printer.base import PrintState
from tests.conftest import (
    FakeDetector,
    FakePrinter,
    RecordingAck,
    RecordingNotifier,
    run_briefly,
)


def build(settings, printer, detector, notifier) -> MonitorLoop:
    return MonitorLoop(printer, detector, notifier, settings)


class TestHappyPath:
    async def test_announces_start_and_stop(self, settings, frame):
        notifier = RecordingNotifier()
        loop = build(settings, FakePrinter(frame), FakeDetector(), notifier)
        await run_briefly(loop, 0.05)

        assert notifier.of_type(MonitoringStarted)
        assert notifier.of_type(MonitoringStopped)

    async def test_clean_frame_produces_a_status_update(self, settings, frame):
        notifier = RecordingNotifier()
        detector = FakeDetector(DetectionResult())
        loop = build(settings, FakePrinter(frame), detector, notifier)
        await run_briefly(loop, 0.05)

        statuses = notifier.of_type(StatusUpdate)
        assert statuses
        assert statuses[0].state is PrintState.PRINTING
        assert statuses[0].image is frame

    async def test_idle_printer_skips_inference(self, settings, frame):
        notifier = RecordingNotifier()
        detector = FakeDetector()
        printer = FakePrinter(frame, state=PrintState.STANDBY)
        await run_briefly(build(settings, printer, detector, notifier), 0.05)

        assert detector.calls == 0
        assert "Idle" in notifier.of_type(StatusUpdate)[0].detail


class TestDetection:
    async def test_pauses_and_waits_for_acknowledgement(
        self, settings, frame, detection_result
    ):
        ack = RecordingAck()
        notifier = RecordingNotifier(ack=ack)
        printer = FakePrinter(frame)
        loop = build(settings, printer, FakeDetector(detection_result), notifier)
        await run_briefly(loop, 0.05)

        events = notifier.of_type(SpaghettiDetected)
        assert events
        assert events[0].paused is True
        assert events[0].pause_requested is True
        assert printer.pause_calls >= 1
        assert ack.waited

    async def test_reports_a_failed_pause_honestly(
        self, settings, frame, detection_result
    ):
        notifier = RecordingNotifier(ack=RecordingAck())
        printer = FakePrinter(frame, pause_succeeds=False)
        loop = build(settings, printer, FakeDetector(detection_result), notifier)
        await run_briefly(loop, 0.05)

        event = notifier.of_type(SpaghettiDetected)[0]
        assert event.pause_requested is True
        assert event.paused is False

    async def test_respects_pause_on_spaghetti_disabled(
        self, settings, frame, detection_result
    ):
        settings.pause_on_spaghetti = False
        notifier = RecordingNotifier(ack=RecordingAck())
        printer = FakePrinter(frame)
        loop = build(settings, printer, FakeDetector(detection_result), notifier)
        await run_briefly(loop, 0.05)

        assert printer.pause_calls == 0
        assert notifier.of_type(SpaghettiDetected)[0].pause_requested is False

    async def test_annotated_image_is_attached(self, settings, frame, detection_result):
        notifier = RecordingNotifier(ack=RecordingAck())
        loop = build(
            settings, FakePrinter(frame), FakeDetector(detection_result), notifier
        )
        await run_briefly(loop, 0.05)

        event = notifier.of_type(SpaghettiDetected)[0]
        assert event.annotated_image is not None
        assert event.annotated_image.mode == "RGB"
        # The detector's frame must not have been drawn on in place.
        assert detection_result.image.mode == "L"

    async def test_continues_when_no_channel_can_acknowledge(
        self, settings, frame, detection_result
    ):
        notifier = RecordingNotifier(ack=None)
        loop = build(
            settings, FakePrinter(frame), FakeDetector(detection_result), notifier
        )
        await run_briefly(loop, 0.05)

        assert notifier.of_type(SpaghettiDetected)


class TestResilience:
    async def test_repeat_frame_failures_are_throttled(self, settings):
        """A printer that is simply switched off must not flood the channel."""
        notifier = RecordingNotifier()
        printer = FakePrinter(None)
        loop = build(settings, printer, FakeDetector(), notifier)
        await run_briefly(loop, 0.15)

        failures = notifier.of_type(ImageUnavailable)
        assert printer.snapshot_calls > 5, "the loop should have kept polling"
        assert failures[0].consecutive_failures == 1
        assert len(failures) < printer.snapshot_calls / 2
        assert all(
            f.consecutive_failures == 1
            or f.consecutive_failures % FAILURE_NOTIFY_INTERVAL == 0
            for f in failures
        )

    async def test_detector_errors_do_not_kill_the_loop(self, settings, frame):
        notifier = RecordingNotifier()
        detector = FakeDetector(error=PrinterUnavailable("boom"))
        loop = build(settings, FakePrinter(frame), detector, notifier)
        await run_briefly(loop, 0.1)

        assert notifier.of_type(MonitorError)
        assert detector.calls > 1, "the loop should have kept iterating"

    async def test_broken_notifier_does_not_kill_the_loop(self, settings, frame):
        notifier = RecordingNotifier(fail_on=(MonitorError, ImageUnavailable))
        loop = build(settings, FakePrinter(None), FakeDetector(), notifier)
        await run_briefly(loop, 0.1)  # must not raise

    async def test_cancellation_still_reports_a_stop(self, settings, frame):
        notifier = RecordingNotifier()
        loop = build(settings, FakePrinter(frame), FakeDetector(), notifier)

        task = asyncio.ensure_future(loop.run())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert notifier.of_type(MonitoringStopped)


class TestPacing:
    async def test_overrunning_iteration_does_not_sleep_negative(self, settings, frame):
        """`asyncio.sleep(target - elapsed)` went negative and span the loop."""
        settings.target_loop_time = 0.01
        loop = build(settings, FakePrinter(frame), FakeDetector(), RecordingNotifier())

        # 5s of work against a 0.01s target: the clamp must return promptly
        # rather than raising or busy-looping.
        await asyncio.wait_for(loop._sleep_remaining(5.0), timeout=1)

    async def test_stop_interrupts_the_sleep(self, settings, frame):
        settings.target_loop_time = 30
        loop = build(settings, FakePrinter(frame), FakeDetector(), RecordingNotifier())
        loop.request_stop()
        await asyncio.wait_for(loop._sleep_remaining(0.0), timeout=1)
