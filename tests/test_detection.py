from __future__ import annotations

import pytest
from PIL import Image

from src.config import DetectionSettings
from src.detection.annotate import draw_boxes
from src.detection.detector import SpaghettiDetector
from src.detection.results import DetectionBox, DetectionResult


class FakeBox:
    """Mimics an Ultralytics box, whose values are all zero-dim tensors."""

    class _Scalar:
        def __init__(self, value):
            self._value = value

        def item(self):
            return self._value

    def __init__(self, xyxy, conf, cls):
        self.xyxy = [[self._Scalar(v) for v in xyxy]]
        self.conf = [self._Scalar(conf)]
        self.cls = [self._Scalar(cls)]


class FakeYoloResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    def __init__(self, boxes):
        self._boxes = boxes
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return [FakeYoloResult(self._boxes)]


@pytest.fixture
def settings(tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"x")
    return DetectionSettings(model_path=weights, use_onnx=False, min_confidence=0.6)


@pytest.fixture
def frame():
    return Image.new("RGB", (1280, 720), (60, 60, 60))


class TestDetectionResult:
    def test_falsy_when_empty(self):
        assert not DetectionResult()

    def test_truthy_with_boxes(self):
        result = DetectionResult(boxes=(DetectionBox(0, 0, 10, 10, 0.9, 1),))
        assert result
        assert result.count == 1
        assert result.max_confidence == 0.9

    def test_box_geometry(self):
        box = DetectionBox(10, 20, 40, 60, 0.5, 1)
        assert (box.width, box.height, box.area) == (30, 40, 1200)
        assert box.as_tuple() == (10, 20, 40, 60)


class TestDetector:
    def test_construction_loads_nothing(self, settings):
        assert SpaghettiDetector(settings).is_loaded is False

    def test_preprocess_matches_training(self, settings, frame):
        processed = SpaghettiDetector(settings).preprocess(frame)
        assert processed.size == (640, 640)
        assert processed.mode == "L"

    def test_filters_by_class(self, settings, frame):
        model = FakeModel(
            [
                FakeBox((10, 10, 50, 50), 0.95, 1),  # spaghetti
                FakeBox((60, 60, 90, 90), 0.99, 0),  # some other class
            ]
        )
        result = SpaghettiDetector(settings, model=model).detect(frame)

        assert result.count == 1
        assert result.boxes[0].class_id == 1
        assert len(result.considered) == 2, "everything seen is kept for tuning"

    def test_filters_by_confidence(self, settings, frame):
        model = FakeModel([FakeBox((10, 10, 50, 50), 0.4, 1)])
        assert not SpaghettiDetector(settings, model=model).detect(frame)

    def test_returns_original_frame_with_boxes_mapped_to_it(self, settings, frame):
        model = FakeModel([FakeBox((10, 20, 50, 60), 0.9, 1)])
        result = SpaghettiDetector(settings, model=model).detect(frame)

        assert result.image is not None
        assert result.image.size == (1280, 720)
        assert result.boxes[0].as_tuple() == (20, 22, 100, 68)

    def test_writes_nothing_to_disk(self, settings, frame, tmp_path, monkeypatch):
        """The old detect() wrote fail_img.jpg on every call, including misses."""
        workdir = tmp_path / "cwd"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        model = FakeModel([FakeBox((10, 10, 50, 50), 0.9, 1)])
        SpaghettiDetector(settings, model=model).detect(frame)
        assert list(workdir.iterdir()) == []

    def test_device_selection(self, settings, frame):
        settings.use_cuda = True
        model = FakeModel([])
        SpaghettiDetector(settings, model=model).detect(frame)
        assert model.calls[0]["device"] == 0

        settings.use_cuda = False
        model = FakeModel([])
        SpaghettiDetector(settings, model=model).detect(frame)
        assert model.calls[0]["device"] == "cpu"


class TestAnnotate:
    def test_returns_a_copy(self, frame):
        boxes = [DetectionBox(10, 20, 100, 140, 0.91, 1)]
        annotated = draw_boxes(frame, boxes)
        assert annotated is not frame
        assert annotated.mode == "RGB"
        assert annotated.size == frame.size

    def test_grayscale_input_survives(self):
        """Grayscale frames used to go through an RGB->BGR conversion."""
        gray = Image.new("L", (640, 640), 40)
        annotated = draw_boxes(gray, [DetectionBox(5, 5, 60, 60, 0.8, 1)])
        assert annotated.mode == "RGB"

    def test_draws_something(self):
        blank = Image.new("RGB", (200, 200), (0, 0, 0))
        annotated = draw_boxes(blank, [DetectionBox(20, 20, 180, 180, 0.77, 1)])
        assert annotated.getcolors(maxcolors=100000) != blank.getcolors(maxcolors=100000)

    def test_label_stays_in_frame_for_edge_boxes(self):
        blank = Image.new("RGB", (200, 200), (0, 0, 0))
        draw_boxes(blank, [DetectionBox(0, 0, 50, 50, 0.9, 1)])  # must not raise

    def test_no_boxes_is_a_noop_copy(self, frame):
        assert draw_boxes(frame, []).size == frame.size
