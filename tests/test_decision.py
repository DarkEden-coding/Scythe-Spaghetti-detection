from __future__ import annotations

import io

import pytest
import requests
from PIL import Image

from src.config import DecisionSettings, DetectionSettings
from src.detection import decision
from src.detection.decision import DecisionDetector
from src.errors import ConfigError, DetectorError


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def answers(any_failure, layer_shift):
    return FakeResponse(
        payload={
            "answers": {
                "any_failure": {"noul": any_failure},
                "layer_shift": {"noul": layer_shift},
            }
        }
    )


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(decision.time, "sleep", lambda _seconds: None)


@pytest.fixture
def settings():
    return DetectionSettings(
        backend="decision",
        decision=DecisionSettings(api_key="sk-test", threshold=0.6),
    )


@pytest.fixture
def frame():
    return Image.new("RGB", (1024, 576), (60, 60, 60))


class TestDetect:
    def test_flags_when_any_question_reaches_threshold(self, settings, frame):
        session = FakeSession(answers(0.2, 0.75))
        result = DecisionDetector(settings, session).detect(frame)

        assert result
        assert result.max_confidence == 0.75
        assert result.boxes[0].as_tuple() == (0, 0, 1024, 576)
        assert result.label == "Layer shift"
        assert result.scores == (("layer_shift", 0.75), ("any_failure", 0.2))

    def test_clear_below_threshold_keeps_score(self, settings, frame):
        session = FakeSession(answers(0.3, 0.1))
        result = DecisionDetector(settings, session).detect(frame)

        assert not result
        assert result.considered[0].confidence == 0.3

    def test_sends_cropped_image_within_size_limits(self, settings, frame):
        settings.decision.crop = "0.25,0.25,0.75,0.5"
        session = FakeSession(answers(0.9, 0.9))
        detector = DecisionDetector(settings, session)

        result = detector.detect(frame)

        assert result.boxes[0].as_tuple() == (256, 144, 768, 288)
        _, request = session.requests[0]
        assert request["json"]["model"] == "~liquid/d1-ultra-latest"
        assert request["headers"]["Authorization"] == "Bearer sk-test"
        encoded = detector.encode(frame, (256, 144, 768, 288))
        with Image.open(io.BytesIO(encoded)) as sent:
            assert max(sent.size) == 1240 and min(sent.size) >= 256

    def test_ollama_request_format(self, settings, frame):
        settings.decision.provider = "ollama"
        settings.decision.api_key = ""
        session = FakeSession(answers(0.1, 0.1))
        DecisionDetector(settings, session).detect(frame)

        url, request = session.requests[0]
        assert url == "http://localhost:11434/v1/systemone"
        assert isinstance(request["json"]["state"], str)
        assert len(request["json"]["images"]) == 1
        assert "Authorization" not in request["headers"]

    def test_tracks_reported_cost(self, settings, frame):
        priced = answers(0.1, 0.1)
        priced._payload["usage"] = {"cost": 0.00004}
        detector = DecisionDetector(settings, FakeSession(priced))
        assert detector.average_cost is None
        detector.detect(frame)
        assert detector.average_cost == pytest.approx(0.00004)


class TestErrors:
    def test_retries_rate_limits(self, settings, frame):
        session = FakeSession(
            FakeResponse(429, text="slow down"),
            requests.ConnectionError("reset"),
            answers(0.9, 0.1),
        )
        assert DecisionDetector(settings, session).detect(frame)
        assert len(session.requests) == 3

    def test_gives_up_after_repeated_failures(self, settings, frame):
        session = FakeSession(*[FakeResponse(503, text="down")] * 3)
        with pytest.raises(DetectorError, match="after 3 attempts"):
            DecisionDetector(settings, session).detect(frame)

    def test_bad_key_fails_without_retrying(self, settings, frame):
        session = FakeSession(FakeResponse(401, text="invalid key"))
        with pytest.raises(DetectorError, match="HTTP 401"):
            DecisionDetector(settings, session).detect(frame)
        assert len(session.requests) == 1

    def test_malformed_answer(self, settings, frame):
        session = FakeSession(FakeResponse(payload={"answers": {}}))
        with pytest.raises(DetectorError, match="Unexpected"):
            DecisionDetector(settings, session).detect(frame)


class TestSettings:
    def test_openrouter_backend_skips_model_file_check(self, settings, tmp_path):
        settings.model_path = tmp_path / "missing.pt"
        settings.validate()

    def test_requires_api_key(self, settings):
        settings.decision.api_key = ""
        with pytest.raises(ConfigError, match="api_key"):
            settings.validate()

    @pytest.mark.parametrize("crop", ["0.5,0.2,0.4,0.9", "1,2,3", "a,b,c,d"])
    def test_rejects_bad_crop(self, settings, crop):
        settings.decision.crop = crop
        with pytest.raises(ConfigError, match="crop"):
            settings.validate()

    def test_self_hosted_needs_no_key(self, settings):
        settings.decision.provider = "ollama"
        settings.decision.api_key = ""
        settings.validate()

    def test_rejects_unknown_backend(self, settings):
        settings.backend = "magic"
        with pytest.raises(ConfigError, match="backend"):
            settings.validate()
