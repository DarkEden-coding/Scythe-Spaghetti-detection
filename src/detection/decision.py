"""Decision-model failure detector.

Decision models (Liquid d1, Jev, ...) answer yes/no questions about an image
with a probability instead of drawing boxes. This detector asks a few failure
questions about each frame and reports a failure when any answer reaches the
configured threshold. It returns the same :class:`DetectionResult` as the YOLO
detector, with one box over the region that was inspected.

Two request formats are supported: OpenRouter's Decisions API, and the
``/v1/systemone`` endpoint that Ollama serves for self-hosted decision models.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from io import BytesIO
from time import perf_counter

import requests
from PIL import Image

from src.config import DetectionSettings
from src.detection.results import DetectionBox, DetectionResult
from src.errors import DetectorError

log = logging.getLogger(__name__)

CONTEXT = "Camera image of a 3D printer during or after a print."

#: The prompts that scored best in the print-failure evaluation: the tuned
#: any-failure list from the public-dataset search, and the layer-shift
#: question that separated shifted from clean frames on this printer.
QUESTIONS = {
    "any_failure": {
        "type": "noul",
        "instructions": (
            "The print in this image shows at least one printing failure: spaghetti "
            "(loose tangled filament), stringing, under-extrusion (gaps or missing "
            "lines), warping (corners lifting off the bed), layer shifting, cracking "
            "between layers, or a part that has come loose or been knocked over. "
            "Ignore things that are normal on a working printer: filament residue or "
            "blobs stuck to the nozzle or hotend, purge lines, skirts, brims, "
            "supports, and the bare print bed."
        ),
    },
    "layer_shift": {
        "type": "noul",
        "instructions": (
            "The printed part in this image has a layer shift: some upper layers are "
            "offset sideways from the layers below, leaving a visible step, ledge or "
            "misaligned band on the side of the part."
        ),
    },
}

#: Image models reject images with a side under 256 px.
MIN_SIDE = 280
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
ATTEMPTS = 3


class DecisionDetector:
    """Classifies camera frames with a decision model."""

    def __init__(
        self, settings: DetectionSettings, session: requests.Session | None = None
    ):
        self._settings = settings
        self._session = session or requests.Session()
        self._cost_lock = threading.Lock()
        self._cost_total = 0.0
        self._cost_count = 0

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return self._settings.decision.model

    @property
    def average_cost(self) -> float | None:
        """Mean reported cost per request so far, if the provider reports one."""
        with self._cost_lock:
            return self._cost_total / self._cost_count if self._cost_count else None

    def load(self) -> None:
        """Nothing to load; the model runs on the provider's side."""

    # -- inference --------------------------------------------------------- #

    def region(self, image: Image.Image) -> tuple[int, int, int, int]:
        """Pixel box of the frame that is sent to the model."""
        box = self._settings.decision.crop_box()
        if box is None:
            return 0, 0, image.width, image.height
        left, top, right, bottom = box
        return (
            round(left * image.width),
            round(top * image.height),
            round(right * image.width),
            round(bottom * image.height),
        )

    def encode(self, image: Image.Image, region: tuple[int, int, int, int]) -> bytes:
        """Crop and resize the frame within the model's image size limits."""
        view = image.crop(region).convert("RGB")
        scale = max(
            self._settings.decision.image_max_side / max(view.size),
            MIN_SIDE / min(view.size),
        )
        view = view.resize(
            (round(view.width * scale), round(view.height * scale)), Image.LANCZOS
        )
        buffer = BytesIO()
        view.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()

    def detect(self, image: Image.Image) -> DetectionResult:
        """Ask the decision model about one frame.

        Blocking network I/O — callers on an event loop should dispatch this to
        a worker thread.
        """
        started = perf_counter()
        region = self.region(image)
        scores = self._ask(base64.b64encode(self.encode(image, region)).decode())

        score = max(scores.values())
        box = DetectionBox(
            *region,
            confidence=round(score, 4),
            class_id=self._settings.spaghetti_class_id,
        )
        flagged = score >= self._settings.decision.threshold
        log.debug(
            "Decision scores: %s",
            ", ".join(f"{name}={value:.2f}" for name, value in scores.items()),
        )
        return DetectionResult(
            boxes=(box,) if flagged else (),
            image=image.copy(),
            duration_seconds=perf_counter() - started,
            considered=(box,),
        )

    def _request(self, jpeg_b64: str) -> tuple[dict, dict[str, str]]:
        """Body and headers in the configured provider's format."""
        settings = self._settings.decision
        if settings.provider == "ollama":
            body = {
                "model": settings.model,
                "state": CONTEXT,
                "images": [jpeg_b64],
                "questions": QUESTIONS,
            }
        else:
            image = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"},
            }
            body = {
                "model": settings.model,
                "state": [CONTEXT, image],
                "questions": QUESTIONS,
            }
        headers = {"X-Title": "Scythe spaghetti detection"}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        return body, headers

    def _ask(self, jpeg_b64: str) -> dict[str, float]:
        """POST one decision request and return P(yes) per question."""
        settings = self._settings.decision
        body, headers = self._request(jpeg_b64)

        error = "no attempts made"
        for attempt in range(ATTEMPTS):
            if attempt:
                time.sleep(2**attempt)
            try:
                response = self._session.post(
                    settings.endpoint,
                    json=body,
                    headers=headers,
                    timeout=settings.request_timeout,
                )
            except requests.RequestException as exc:
                error = f"{type(exc).__name__}: {exc}"
                continue
            if response.status_code in RETRY_STATUSES:
                error = f"HTTP {response.status_code}: {response.text[:200]}"
                continue
            if response.status_code != 200:
                raise DetectorError(
                    f"The decision model rejected the request (HTTP "
                    f"{response.status_code}): {response.text[:300]}"
                )
            return self._parse(response)
        raise DetectorError(f"Decision request failed after {ATTEMPTS} attempts: {error}")

    def _parse(self, response: requests.Response) -> dict[str, float]:
        try:
            data = response.json()
            scores = {name: float(data["answers"][name]["noul"]) for name in QUESTIONS}
        except (ValueError, KeyError, TypeError) as exc:
            raise DetectorError(
                f"Unexpected decision-model response: {response.text[:300]}"
            ) from exc
        cost = (data.get("usage") or {}).get("cost")
        if isinstance(cost, (int, float)):
            with self._cost_lock:
                self._cost_total += cost
                self._cost_count += 1
        return scores
