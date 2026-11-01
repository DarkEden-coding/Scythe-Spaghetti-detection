"""Cost estimates for decision-model detection.

The per-request costs were measured in the print-failure evaluation, sending a
part crop enlarged to 1240 px (the default image size) with the same context
and two questions this project asks. Once the detector has made real requests, their reported cost is used
instead.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Shown in the dashboard's model picker, cheapest first within each tier.
KNOWN_MODELS = (
    ("~liquid/d1-ultra-latest", "Liquid d1-ultra (recommended)", 0.0000454),
    ("typesafe/jev-1.14", "Jev 1.14", 0.0000839),
    ("liquid/d1.1", "Liquid d1.1 (cheaper, weaker in the dark)", 0.0000216),
)

COST_PER_REQUEST = {model: cost for model, _label, cost in KNOWN_MODELS}
#: d1-ultra resolves to a dated id; treat its aliases as the same price.
COST_PER_REQUEST["liquid/d1-ultra"] = COST_PER_REQUEST["~liquid/d1-ultra-latest"]


@dataclass(frozen=True)
class CostEstimate:
    detections_per_hour: float
    cost_per_request: float | None
    #: ``measured`` from this run's API responses, ``table`` from the
    #: evaluation, ``free`` for self-hosted models, or ``unknown``.
    source: str

    @property
    def cost_per_hour(self) -> float | None:
        if self.cost_per_request is None:
            return None
        return self.detections_per_hour * self.cost_per_request


def detections_per_hour(capture_interval: float, detection_interval: float) -> float:
    """Detections per print-hour; detection can't run faster than capture."""
    return 3600.0 / max(capture_interval, detection_interval)


def estimate(
    provider: str,
    model: str,
    capture_interval: float,
    detection_interval: float,
    measured_cost: float | None = None,
) -> CostEstimate:
    rate = detections_per_hour(capture_interval, detection_interval)
    if provider != "openrouter":
        return CostEstimate(rate, 0.0, "free")
    if measured_cost is not None:
        return CostEstimate(rate, measured_cost, "measured")
    if model in COST_PER_REQUEST:
        return CostEstimate(rate, COST_PER_REQUEST[model], "table")
    return CostEstimate(rate, None, "unknown")
