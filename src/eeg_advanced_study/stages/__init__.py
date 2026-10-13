from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from ..runtime import RunContext

StageFunction = Callable[[RunContext, bool], tuple[list, dict]]


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    dependencies: tuple[str, ...]
    function: StageFunction


def stage_registry() -> dict[str, Stage]:
    from .freeze import run as freeze
    from .features import run as features
    from .stream import run as stream
    from .calibration import run as calibration
    from .condition import run as condition
    from .negatives import run as negatives
    from .statistics import run as statistics
    from .report import run as report
    values = (
        Stage("00_freeze", "Lock methods, endpoints, nulls, and validation", (), freeze),
        Stage("01_features", "Cache exact target and off-target coefficients", ("00_freeze",), features),
        Stage("02_stream", "Run nested stream-level detector benchmark", ("01_features",), stream),
        Stage("03_calibration", "Calibrate sequential and shrinkage boundaries", ("02_stream",), calibration),
        Stage("04_condition", "Run matched-unit condition regional benchmark", ("01_features",), condition),
        Stage("05_negatives", "Evaluate off-target and optional external controls", ("03_calibration",), negatives),
        Stage("06_statistics", "Classify locked primary and secondary outcomes", ("03_calibration", "04_condition", "05_negatives"), statistics),
        Stage("07_report", "Generate advanced detector evidence report", ("06_statistics",), report),
    )
    return {stage.name: stage for stage in values}
