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
    from .benchmark import run as benchmark
    from .robustness import run as robustness
    from .report import run as report
    values = (
        Stage("00_freeze", "Lock target, algorithms, units, and validation rules", (), freeze),
        Stage("01_condition_features", "Compute condition, regional, reliability, and surrogate features", ("00_freeze",), features),
        Stage("02_nested_benchmark", "Run matched-unit LOPO condition-level detector comparison", ("01_condition_features",), benchmark),
        Stage("03_robustness", "Evaluate surrogate-seed and equal-window robustness", ("02_nested_benchmark",), robustness),
        Stage("04_report", "Classify the 90-percent target and generate evidence report", ("03_robustness",), report),
    )
    return {stage.name: stage for stage in values}
