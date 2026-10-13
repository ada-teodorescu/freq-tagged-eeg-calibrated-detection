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
    from .estimators import run as estimators
    from .benchmarks import run as benchmarks
    from .double_holdout import run as double_holdout
    from .chronological import run as chronological
    from .external_robustness import run as external_robustness
    from .statistics import run as statistics
    from .report import run as report
    values = (
        Stage("00_freeze", "Freeze extension design, provenance, and optional ASSR protocol", (), freeze),
        Stage("01_estimators", "Compute mechanism coefficients and effective-kernel diagnostics", ("00_freeze",), estimators),
        Stage("02_benchmarks", "Run matched estimator, sequential, and simple-detector benchmarks", ("01_estimators",), benchmarks),
        Stage("03_double_holdout", "Run participant-and-condition double-holdout evaluation", ("02_benchmarks",), double_holdout),
        Stage("04_chronological", "Map sequential crossings into actual presentation order and time", ("02_benchmarks",), chronological),
        Stage("05_external_robustness", "Run 28-channel and deterministic raw-window-selection robustness", ("00_freeze",), external_robustness),
        Stage("06_statistics", "Apply paired inference, multiplicity, and non-inferiority rules", ("03_double_holdout", "04_chronological", "05_external_robustness"), statistics),
        Stage("07_report", "Generate paper-ready tables, figures, and manuscript text", ("06_statistics",), report),
    )
    return {stage.name: stage for stage in values}
