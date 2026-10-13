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
    from .coefficients import run as coefficients
    from .hilbert_condition import run as hilbert_condition
    from .hilbert_sequential import run as hilbert_sequential
    from .trca_features import run as trca_features
    from .trca_sequential import run as trca_sequential
    from .statistics import run as statistics
    from .report import run as report

    values = (
        Stage("00_freeze", "Lock cohort, methods, endpoints, and the confirmatory family", (), freeze),
        Stage("01_coefficients", "Compute matched-window Hann, full-window Hann, and Hilbert coefficients",
              ("00_freeze",), coefficients),
        Stage("02_hilbert_condition", "Static single-look K=10 condition comparison (H1 vs H2 vs H3)",
              ("01_coefficients",), hilbert_condition),
        Stage("03_hilbert_sequential", "Secondary sequential condition-level Pocock trajectories (H1/H2/H3)",
              ("01_coefficients",), hilbert_sequential),
        Stage("04_trca_features", "Fit fold-specific grouped TRCA filters and project held-out streams",
              ("00_freeze",), trca_features),
        Stage("05_trca_sequential", "Score and calibrate TRCA+S2 trajectories against S2/S4",
              ("01_coefficients", "04_trca_features"), trca_sequential),
        Stage("06_statistics", "Apply the locked three-hypothesis confirmatory family and secondary evidence",
              ("02_hilbert_condition", "03_hilbert_sequential", "05_trca_sequential"), statistics),
        Stage("07_report", "Generate conclusions, tables, figures, and the export manifest",
              ("06_statistics",), report),
    )
    return {stage.name: stage for stage in values}
