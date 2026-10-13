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
    from .inventory import run as inventory
    from .coefficients import run as coefficients
    from .trajectories import run as trajectories
    from .static_validation import run as static_validation
    from .sequential_validation import run as sequential_validation
    from .statistics import run as statistics
    from .report import run as report
    values = (
        Stage("00_freeze", "Freeze the independent ASSR protocol and provenance", (), freeze),
        Stage("01_inventory", "Validate the BIDS cohort and construct chronological blocks", ("00_freeze",), inventory),
        Stage("02_coefficients", "Preprocess continuous EEG and extract three matched estimators", ("01_inventory",), coefficients),
        Stage("03_null_trajectories", "Build positive and shared phase-rotation null trajectories", ("02_coefficients",), trajectories),
        Stage("04_static_validation", "Run participant-held-out K=10 calibration", ("03_null_trajectories",), static_validation),
        Stage("05_sequential_validation", "Run participant-held-out sequential calibration", ("03_null_trajectories",), sequential_validation),
        Stage("06_sensitivity_statistics", "Apply inference, pass rules, and artifact sensitivity", ("04_static_validation", "05_sequential_validation"), statistics),
        Stage("07_report", "Generate paper-ready reports, tables, and figures", ("06_sensitivity_statistics",), report),
    )
    return {stage.name: stage for stage in values}
