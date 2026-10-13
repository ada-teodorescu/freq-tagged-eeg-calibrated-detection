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
    from .preprocess import run as preprocess
    from .units import run as units
    from .features import run as features
    from .deployment import run as deployment
    from .scores import run as scores
    from .statistics import run as statistics
    from .report import run as report
    stages = (
        Stage("00_freeze", "Freeze dataset, baseline, preprocessing, and pass rules", (), freeze),
        Stage("01_inventory", "Inventory BIDS recordings, channels, tasks, and fingerprints", ("00_freeze",), inventory),
        Stage("02_preprocess", "Validate fixed preprocessing and channel-interpolation eligibility", ("01_inventory",), preprocess),
        Stage("03_negative_units", "Construct deterministic condition-shaped negative windows", ("02_preprocess",), units),
        Stage("04_hilbert_features", "Preprocess EEG and extract locked regional Hilbert coefficients", ("03_negative_units",), features),
        Stage("05_deployment_model", "Fit frozen Cucu-only static and sequential deployment models", ("00_freeze",), deployment),
        Stage("06_external_scores", "Apply frozen models and boundaries to external negative units", ("04_hilbert_features", "05_deployment_model"), scores),
        Stage("07_statistics", "Calculate clustered uncertainty and the submission gate", ("06_external_scores",), statistics),
        Stage("08_report", "Generate immutable report, tables, figures, and manifest inputs", ("07_statistics",), report),
    )
    return {stage.name: stage for stage in stages}
