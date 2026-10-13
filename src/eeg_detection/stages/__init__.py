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
    from .inventory import run as inventory
    from .stored_coefficients import run as stored_coefficients
    from .null_validation import run as null_validation
    from .detector_screen import run as detector_screen
    from .public_estimators import run as public_estimators
    from .spatial_screen import run as spatial_screen
    from .outcomes import run as outcomes
    from .report import run as report

    values = (
        Stage("00_inventory", "Validate benchmark inputs and method compatibility", (), inventory),
        Stage("01_stored_coefficients", "Cache target and neighboring stored FFT coefficients", ("00_inventory",), stored_coefficients),
        Stage("02_null_validation", "Validate detector calibration with phase-randomized surrogates", ("01_stored_coefficients",), null_validation),
        Stage("03_detector_screen", "LOPO stored-branch detector and trial-count screen", ("01_stored_coefficients", "02_null_validation"), detector_screen),
        Stage("04_public_estimators", "Cache public time-domain estimator coefficients", ("00_inventory",), public_estimators),
        Stage("05_spatial_screen", "Cross-validated public-branch spatial-filter screen", ("04_public_estimators",), spatial_screen),
        Stage("06_outcomes", "Compute trial efficiency, uncertainty, ranking, runtime and memory", ("03_detector_screen", "05_spatial_screen"), outcomes),
        Stage("07_report", "Assemble figures, provenance report and claim ledger", ("06_outcomes",), report),
    )
    return {value.name: value for value in values}
