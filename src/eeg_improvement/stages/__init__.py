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
    from .stored_d8 import run as stored_d8
    from .public_coefficients import run as public_coefficients
    from .public_focused import run as public_focused
    from .report import run as report
    values = (
        Stage("00_inventory", "Lock the completed baseline and validate focused inputs", (), inventory),
        Stage("01_stored_d8", "Compare all-harmonic, 4+8-Hz and train-weighted D8", ("00_inventory",), stored_d8),
        Stage("02_public_coefficients", "Cache corrected Hann, demodulated Hilbert and Thomson coefficients", ("00_inventory",), public_coefficients),
        Stage("03_public_focused", "Neighbor-calibrated S0/S1 focused public-EEG screen", ("02_public_coefficients",), public_focused),
        Stage("04_report", "Summarize corrected follow-up without changing the baseline claims", ("01_stored_d8", "03_public_focused"), report),
    )
    return {value.name: value for value in values}
