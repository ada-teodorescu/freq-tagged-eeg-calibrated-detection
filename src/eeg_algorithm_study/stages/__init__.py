from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from ..runtime import RunContext

StageFunction=Callable[[RunContext,bool],tuple[list,dict]]
@dataclass(frozen=True)
class Stage:
    name:str; description:str; dependencies:tuple[str,...]; function:StageFunction

def stage_registry()->dict[str,Stage]:
    from .diagnosis import run as diagnosis
    from .features import run as features
    from .benchmark import run as benchmark
    from .report import run as report
    values=(
        Stage("00_diagnosis","Lock baseline failures and repaired comparison",(),diagnosis),
        Stage("01_surrogate_features","Compute same-frequency phase-randomized feature nulls",("00_diagnosis",),features),
        Stage("02_repaired_benchmark","Compare equal, subset-selected, and monotone learned fusion",("01_surrogate_features",),benchmark),
        Stage("03_report","Analyze repaired algorithms and participant-adaptive calibration",("02_repaired_benchmark",),report),
    )
    return {stage.name:stage for stage in values}
