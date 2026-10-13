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
    from .stored_fft import run as stored_fft
    from .source_audit import run as source_audit
    from .published_results import run as published_results
    from .hilbert import run as hilbert
    from .filter_null import run as filter_null
    from .frequency_correction import run as frequency_correction
    from .report import run as report

    stages = (
        Stage("00_inventory", "Validate and fingerprint all local inputs", (), inventory),
        Stage("01_stored_fft", "Replicate stored MATLAB FFT to ITPC products", ("00_inventory",), stored_fft),
        Stage("02_source_audit", "Rebuild FFT ITPC from public SET/FDT data", ("00_inventory", "01_stored_fft"), source_audit),
        Stage("03_published_results", "Reproduce PCA and core published outcomes", ("01_stored_fft",), published_results),
        Stage("04_hilbert", "Run Hilbert-method robustness analysis", ("02_source_audit", "03_published_results"), hilbert),
        Stage("05_filter_null", "Measure narrow-filter null coherence", ("02_source_audit",), filter_null),
        Stage("06_frequency_correction", "Estimate and correct true stream frequencies", ("02_source_audit", "03_published_results"), frequency_correction),
        Stage("07_report", "Assemble claim ledger, tables, and report", (
            "01_stored_fft", "02_source_audit", "03_published_results", "04_hilbert",
            "05_filter_null", "06_frequency_correction",
        ), report),
    )
    return {stage.name: stage for stage in stages}
