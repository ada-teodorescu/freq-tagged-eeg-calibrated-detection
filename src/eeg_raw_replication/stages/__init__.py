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
    from .event_audit import run as event_audit
    from .preprocess import run as preprocess
    from .epoch_comparison import run as epoch_comparison
    from .spectral import run as spectral
    from .product_comparison import run as product_comparison
    from .paper_results import run as paper_results
    from .report import run as report

    stages = (
        Stage("00_inventory", "Validate and fingerprint raw and released inputs", (), inventory),
        Stage("01_events", "Reconstruct and audit target-stream events", ("00_inventory",), event_audit),
        Stage("02_preprocess", "Filter, rereference, epoch, and apply declared ICA branches", ("01_events",), preprocess),
        Stage("03_epoch_comparison", "Compare raw-derived epochs with same-ID SET/FDT data", ("02_preprocess",), epoch_comparison),
        Stage("04_spectral", "Compute compact Hann FFT and squared ITPC products", ("02_preprocess",), spectral),
        Stage("05_product_comparison", "Compare ITPC with released products and audit ordering", ("04_spectral",), product_comparison),
        Stage("06_paper_results", "Recompute PCA, planned tests, acoustic links, and figures", ("04_spectral",), paper_results),
        Stage("07_report", "Assemble QC pages, claim ledger, and evidence report", (
            "03_epoch_comparison", "05_product_comparison", "06_paper_results",
        ), report),
    )
    return {stage.name: stage for stage in stages}
