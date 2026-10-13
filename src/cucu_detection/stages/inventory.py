from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from ..detectors import compatible
from ..runtime import RunContext, atomic_write_json


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "00_inventory"
    context.start_stage(stage, 4)
    config = context.config
    required = [
        config.data_root / "Matlab variables" / "Data_fft_all.mat",
        config.data_root / "Pre-processed EEGLAB datasets",
        config.data_root / "Stimuli",
        config.project_root / "replication_config.toml",
    ]
    missing = [str(path) for path in required if not path.exists()]
    context.progress(stage, 1, 4, "required inputs")
    set_files = sorted((config.data_root / "Pre-processed EEGLAB datasets").glob("*.set"))
    fdt_files = sorted((config.data_root / "Pre-processed EEGLAB datasets").glob("*.fdt"))
    wav_files = sorted((config.data_root / "Stimuli").glob("*.wav"))
    if len(set_files) != 25: missing.append(f"expected 25 SET files, found {len(set_files)}")
    if len(fdt_files) != 25: missing.append(f"expected 25 FDT files, found {len(fdt_files)}")
    if len(wav_files) < 45: missing.append(f"expected at least 45 WAV files, found {len(wav_files)}")
    context.progress(stage, 2, 4, "branch files")
    compatibility = []
    for estimator in config.estimators:
        for detector in config.detectors:
            compatibility.append({
                "estimator": estimator, "detector": detector,
                "compatible": compatible(estimator.split("_")[0], detector),
                "stored_matlab": estimator == "F1" and detector != "D7",
                "public_set_fdt": compatible(estimator.split("_")[0], detector),
            })
    compatibility_path = context.table_dir / "method_compatibility.csv"
    pd.DataFrame(compatibility).to_csv(compatibility_path, index=False)
    context.progress(stage, 3, 4, "compatibility matrix")
    provenance = {
        "stored_matlab": {
            "allowed_estimators": ["F1"],
            "source": str(required[0]),
            "cross_source_merge_allowed": False,
        },
        "public_set_fdt": {
            "allowed_estimators": list(config.estimators),
            "set_files": len(set_files), "fdt_files": len(fdt_files),
            "cross_source_merge_allowed": False,
        },
        "missing_or_invalid": missing,
    }
    provenance_path = context.run_dir / "branch_provenance.json"
    atomic_write_json(provenance_path, provenance)
    context.progress(stage, 4, 4, "provenance contract")
    if missing: raise FileNotFoundError("Benchmark input validation failed:\n" + "\n".join(missing))
    outputs = [compatibility_path, provenance_path]
    context.finish_stage(stage, outputs, provenance)
    return outputs, provenance
