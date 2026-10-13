from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json

from cucu_article.runtime import atomic_write_json, atomic_write_text
from ..runtime import RunContext


def _file_record(path: Path) -> dict:
    record = {"path": str(path), "exists": path.exists()}
    if path.exists():
        record.update({"bytes": path.stat().st_size, "sha256": sha256(path.read_bytes()).hexdigest()})
    return record


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage, config = "00_freeze", context.config
    context.start_stage(stage, 3)
    public = config.data_root / "Pre-processed EEGLAB datasets"
    set_paths = sorted(public.glob("*.set"))
    if len(set_paths) != 25: raise ValueError(f"expected 25 public SET files, found {len(set_paths)}")
    baseline_design = json.loads((config.baseline_confirmatory_run / "design_lock.json").read_text(encoding="utf-8"))
    participants = [str(value) for value in baseline_design["participants"]]
    required = [
        config.baseline_confirmatory_run / "confirmatory_conclusions.json",
        config.baseline_confirmatory_run / "checkpoints/02_hilbert_condition/condition_features.npz",
        config.baseline_confirmatory_run / "checkpoints/03_hilbert_sequential/trajectories.npz",
        config.baseline_external_run / "external_validation_conclusions.json",
        config.baseline_external_run / "tables/external_negative_scores.csv",
        config.raw_events,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing: raise FileNotFoundError("missing frozen inputs: " + ", ".join(missing))
    context.progress(stage, 1, 3, "baseline contracts")
    design = {
        "analysis_status": "post_hoc_robustness_extension",
        "participants": participants, "conditions": config.conditions,
        "methods": {
            "primary": ["H1_Hann4s", "H2_Hilbert4s", "M1_RectLS4s", "M2_FilteredLS4s"],
            "hilbert_sensitivity": {
                "bandwidth_hz": list(config.bandwidth_sensitivity_hz),
                "edge_crop_seconds": list(config.crop_sensitivity_seconds),
                "causal": "complex demodulation plus order-4 causal low-pass after measured 99% burn-in",
            },
            "simple_primary": "equal mean of 20 regional-by-harmonic 2K*R2 features",
            "simple_secondary": ["equal regional 4+8 Hz", "equal regional 4 Hz"],
        },
        "primary_inference": {
            "mechanism_contrasts": ["H2-H1", "H2-M1"], "correction": "Holm",
            "sequential_endpoints": ["detection_by_K10", "restricted_mean_stopping_K"],
            "simple_noninferiority_margin": config.simple_margin,
            "fpr_acceptance_high": config.fpr_acceptance_high,
        },
        "guardrails": [
            "existing confirmatory and external results are immutable inputs",
            "sensitivity analyses are explanatory and cannot replace locked endpoints",
            "external robustness never retunes a model or threshold using external outcomes",
        ],
    }
    design_path = context.run_dir / "design_lock.json"; atomic_write_json(design_path, design)
    context.progress(stage, 2, 3, "design lock")
    fingerprints = {"inputs": [_file_record(path) for path in required],
                    "public_set_fdt": [_file_record(path) for path in sorted(public.glob("*.set")) + sorted(public.glob("*.fdt"))]}
    fingerprint_path = context.run_dir / "input_fingerprints.json"; atomic_write_json(fingerprint_path, fingerprints)
    protocol = """# Optional independent positive ASSR protocol

Dataset: OpenNeuro ds006780 (https://openneuro.org/datasets/ds006780). This phase is estimator-level validation, not validation of the complete Cucu detector.

- Primary cohort: typically developing participants; require at least 35 eligible recordings.
- Primary endpoint: standard 40-Hz trials; 27 Hz is secondary and oddballs are excluded.
- Filter continuous EEG before epoching; use non-overlapping chronological blocks of ten trials.
- Compare matched Hann, rectangular least squares, and Hilbert fundamental coefficients.
- Hilbert lock for 0.5-s stimuli: order 4, +/-4 Hz, retained [0.05, 0.45] s.
- Use participant-held-out null calibration and participant-cluster inference.
- Generalization requires a positive 95% CI for Hilbert minus Hann at 40 Hz and held-out FPR <=5.5%.
- Execute only after stages 01-06 pass and matched-support results do not reveal an implementation artifact.
"""
    protocol_path = context.run_dir / "assr_positive_protocol.md"; atomic_write_text(protocol_path, protocol)
    context.progress(stage, 3, 3, "ASSR protocol")
    outputs = [design_path, fingerprint_path, protocol_path]
    context.finish_stage(stage, outputs, {"participants": len(participants), "assr_execution": "deferred"})
    return outputs, {"participants": len(participants)}
