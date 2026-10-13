from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from eeg_core.runtime import atomic_write_json
from eeg_detection.evaluation import holm_adjust
from ..algorithms import (crossed_bootstrap_draws, mcnemar_exact, paired_sign_flip,
                          simultaneous_crossed_band)
from ..runtime import RunContext


def _matrix(frame: pd.DataFrame, value: str, participants: list[str]) -> np.ndarray:
    pivot = frame.pivot(index="participant", columns="condition", values=value).reindex(participants)
    if pivot.shape != (len(participants), 15) or pivot.isna().any().any():
        raise ValueError(f"incomplete participant-condition matrix for {value}")
    return pivot.to_numpy(float)


def _contrast(first: np.ndarray, second: np.ndarray, config, seed_salt: int) -> dict:
    gains = np.asarray(first, float) - np.asarray(second, float)
    draws = crossed_bootstrap_draws(gains, config.bootstrap_iterations, config.seed + seed_salt)
    participant_gain = gains.mean(axis=1)
    return {"estimate": float(gains.mean()), "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "p_value": paired_sign_flip(participant_gain, config.sign_flip_iterations, config.seed + 10000 + seed_salt)}


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    del force
    stage, config = "06_statistics", context.config
    participants = json.loads((context.run_dir / "design_lock.json").read_text(encoding="utf-8"))["participants"]
    context.start_stage(stage, 6); comparison_rows = []

    static = pd.read_csv(context.table_dir / "static_scores.csv")
    learned = static[static["detector"] == "learned"]
    matrices = {method: _matrix(learned[learned["method"] == method], "detected", participants)
                for method in ("H1_Hann4s", "H2_Hilbert4s", "M1_RectLS4s", "M2_FilteredLS4s")}
    mechanism = []
    for salt, comparator in enumerate(("H1_Hann4s", "M1_RectLS4s")):
        result = _contrast(matrices["H2_Hilbert4s"], matrices[comparator], config, salt)
        mechanism.append({"family": "mechanism", "contrast": f"H2_Hilbert4s - {comparator}", **result})
    adjusted = holm_adjust(np.asarray([row["p_value"] for row in mechanism]))
    for row, value in zip(mechanism, adjusted, strict=True): row["holm_p_value"] = float(value)
    comparison_rows += mechanism; context.progress(stage, 1, 6, "mechanism family")

    # Post hoc exploratory control, outside the locked two-contrast mechanism family
    # (design_lock.json primary_inference.mechanism_contrasts == ["H2-H1", "H2-M1"] only):
    # isolates explicit narrow-band filtering from the analytic-signal/demodulation step by
    # comparing Hilbert against a filter-matched sinusoidal-regression control. Reported
    # pointwise, without Holm pooling into the confirmatory-style mechanism family above.
    filter_matched = _contrast(matrices["H2_Hilbert4s"], matrices["M2_FilteredLS4s"], config, 2)
    filter_matched_mcnemar = mcnemar_exact(matrices["H2_Hilbert4s"].reshape(-1), matrices["M2_FilteredLS4s"].reshape(-1))
    filter_matched_row = {"family": "mechanism_exploratory", "contrast": "H2_Hilbert4s - M2_FilteredLS4s", **filter_matched}
    comparison_rows.append(filter_matched_row)

    sequential = pd.read_csv(context.table_dir / "sequential_scores.csv")
    h1 = sequential[(sequential["method"] == "H1_Hann4s") & (sequential["detector"] == "learned")]
    h2 = sequential[(sequential["method"] == "H2_Hilbert4s") & (sequential["detector"] == "learned")]
    gains = np.stack([_matrix(h2, f"detected_by_k{k}", participants) - _matrix(h1, f"detected_by_k{k}", participants)
                      for k in config.k_values], axis=-1)
    band_low, band_high = simultaneous_crossed_band(gains, config.bootstrap_iterations, config.seed + 100)
    curve_rows = [{"look_k": k, "gain": float(gains[..., look].mean()),
                   "simultaneous_ci95_low": float(band_low[look]), "simultaneous_ci95_high": float(band_high[look])}
                  for look, k in enumerate(config.k_values)]
    curve_path = context.table_dir / "sequential_gain_curve.csv"; pd.DataFrame(curve_rows).to_csv(curve_path, index=False)
    k10 = _contrast(gains[..., -1], np.zeros_like(gains[..., -1]), config, 101)
    rmst_gain = _matrix(h1, "first_detection_k", participants) - _matrix(h2, "first_detection_k", participants)
    rmst = _contrast(rmst_gain, np.zeros_like(rmst_gain), config, 102)
    seq_rows = [{"family": "sequential", "contrast": "H2-H1 detection by K10", **k10},
                {"family": "sequential", "contrast": "H1-H2 restricted mean stopping K", **rmst}]
    adjusted = holm_adjust(np.asarray([row["p_value"] for row in seq_rows]))
    for row, value in zip(seq_rows, adjusted, strict=True): row["holm_p_value"] = float(value)
    comparison_rows += seq_rows
    mcnemar = mcnemar_exact(_matrix(h2, "detected", participants).reshape(-1),
                            _matrix(h1, "detected", participants).reshape(-1))
    context.progress(stage, 2, 6, "sequential family")

    simple = static[(static["method"] == "H2_Hilbert4s") & (static["detector"] == "simple_equal_4h")]
    learned_h2 = learned[learned["method"] == "H2_Hilbert4s"]
    learned_minus_simple = _matrix(learned_h2, "detected", participants) - _matrix(simple, "detected", participants)
    simple_draws = crossed_bootstrap_draws(learned_minus_simple, config.bootstrap_iterations, config.seed + 200)
    simple_upper = float(np.quantile(simple_draws, config.one_sided_confidence))
    simple_fpr = float(simple["heldout_fpr"].mean())
    simple_noninferior = bool(simple_upper <= config.simple_margin and simple_fpr <= config.fpr_acceptance_high)
    simple_result = {"estimate_learned_minus_simple": float(learned_minus_simple.mean()),
                     "one_sided_95_upper": simple_upper, "margin": config.simple_margin,
                     "heldout_fpr": simple_fpr, "noninferior": simple_noninferior}
    context.progress(stage, 3, 6, "simple non-inferiority")

    double_static = pd.read_csv(context.table_dir / "double_holdout_static.csv")
    dh_h1 = double_static[(double_static["method"] == "H1_Hann4s") & (double_static["detector"] == "learned")]
    dh_h2 = double_static[(double_static["method"] == "H2_Hilbert4s") & (double_static["detector"] == "learned")]
    dh_gain = _contrast(_matrix(dh_h2, "static_detected", participants), _matrix(dh_h1, "static_detected", participants), config, 300)
    dh_fpr = float(dh_h2["static_fpr"].mean())
    dh_supported = bool(dh_gain["ci95_low"] > 0 and dh_fpr <= config.fpr_acceptance_high)
    dh_simple = double_static[(double_static["method"] == "H2_Hilbert4s") & (double_static["detector"] == "simple_equal_4h")]
    dh_simple_gap = _matrix(dh_h2, "static_detected", participants) - _matrix(dh_simple, "static_detected", participants)
    dh_simple_upper = float(np.quantile(crossed_bootstrap_draws(dh_simple_gap, config.bootstrap_iterations,
                                                               config.seed + 301), config.one_sided_confidence))
    double_result = {"hilbert_minus_hann": dh_gain, "hilbert_fpr": dh_fpr, "generalization_supported": dh_supported,
                     "learned_minus_simple_upper": dh_simple_upper,
                     "simple_noninferior": bool(dh_simple_upper <= config.simple_margin and dh_simple["static_fpr"].mean() <= config.fpr_acceptance_high)}
    context.progress(stage, 4, 6, "double holdout")

    chronological = pd.read_csv(context.table_dir / "chronological_stopping.csv")
    chron_h1 = chronological[(chronological["method"] == "H1_Hann4s") & (chronological["detector"] == "learned")]
    chron_h2 = chronological[(chronological["method"] == "H2_Hilbert4s") & (chronological["detector"] == "learned")]
    chronological_result = {
        "presentations_saved": _contrast(_matrix(chron_h1, "global_presentation_index", participants),
                                          _matrix(chron_h2, "global_presentation_index", participants), config, 400),
        "seconds_saved": _contrast(_matrix(chron_h1, "elapsed_from_first_target_seconds", participants),
                                    _matrix(chron_h2, "elapsed_from_first_target_seconds", participants), config, 401),
    }
    context.progress(stage, 5, 6, "chronological stopping")

    external_summary = pd.read_csv(context.table_dir / "external_robustness_summary.csv").to_dict(orient="records")
    baseline_conclusions = json.loads((config.baseline_external_run / "external_validation_conclusions.json").read_text(encoding="utf-8"))
    conclusions = {"analysis_id": context.analysis_id, "status": "post_hoc_robustness_extension",
                   "mechanism": mechanism, "sequential": seq_rows, "mcnemar_k10": mcnemar,
                   "mechanism_exploratory_filter_matched": {**filter_matched_row, "mcnemar": filter_matched_mcnemar},
                   "simple_detector": simple_result, "double_holdout": double_result,
                   "chronological": chronological_result,
                   "external_locked_gate_unchanged": baseline_conclusions.get("submission_gate_passed"),
                   "external_robustness": external_summary,
                   "interpretation": "Sensitivity and extension results do not replace the immutable confirmatory family."}
    comparison_path = context.table_dir / "inferential_comparisons.csv"; pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    conclusion_path = context.run_dir / "validation_extension_conclusions.json"; atomic_write_json(conclusion_path, conclusions)
    context.progress(stage, 6, 6, "conclusions")
    outputs = [curve_path, comparison_path, conclusion_path]
    context.finish_stage(stage, outputs, {"simple_noninferior": simple_noninferior, "double_holdout_supported": dh_supported})
    return outputs, conclusions
