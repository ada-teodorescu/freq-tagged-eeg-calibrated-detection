from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eeg_advanced_study.algorithms import calibrate_pocock_boundaries, first_crossings
from eeg_detection.evaluation import empirical_threshold
from ..algorithms import (METHODS, holm_adjust, paired_sign_flip, participant_bootstrap,
                          phase_rotations, score_trajectories, simultaneous_participant_band)
from ..runtime import RunContext, atomic_save_npz, atomic_write_json
from .coefficients import coefficient_path


def primary_gate(ci95_low: float, hilbert_fpr: float, participants: int, config) -> bool:
    return bool(participants >= config.minimum_participants and ci95_low > 0 and
                hilbert_fpr <= config.fpr_acceptance_high)


def _metric_vector(frame: pd.DataFrame, method: str, frequency: float, value: str,
                   participants: tuple[str, ...]) -> np.ndarray:
    selected = frame[(frame["method"] == method) & (frame["frequency_hz"] == frequency)]
    series = selected.set_index("participant")[value].reindex(participants)
    if series.isna().any():
        raise ValueError(f"incomplete {method} {frequency:g}-Hz metric {value}")
    return series.to_numpy(float)


def _comparison(first: np.ndarray, second: np.ndarray, config, salt: int) -> dict:
    difference = np.asarray(first, float) - np.asarray(second, float)
    return {**participant_bootstrap(difference, config.bootstrap_iterations, config.seed + salt),
            "p_value": paired_sign_flip(difference, config.sign_flip_iterations, config.seed + 10_000 + salt)}


def _clean_path(context: RunContext, participant: str) -> Path:
    return context.checkpoint_dir / "06_sensitivity_statistics" / f"clean_{participant}.npz"


def _clean_trajectory(context: RunContext, participant: str, participant_index: int, force: bool) -> Path:
    config = context.config
    destination = _clean_path(context, participant)
    if context.checkpoint_valid(destination) and not force:
        return destination
    with np.load(coefficient_path(context, participant), allow_pickle=False) as archive:
        coefficients, artifacts = archive["coefficients"], archive["peak_to_peak_uv"]
    max_blocks = max(config.standards_per_frequency) // config.trials_per_block
    positive = np.full((len(METHODS), len(config.frequencies_hz), max_blocks, len(config.k_values)), np.nan, np.float32)
    null = np.full((len(METHODS), len(config.frequencies_hz), max_blocks,
                    config.surrogates_per_block, len(config.k_values)), np.nan, np.float32)
    block_counts, clean_counts = [], []
    for f_index, frequency in enumerate(config.frequencies_hz):
        limit = config.standard_limit(frequency)
        keep = artifacts[f_index, :limit] <= config.artifact_peak_to_peak_uv
        clean_counts.append(int(keep.sum()))
        blocks = min(int(keep.sum()) // config.trials_per_block, max_blocks)
        block_counts.append(blocks)
        if blocks == 0:
            continue
        rotations = np.stack([phase_rotations(config.surrogates_per_block, config.trials_per_block,
                                              np.random.SeedSequence([config.seed, participant_index, f_index, block, 1]))
                              for block in range(blocks)])
        for m_index in range(len(METHODS)):
            values = coefficients[m_index, f_index, :limit][keep][:blocks * config.trials_per_block]
            values = values.reshape(blocks, config.trials_per_block, 64)
            positive[m_index, f_index, :blocks] = score_trajectories(values, config.k_values)
            null[m_index, f_index, :blocks] = score_trajectories(values, config.k_values, rotations)
    atomic_save_npz(destination, positive=positive, null=null, block_counts=np.asarray(block_counts),
                    clean_counts=np.asarray(clean_counts), participant=np.asarray(participant))
    context.mark_checkpoint(destination, {"participant": participant, "artifact_threshold_uv": config.artifact_peak_to_peak_uv,
                                          "clean_counts": clean_counts, "block_counts": block_counts,
                                          "shape_positive": list(positive.shape), "shape_null": list(null.shape),
                                          "input_fingerprint": coefficient_path(context, participant).with_suffix(".npz.meta.json").read_text(encoding="utf-8")})
    return destination


def _artifact_sensitivity(context: RunContext, participants: tuple[str, ...], force: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = context.config
    bundles = []
    for index, participant in enumerate(participants):
        path = _clean_trajectory(context, participant, index, force)
        with np.load(path, allow_pickle=False) as archive:
            bundles.append({name: archive[name] for name in ("positive", "null", "block_counts", "clean_counts")})
    static_rows, sequential_rows = [], []
    for f_index, frequency in enumerate(config.frequencies_hz):
        eligible_indices = [index for index, bundle in enumerate(bundles)
                            if bundle["clean_counts"][f_index] >= config.minimum_clean_trials]
        eligible = tuple(participants[index] for index in eligible_indices)
        for m_index, method in enumerate(METHODS):
            for local_index, held_index in enumerate(eligible_indices):
                held = bundles[held_index]
                held_blocks = int(held["block_counts"][f_index])
                training_indices = [value for value in eligible_indices if value != held_index]
                train_null = np.concatenate([bundles[index]["null"][m_index, f_index, :int(bundles[index]["block_counts"][f_index])]
                                             for index in training_indices], axis=0)
                threshold = empirical_threshold(train_null[..., -1].reshape(-1), config.target_fpr)
                held_positive = held["positive"][m_index, f_index, :held_blocks]
                held_null = held["null"][m_index, f_index, :held_blocks]
                static_rows.append({"participant": participants[held_index], "method": method, "frequency_hz": frequency,
                                    "clean_trials": int(held["clean_counts"][f_index]), "blocks": held_blocks,
                                    "detection": float(np.mean(held_positive[:, -1] > threshold)),
                                    "heldout_fpr": float(np.mean(held_null[..., -1] > threshold)), "threshold": threshold})
                fitted = calibrate_pocock_boundaries(train_null.reshape(-1, len(config.k_values)), config.target_fpr)
                first = first_crossings(held_positive, fitted.boundaries, config.k_values)
                null_first = first_crossings(held_null.reshape(-1, len(config.k_values)), fitted.boundaries, config.k_values)
                sequential_rows.append({"participant": participants[held_index], "method": method, "frequency_hz": frequency,
                                        "clean_trials": int(held["clean_counts"][f_index]), "blocks": held_blocks,
                                        "detection": float(np.mean(first <= config.k_values[-1])),
                                        "rmst_k": float(np.mean(first)),
                                        "familywise_fpr": float(np.mean(null_first <= config.k_values[-1]))})
    static = pd.DataFrame(static_rows); sequential = pd.DataFrame(sequential_rows)
    frequency = config.primary_frequency_hz
    eligible = tuple(sorted(static.loc[static["frequency_hz"] == frequency, "participant"].unique()))
    if len(eligible) >= 2:
        hilbert = _metric_vector(static, "hilbert", frequency, "detection", eligible)
        hann = _metric_vector(static, "hann", frequency, "detection", eligible)
        h_rmst = _metric_vector(sequential, "hann", frequency, "rmst_k", eligible)
        i_rmst = _metric_vector(sequential, "hilbert", frequency, "rmst_k", eligible)
        result = {"eligible_participants": len(eligible),
                  "hilbert_minus_hann": participant_bootstrap(hilbert - hann, config.bootstrap_iterations, config.seed + 900),
                  "hann_minus_hilbert_rmst": participant_bootstrap(h_rmst - i_rmst, config.bootstrap_iterations, config.seed + 901),
                  "hilbert_heldout_fpr": float(static[(static["method"] == "hilbert") &
                                                       (static["frequency_hz"] == frequency)]["heldout_fpr"].mean())}
    else:
        result = {"eligible_participants": len(eligible), "status": "insufficient_clean_participants"}
    return static, sequential, result


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage, config = "06_sensitivity_statistics", context.config
    participants = tuple(pd.read_csv(context.table_dir / "recording_inventory.csv")["participant"].astype(str))
    context.start_stage(stage, 7 + len(participants))
    static = pd.read_csv(context.table_dir / "static_participant_metrics.csv")
    sequential = pd.read_csv(context.table_dir / "sequential_participant_metrics.csv")
    primary_frequency = config.primary_frequency_hz
    hann = _metric_vector(static, "hann", primary_frequency, "detection", participants)
    hilbert = _metric_vector(static, "hilbert", primary_frequency, "detection", participants)
    primary = _comparison(hilbert, hann, config, 1)
    hilbert_fpr = float(_metric_vector(static, "hilbert", primary_frequency, "heldout_fpr", participants).mean())
    primary.update({"contrast": "hilbert - hann", "frequency_hz": primary_frequency, "k": config.k_values[-1],
                    "eligible_participants": len(participants), "hilbert_heldout_fpr": hilbert_fpr})
    primary["passed"] = primary_gate(primary["ci95_low"], hilbert_fpr, len(participants), config)
    context.progress(stage, 1, 7 + len(participants), "primary gate")

    comparison_rows = [{"family": "primary", **primary, "holm_p_value": primary["p_value"]}]
    families = {
        "secondary_40hz": [("rectangular_ls - hann", "rectangular_ls", "hann"),
                            ("hilbert - rectangular_ls", "hilbert", "rectangular_ls")],
        "secondary_27hz": [("hilbert - hann", "hilbert", "hann"),
                            ("rectangular_ls - hann", "rectangular_ls", "hann"),
                            ("hilbert - rectangular_ls", "hilbert", "rectangular_ls")],
    }
    for family_index, (family, contrasts) in enumerate(families.items(), 1):
        frequency = 40.0 if family == "secondary_40hz" else 27.0
        local = []
        for contrast_index, (label, first, second) in enumerate(contrasts):
            result = _comparison(_metric_vector(static, first, frequency, "detection", participants),
                                 _metric_vector(static, second, frequency, "detection", participants),
                                 config, 100 * family_index + contrast_index)
            local.append({"family": family, "contrast": label, "frequency_hz": frequency,
                          "k": config.k_values[-1], **result})
        adjusted = holm_adjust(np.asarray([row["p_value"] for row in local]))
        for row, value in zip(local, adjusted, strict=True):
            row["holm_p_value"] = float(value)
        comparison_rows += local
    context.progress(stage, 2, 7 + len(participants), "secondary estimator families")

    gain_by_participant = []
    score_frame = pd.read_csv(context.table_dir / "sequential_scores.csv")
    for participant in participants:
        h1 = score_frame[(score_frame["participant"] == participant) & (score_frame["method"] == "hann") &
                         (score_frame["frequency_hz"] == primary_frequency)]
        h2 = score_frame[(score_frame["participant"] == participant) & (score_frame["method"] == "hilbert") &
                         (score_frame["frequency_hz"] == primary_frequency)]
        gain_by_participant.append([float(h2[f"detected_by_k{k}"].mean() - h1[f"detected_by_k{k}"].mean())
                                    for k in config.k_values])
    gains = np.asarray(gain_by_participant)
    low, high = simultaneous_participant_band(gains, config.bootstrap_iterations, config.seed + 500)
    gain_curve = pd.DataFrame({"k": config.k_values, "gain": gains.mean(axis=0),
                               "simultaneous_ci95_low": low, "simultaneous_ci95_high": high})
    curve_path = context.table_dir / "sequential_gain_curve.csv"
    gain_curve.to_csv(curve_path, index=False)
    hann_rmst = _metric_vector(sequential, "hann", primary_frequency, "rmst_k", participants)
    hilbert_rmst = _metric_vector(sequential, "hilbert", primary_frequency, "rmst_k", participants)
    rmst = participant_bootstrap(hann_rmst - hilbert_rmst, config.bootstrap_iterations, config.seed + 501)
    rmst.update({"contrast": "hann - hilbert restricted mean stopping K", "frequency_hz": primary_frequency})
    context.progress(stage, 3, 7 + len(participants), "sequential inference")

    for index, participant in enumerate(participants, 1):
        _clean_trajectory(context, participant, index - 1, force)
        context.progress(stage, 3 + index, 7 + len(participants), f"artifact {participant}")
    artifact_static, artifact_sequential, artifact_result = _artifact_sensitivity(context, participants, False)
    artifact_static_path = context.table_dir / "artifact_sensitivity_static.csv"
    artifact_sequential_path = context.table_dir / "artifact_sensitivity_sequential.csv"
    artifact_static.to_csv(artifact_static_path, index=False)
    artifact_sequential.to_csv(artifact_sequential_path, index=False)
    context.progress(stage, 4 + len(participants), 7 + len(participants), "artifact sensitivity")

    comparisons_path = context.table_dir / "inferential_comparisons.csv"
    pd.DataFrame(comparison_rows).to_csv(comparisons_path, index=False)
    conclusions = {"analysis_id": context.analysis_id, "status": "independent_positive_estimator_validation",
                   "primary": primary, "secondary_static": comparison_rows[1:], "sequential": rmst,
                   "artifact_sensitivity": artifact_result,
                   "generalization_supported": primary["passed"],
                   "interpretation_guardrail": "This validates estimator behavior on ASSR data, not syllable classification or a clinical detector."}
    conclusion_path = context.run_dir / "assr_validation_conclusions.json"
    atomic_write_json(conclusion_path, conclusions)
    context.progress(stage, 7 + len(participants), 7 + len(participants), "conclusions")
    outputs = [curve_path, artifact_static_path, artifact_sequential_path, comparisons_path, conclusion_path]
    context.finish_stage(stage, outputs, {"primary_passed": primary["passed"], "eligible_participants": len(participants)})
    return outputs, conclusions
