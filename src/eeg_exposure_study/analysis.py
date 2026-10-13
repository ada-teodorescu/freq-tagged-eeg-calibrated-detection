"""Five-trial features, participant-held-out fitting, and paired summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_accuracy_study.features import region_indices
from eeg_advanced_study.algorithms import phase_trajectories, temporal_shift_multipliers
from eeg_algorithm_study.algorithms import fit_monotone_logistic
from eeg_core.runtime import atomic_write_text
from eeg_confirmatory_study.config import ConfirmatoryConfig
from eeg_detection.evaluation import crossed_bootstrap_gain, empirical_threshold

from .runtime import Config, Run


METHODS = ("Hann", "Hilbert")
FEATURE_KEYS = ("hann_matched", "hilbert")
HALVES = ("early", "late")
K = 5


def half_features(
    coefficients: np.ndarray,
    regions: tuple[np.ndarray, ...],
    source: ConfirmatoryConfig,
    participant_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return half x condition x [surrogate x] region x harmonic features.

    Each condition averages three independently scored streams. The same seeded
    five-trial rotations are used for both methods and halves; channels share
    each rotation and harmonic angles preserve their integer relationship.
    """
    positive, negative, global_r2 = [], [], []
    for half in range(2):
        selected = coefficients[:, half * K:(half + 1) * K]
        global_values, regional = phase_trajectories(selected, (K,), regions)
        null = []
        for stream in range(source.streams):
            rotations = np.concatenate([
                temporal_shift_multipliers(
                    K, (1, 2, 3, 4), source.surrogates_per_seed,
                    np.random.SeedSequence([seed, participant_index, stream, 50915]),
                ) for seed in source.surrogate_seeds
            ])
            _, values = phase_trajectories(selected[stream:stream + 1], (K,), regions, rotations)
            null.append(values[0, :, 0])
        positive.append(2 * K * regional[:, 0].reshape(source.conditions, 3, len(regions), 4).mean(axis=1))
        negative.append(2 * K * np.stack(null).reshape(
            source.conditions, 3, source.surrogates, len(regions), 4).mean(axis=1))
        global_r2.append(global_values[:, 0].reshape(source.conditions, 3, 4).mean(axis=1))
    return np.stack(positive), np.stack(negative), np.stack(global_r2)


def features(run: Run) -> None:
    config = run.config
    regions = region_indices(config.channel_labels, config.source.regions)
    for index, participant in enumerate(config.participants):
        run.progress("features", index, len(config.participants), participant)
        destination = run.checkpoint("features", participant)
        if not run.valid(destination):
            arrays = {}
            with np.load(config.coefficient_path(participant), allow_pickle=False) as archive:
                if str(archive["participant"].item()) != participant:
                    raise ValueError(f"Participant mismatch: {participant}")
                if tuple(archive["channel_labels"].tolist()) != config.channel_labels:
                    raise ValueError(f"Channel order mismatch: {participant}")
                if tuple(archive["harmonics_hz"].tolist()) != config.source.harmonics_hz:
                    raise ValueError(f"Harmonic mismatch: {participant}")
                for method, key in zip(METHODS, FEATURE_KEYS, strict=True):
                    coefficients = archive[key]
                    expected = (config.source.streams, 10, config.source.channels, 4)
                    if coefficients.shape != expected or not np.isfinite(coefficients).all():
                        raise ValueError(f"Invalid coefficients: {participant}, {method}")
                    if np.any(np.abs(coefficients) == 0):
                        raise ValueError(f"Zero coefficients prevent valid PPC: {participant}, {method}")
                    pos, null, r2 = half_features(coefficients, regions, config.source, index)
                    arrays.update({f"{method}_positive": pos, f"{method}_null": null,
                                   f"{method}_r2": r2})
            run.save(destination, **arrays)
        run.progress("features", index + 1, len(config.participants), participant)
    run.progress("features", len(config.participants), len(config.participants), "all participants", "completed")


def load_bundle(run: Run, method: str, key: str) -> np.ndarray:
    values = []
    for participant in run.config.participants:
        with np.load(run.checkpoint("features", participant), allow_pickle=False) as archive:
            values.append(archive[f"{method}_{key}"])
    return np.stack(values)


def fit_fold(positive: np.ndarray, null: np.ndarray, held: int, config: Config) -> dict:
    """Exclude both halves of held participant from ALL fitting/calibration.

    Input dimensions: participant x half x condition x [surrogate x] feature.
    A single model and threshold are shared across held-out early/late data.
    """
    training = np.arange(len(positive)) != held
    train_positive = positive[training].reshape(-1, positive.shape[-1])
    train_null = null[training].reshape(-1, null.shape[-1])
    values = np.concatenate((train_positive, train_null))
    labels = np.concatenate((np.ones(len(train_positive)), np.zeros(len(train_null))))
    model = fit_monotone_logistic(values, labels, config.source.locked_c,
                                  config.maximum_iterations, config.tolerance)
    if not model.converged:
        raise RuntimeError(f"Optimizer did not converge for held participant index {held}; no checkpoint saved")
    threshold = empirical_threshold(model.decision_function(train_null), config.source.target_fpr)
    positive_scores = model.decision_function(positive[held])
    null_scores = model.decision_function(null[held])
    if not all(np.isfinite(a).all() for a in (positive_scores, null_scores, threshold)):
        raise ValueError("Nonfinite model scores or threshold")
    return {"positive_scores": positive_scores, "null_scores": null_scores,
            "threshold": np.asarray(threshold), "coefficients": model.coefficients,
            "intercept": np.asarray(model.intercept), "mean": model.standardizer.mean,
            "scale": model.standardizer.scale, "converged": np.asarray(model.converged),
            "training_indices": np.flatnonzero(training)}


def folds(run: Run) -> None:
    total = len(METHODS) * len(run.config.participants)
    completed = 0
    for method in METHODS:
        positive = load_bundle(run, method, "positive")
        null = load_bundle(run, method, "null")
        positive = positive.reshape(*positive.shape[:3], -1)
        null = null.reshape(*null.shape[:4], -1)
        for held, participant in enumerate(run.config.participants):
            run.progress("folds", completed, total, f"{method}/{participant}")
            destination = run.checkpoint("folds", f"{method}_{participant}")
            if not run.valid(destination):
                run.save(destination, **fit_fold(positive, null, held, run.config))
            completed += 1
            run.progress("folds", completed, total, f"{method}/{participant}")
    run.progress("folds", completed, total, "all held-out fits", "completed")


def comparison_rows(matrices: dict[str, np.ndarray], run: Run) -> list[dict]:
    """Paired crossed bootstrap; participants and conditions remain aligned."""
    rows = []
    for index, (name, differences) in enumerate(matrices.items()):
        low, high = crossed_bootstrap_gain(differences, run.config.bootstrap_iterations,
                                           run.config.seed + index)
        rows.append({"comparison": name, "estimate": float(differences.mean()),
                     "ci95_low": low, "ci95_high": high,
                     "inference": "exploratory pointwise crossed-bootstrap interval"})
    return rows


def report(run: Run) -> None:
    run.progress("report", 0, 1, "paired summaries")
    scores, metrics, membership, phase_rows = [], [], [], []
    detected = np.empty((2, len(run.config.participants), 2, run.config.source.conditions))
    for method_index, method in enumerate(METHODS):
        r2 = load_bundle(run, method, "r2")
        for index, participant in enumerate(run.config.participants):
            with np.load(run.checkpoint("folds", f"{method}_{participant}"), allow_pickle=False) as fold:
                threshold = float(fold["threshold"])
                decisions = fold["positive_scores"] > threshold
                false_positives = fold["null_scores"] > threshold
                detected[method_index, index] = decisions
                for half, name in enumerate(HALVES):
                    metrics.append({"participant": participant, "method": method, "half": name,
                                    "detection_rate": float(decisions[half].mean()),
                                    "surrogate_fpr": float(false_positives[half].mean()),
                                    "threshold": threshold})
                    for condition in range(run.config.source.conditions):
                        scores.append({"participant": participant, "condition": condition + 1,
                                       "method": method, "half": name,
                                       "score": float(fold["positive_scores"][half, condition]),
                                       "threshold": threshold, "detected": bool(decisions[half, condition]),
                                       "surrogate_fpr": float(false_positives[half, condition].mean())})
                        for harmonic, frequency in enumerate(run.config.source.harmonics_hz):
                            value = float(r2[index, half, condition, harmonic])
                            phase_rows.append({"participant": participant, "condition": condition + 1,
                                               "method": method, "half": name, "frequency_hz": frequency,
                                               "mean_channel_itpc_squared": value,
                                               "mean_channel_ppc": (K * value - 1) / (K - 1)})
    for participant in run.config.participants:
        for stream in range(run.config.source.streams):
            for repetition in range(10):
                membership.append({"participant": participant, "condition": stream // 3 + 1,
                                   "stream": stream + 1, "stream_within_condition": stream % 3 + 1,
                                   "repetition_within_stream": repetition + 1,
                                   "half": HALVES[repetition // K]})
    hann, hilbert = detected
    contrasts = {
        "Hilbert_minus_Hann_early": hilbert[:, 0] - hann[:, 0],
        "Hilbert_minus_Hann_late": hilbert[:, 1] - hann[:, 1],
        "interaction_late_gain_minus_early_gain": (hilbert[:, 1] - hann[:, 1]) - (hilbert[:, 0] - hann[:, 0]),
        "Hann_late_minus_early": hann[:, 1] - hann[:, 0],
        "Hilbert_late_minus_early": hilbert[:, 1] - hilbert[:, 0],
    }
    comparisons = pd.DataFrame(comparison_rows(contrasts, run))
    metric_frame = pd.DataFrame(metrics)
    summary = metric_frame.groupby(["method", "half"], sort=False)[["detection_rate", "surrogate_fpr"]].mean().reset_index()
    phase_contrasts = {}
    for method in METHODS:
        r2 = load_bundle(run, method, "r2")
        for harmonic, frequency in enumerate(run.config.source.harmonics_hz[:2]):
            phase_contrasts[f"{method}_{frequency:g}Hz_PPC_late_minus_early"] = (
                K / (K - 1) * (r2[:, 1, :, harmonic] - r2[:, 0, :, harmonic]))
    tables = {"condition_scores": pd.DataFrame(scores), "participant_metrics": metric_frame,
              "summary": summary, "detection_comparisons": comparisons,
              "phase_measurements": pd.DataFrame(phase_rows),
              "phase_comparisons": pd.DataFrame(comparison_rows(phase_contrasts, run)),
              "trial_membership": pd.DataFrame(membership)}
    for name, frame in tables.items():
        atomic_write_text(run.path / "tables" / f"{name}.csv", frame.to_csv(index=False))
    lines = ["# Early/late exposure robustness (exploratory)", "",
             f"Analysis ID: `{run.identifier}`", "",
             "Each half uses five repetitions per stream and three streams per condition.",
             "Each method uses a pooled-half training model and threshold, excluding both halves of the tested participant.", "",
             "## Detection and calibration", ""]
    for row in summary.itertuples(index=False):
        lines.append(f"- {row.method}, {row.half}: detection {100 * row.detection_rate:.1f}%; surrogate FPR {100 * row.surrogate_fpr:.2f}%.")
    lines.extend(["", "## Paired detection contrasts", ""])
    for row in comparisons.itertuples(index=False):
        lines.append(f"- {row.comparison}: {100 * row.estimate:.2f} percentage points "
                     f"(pointwise 95% CI {100 * row.ci95_low:.2f} to {100 * row.ci95_high:.2f}).")
    lines.extend(["", "## Interpretation", "",
                  "The interaction estimates how the Hilbert-minus-Hann gain changes between halves. "
                  "An interval containing zero does not establish equivalence or absence of an exposure effect.", "",
                  "Review FPR in each half before interpreting detection differences: pooled calibration "
                  "targets 5% overall and does not guarantee 5% separately in each half.", "",
                  "Intervals resample participants and conditions with pairing preserved; models are not refitted "
                  "inside the bootstrap. These are conditional, pointwise exploratory intervals, without multiplicity adjustment.", "",
                  "PPC summaries at 4 and 8 Hz describe phase-consistency changes at equal trial counts. "
                  "They do not identify memorization, perception, fatigue, or adaptation as a cause. "
                  "Early/late refers to repetition within each stream, not total prior exposure to a vowel "
                  "sequence across conditions. Membership uses the source checkpoint's natural-order convention.", "",
                  "No ten-trial threshold is reused. No cumulative or sequential endpoint is evaluated. "
                  "The source Hilbert filtering margins remain part of the pipeline contrast.", "",
                  "Full scores, per-half calibration, phase measurements, and trial membership are in tables/. "
                  "Fold checkpoints contain null scores, model weights, standardization, and training indices."])
    atomic_write_text(run.path / "report.md", "\n".join(lines) + "\n")
    run.progress("report", 1, 1, "report.md", "completed")
