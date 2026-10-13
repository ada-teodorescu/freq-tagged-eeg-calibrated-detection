from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_detection.detectors import deterministic_subsets
from eeg_detection.evaluation import binomial_interval
from ..runtime import RunContext, atomic_save_npz, atomic_write_json


def _harmonic_scores(signal: np.ndarray, null: np.ndarray, subsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = signal[subsets]
    units = np.divide(selected, np.abs(selected), out=np.zeros_like(selected), where=np.abs(selected) > 0)
    positive = np.mean(np.abs(units.mean(axis=1)) ** 2, axis=1)
    selected_null = null[subsets]
    null_units = np.divide(selected_null, np.abs(selected_null), out=np.zeros_like(selected_null), where=np.abs(selected_null) > 0)
    negative = np.mean(np.abs(null_units.mean(axis=1)) ** 2, axis=1)
    return positive, negative


def _participant_harmonics(context: RunContext, participant: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    archive = np.load(context.config.baseline_detection_run / "cache" / "stored_matlab" / f"participant_{participant:02d}.npz")
    signal, null = archive["signal"], archive["null"]
    subsets = deterministic_subsets(10, k, context.config.subset_cap, context.config.seed)
    positive = np.empty((45, len(subsets), 4), dtype=float)
    negative = np.empty((45, len(subsets), 4, 18), dtype=float)
    for stream in range(45): positive[stream], negative[stream] = _harmonic_scores(signal[stream], null[stream], subsets)
    return positive, negative


def fit_train_only_weights(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Standardized target-minus-neighbor effects; accepts training folds only."""
    signal_mean = np.asarray(positive).mean(axis=tuple(range(np.asarray(positive).ndim - 1)))
    axes = tuple(index for index in range(np.asarray(negative).ndim) if index != np.asarray(negative).ndim - 2)
    null_mean = np.asarray(negative).mean(axis=axes)
    null_sd = np.asarray(negative).std(axis=axes, ddof=1)
    effects = np.maximum((signal_mean - null_mean) / np.maximum(null_sd, np.finfo(float).eps), 0.0)
    if not np.any(effects > 0): return np.full(4, 0.25)
    return effects / effects.sum()


def _method_scores(method: str, positive: np.ndarray, negative: np.ndarray, k: int, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if method == "D1_4Hz": return positive[..., 0], negative[..., 0, :]
    if method == "D8_all_equal":
        return np.sum(2 * k * positive, axis=-1), np.sum(2 * k * negative, axis=-2)
    if method == "D8_4_8_equal":
        return np.sum(2 * k * positive[..., :2], axis=-1), np.sum(2 * k * negative[..., :2, :], axis=-2)
    if method == "D8_train_weighted":
        return np.einsum("...h,h->...", 2 * k * positive, weights), np.einsum("...hb,h->...b", 2 * k * negative, weights)
    raise ValueError(method)


def _empirical_threshold_fast(values: np.ndarray, target_fpr: float) -> float:
    flattened = np.asarray(values, dtype=float).reshape(-1)
    index = int(np.ceil((1.0 - target_fpr) * (flattened.size + 1))) - 1
    index = int(np.clip(index, 0, flattened.size - 1))
    return float(np.partition(flattened, index)[index])


def _auc_against_sorted_null(signal_scores: np.ndarray, sorted_null: np.ndarray) -> float:
    values = np.asarray(signal_scores).reshape(-1)
    left = np.searchsorted(sorted_null, values, side="left")
    right = np.searchsorted(sorted_null, values, side="right")
    return float(np.mean((left + right) / (2.0 * sorted_null.size)))


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "01_stored_d8"; total = 25 * len(context.config.k_values); context.start_stage(stage, total)
    outputs = []; completed = 0
    for participant in range(25):
        for k in context.config.k_values:
            path = context.checkpoint_dir / stage / f"participant_{participant:02d}_k{k:02d}.npz"
            if not context.checkpoint_valid(path) or force:
                positive, negative = _participant_harmonics(context, participant, k)
                atomic_save_npz(path, positive=positive, negative=negative)
                context.mark_checkpoint(path, {"participant": participant, "k": k, "source": "stored_matlab"})
            outputs.append(path); completed += 1
            context.progress(stage, completed, total, f"participant {participant + 1}, K={k}")
    rows, calibration_rows, weight_rows = [], [], []
    for k in context.config.k_values:
        archives = []
        for participant in range(25):
            with np.load(context.checkpoint_dir / stage / f"participant_{participant:02d}_k{k:02d}.npz") as archive:
                archives.append((archive["positive"], archive["negative"]))
        for heldout in range(25):
            train_positive = np.concatenate([archives[p][0] for p in range(25) if p != heldout])
            train_negative = np.concatenate([archives[p][1] for p in range(25) if p != heldout])
            weights = fit_train_only_weights(train_positive, train_negative)
            for harmonic, weight in zip(context.config.harmonics_hz, weights, strict=True):
                weight_rows.append({"heldout_participant": heldout, "k": k, "harmonic_hz": harmonic, "weight": weight,
                                    "fit_participants": 24, "heldout_excluded": True})
            held_positive_h, held_negative_h = archives[heldout]
            for method in context.config.stored_methods:
                held_positive, held_negative = _method_scores(method, held_positive_h, held_negative_h, k, weights)
                _, training_null = _method_scores(method, train_positive, train_negative, k, weights)
                threshold = _empirical_threshold_fast(training_null, context.config.target_fpr)
                held_null = held_negative.reshape(-1); false_positives = int(np.count_nonzero(held_null > threshold))
                sorted_held_null = np.sort(held_null)
                ci_low, ci_high = binomial_interval(false_positives, held_null.size)
                calibration_rows.append({"heldout_participant": heldout, "method": method, "k": k,
                    "threshold": threshold, "heldout_fpr": false_positives / held_null.size,
                    "ci_low": ci_low, "ci_high": ci_high, "n_heldout_null": held_null.size})
                for stream in range(45):
                    scores = held_positive[stream]
                    rows.append({"branch": "stored_matlab", "participant": heldout, "condition": stream // 3,
                        "stream": stream, "method": method, "k": k, "mean_score": float(np.mean(scores)),
                        "detection_rate": float(np.mean(scores > threshold)),
                        "auc": _auc_against_sorted_null(scores, sorted_held_null), "threshold": threshold})
    score_path = context.table_dir / "stored_d8_scores.csv"; pd.DataFrame(rows).to_csv(score_path, index=False)
    calibration_path = context.table_dir / "stored_d8_calibration.csv"; pd.DataFrame(calibration_rows).to_csv(calibration_path, index=False)
    weights_path = context.table_dir / "stored_train_only_weights.csv"; pd.DataFrame(weight_rows).to_csv(weights_path, index=False)
    report = {"source": "stored_matlab", "methods": list(context.config.stored_methods),
              "weight_training": "24 participants in each LOPO fold; held participant excluded",
              "null_calibration": "18 neighboring frequency bins, LOPO", "branches_numerically_merged": False}
    report_path = context.run_dir / "stored_d8_followup.json"; atomic_write_json(report_path, report)
    outputs.extend([score_path, calibration_path, weights_path, report_path]); context.finish_stage(stage, outputs, report)
    return outputs, report
