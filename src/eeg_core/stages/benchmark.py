from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd

from eeg_detection.evaluation import binomial_interval
from ..algorithms import fit_logistic_l2, make_training_observations, select_regularization
from ..runtime import RunContext, atomic_write_json
from .features import feature_path


def empirical_threshold_fast(values: np.ndarray, target_fpr: float) -> float:
    flattened = np.asarray(values, dtype=float).reshape(-1)
    if not flattened.size: raise ValueError("no calibration scores")
    index = int(np.ceil((1.0 - target_fpr) * (flattened.size + 1))) - 1
    index = int(np.clip(index, 0, flattened.size - 1))
    return float(np.partition(flattened, index)[index])


def leave_one_participant_thresholds(values: np.ndarray, target_fpr: float) -> np.ndarray:
    """Exact LOPO thresholds from one global sort and per-participant rank correction."""
    array = np.asarray(values, dtype=float)
    participants = array.shape[0]
    per_participant = int(np.prod(array.shape[1:]))
    flattened = array.reshape(-1)
    labels = np.repeat(np.arange(participants), per_participant)
    order = np.argsort(flattened, kind="stable")
    sorted_values, sorted_labels = flattened[order], labels[order]
    training_size = (participants - 1) * per_participant
    rank = int(np.ceil((1.0 - target_fpr) * (training_size + 1))) - 1
    rank = int(np.clip(rank, 0, training_size - 1))
    thresholds = np.empty(participants)
    for heldout in range(participants):
        positions = np.flatnonzero(sorted_labels == heldout)
        low, high = 0, sorted_values.size - 1
        while low < high:
            middle = (low + high) // 2
            nonheld_through_middle = middle + 1 - np.searchsorted(positions, middle, side="right")
            if nonheld_through_middle >= rank + 1: high = middle
            else: low = middle + 1
        thresholds[heldout] = sorted_values[low]
    return thresholds


def auc_fast(signal_scores: np.ndarray, null_scores: np.ndarray) -> float:
    positive = np.asarray(signal_scores, dtype=float).reshape(-1)
    negative = np.sort(np.asarray(null_scores, dtype=float).reshape(-1))
    left = np.searchsorted(negative, positive, side="left")
    right = np.searchsorted(negative, positive, side="right")
    return float(np.mean((left + right) / (2 * negative.size)))


def _decision_chunks(model, values: np.ndarray, chunk_size: int = 500_000) -> np.ndarray:
    flat = np.asarray(values).reshape(-1, values.shape[-1])
    result = np.empty(flat.shape[0], dtype=np.float32)
    for start in range(0, flat.shape[0], chunk_size):
        stop = min(start + chunk_size, flat.shape[0])
        result[start:stop] = model.decision_function(flat[start:stop]).astype(np.float32)
    return result.reshape(values.shape[:-1])


def _load_cell(context: RunContext, participants: list[str], duration: float, k: int) -> dict[str, np.ndarray]:
    collections: dict[str, list[np.ndarray]] = {}
    for participant in participants:
        with np.load(feature_path(context, participant, duration, k), allow_pickle=False) as archive:
            for name in ("positive_phase", "null_phase", "positive_hotelling", "null_hotelling",
                         "positive_joint", "null_joint"):
                collections.setdefault(name, []).append(archive[name])
    return {name: np.stack(values) for name, values in collections.items()}


def _algorithm_arrays(features: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "A1_ITPC": (features["positive_phase"][..., 0], features["null_phase"][..., 0]),
        "A2_Hotelling": (features["positive_hotelling"], features["null_hotelling"]),
        "A3_JointHarmonic": (features["positive_joint"], features["null_joint"]),
    }


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    del force
    stage = "03_nested_benchmark"
    config = context.config
    provenance = json.loads((context.run_dir / "provenance_gate.json").read_text(encoding="utf-8"))
    participants = list(provenance["participants"])
    total = len(config.durations_seconds) * len(config.k_values)
    context.start_stage(stage, total)
    rows: list[dict] = []
    calibration_rows: list[dict] = []
    model_rows: list[dict] = []
    cv_rows: list[dict] = []
    started = time.perf_counter(); completed = 0
    participant_numbers = np.arange(len(participants))
    for duration_index, duration in enumerate(config.durations_seconds):
        for k in config.k_values:
            cell_started = time.perf_counter()
            features = _load_cell(context, participants, duration, k)
            fixed = _algorithm_arrays(features)
            fixed_thresholds = {algorithm: leave_one_participant_thresholds(negative, config.target_fpr)
                                for algorithm, (_, negative) in fixed.items()}
            ml_calibration: list[np.ndarray] = []
            for participant_index in participant_numbers:
                values = features["null_phase"][participant_index].reshape(-1, 4)
                cap = min(config.calibration_null_cap_per_participant, len(values))
                rng = np.random.default_rng(np.random.SeedSequence(
                    [config.seed, duration_index, k, int(participant_index), 771]
                ))
                indices = np.sort(rng.choice(len(values), cap, replace=False)) if cap < len(values) else np.arange(len(values))
                ml_calibration.append(values[indices])
            for heldout, participant in enumerate(participants):
                training = participant_numbers != heldout
                for algorithm, (positive, negative) in fixed.items():
                    threshold = float(fixed_thresholds[algorithm][heldout])
                    held_positive, held_null = positive[heldout], negative[heldout]
                    false_positives = int(np.count_nonzero(held_null > threshold))
                    low, high = binomial_interval(false_positives, held_null.size)
                    calibration_rows.append({
                        "participant": participant, "algorithm": algorithm, "k": k,
                        "duration_seconds": duration, "threshold": threshold,
                        "heldout_fpr": false_positives / held_null.size,
                        "ci_low": low, "ci_high": high, "n_null": held_null.size,
                        "threshold_fit_participants": len(participants) - 1,
                    })
                    for stream in range(config.streams):
                        rows.append({
                            "participant": participant, "participant_index": heldout, "condition": stream // 3,
                            "stream": stream, "algorithm": algorithm, "k": k, "duration_seconds": duration,
                            "detection_rate": float(np.mean(held_positive[stream] > threshold)),
                            "heldout_fpr": float(np.mean(held_null[stream] > threshold)),
                            "auc": auc_fast(held_positive[stream], held_null[stream]),
                            "mean_score": float(np.mean(held_positive[stream])), "threshold": threshold,
                        })
                train_values, train_labels, train_groups = make_training_observations(
                    features["positive_phase"][training], features["null_phase"][training], participant_numbers[training]
                )
                if heldout in set(np.asarray(train_groups).tolist()):
                    raise AssertionError("held-out participant leaked into ML training observations")
                cv_seed = int(np.random.SeedSequence([config.seed, duration_index, k, heldout]).generate_state(1)[0])
                best_c, cv_scores = select_regularization(
                    train_values, train_labels, train_groups, config.c_grid, config.inner_group_folds,
                    cv_seed, config.maximum_iterations, config.tolerance,
                )
                model = fit_logistic_l2(train_values, train_labels, best_c,
                                        maximum_iterations=config.maximum_iterations, tolerance=config.tolerance)
                train_null_features = np.concatenate([ml_calibration[index] for index in participant_numbers[training]])
                train_null_scores = _decision_chunks(model, train_null_features)
                threshold = empirical_threshold_fast(train_null_scores, config.target_fpr)
                held_positive = _decision_chunks(model, features["positive_phase"][heldout])
                held_null = _decision_chunks(model, features["null_phase"][heldout])
                false_positives = int(np.count_nonzero(held_null > threshold))
                low, high = binomial_interval(false_positives, held_null.size)
                calibration_rows.append({
                    "participant": participant, "algorithm": "A4_LearnedFusion", "k": k,
                    "duration_seconds": duration, "threshold": threshold,
                    "heldout_fpr": false_positives / held_null.size, "ci_low": low, "ci_high": high,
                    "n_null": held_null.size, "threshold_fit_participants": len(participants) - 1,
                })
                model_rows.append({
                    "participant": participant, "heldout_participant_index": heldout, "k": k,
                    "duration_seconds": duration, "selected_c": best_c, "intercept": model.intercept,
                    "converged": model.converged,
                    **{f"coefficient_{int(frequency)}hz": float(value)
                       for frequency, value in zip(config.harmonics_hz, model.coefficients, strict=True)},
                    **{f"training_mean_{int(frequency)}hz": float(value)
                       for frequency, value in zip(config.harmonics_hz, model.standardizer.mean, strict=True)},
                    **{f"training_scale_{int(frequency)}hz": float(value)
                       for frequency, value in zip(config.harmonics_hz, model.standardizer.scale, strict=True)},
                    "training_participants": len(participants) - 1, "heldout_excluded": True,
                })
                for c_value, inner_auc in cv_scores.items():
                    cv_rows.append({"participant": participant, "k": k, "duration_seconds": duration,
                                    "c_value": c_value, "inner_grouped_auc": inner_auc})
                for stream in range(config.streams):
                    rows.append({
                        "participant": participant, "participant_index": heldout, "condition": stream // 3,
                        "stream": stream, "algorithm": "A4_LearnedFusion", "k": k,
                        "duration_seconds": duration,
                        "detection_rate": float(np.mean(held_positive[stream] > threshold)),
                        "heldout_fpr": float(np.mean(held_null[stream] > threshold)),
                        "auc": auc_fast(held_positive[stream], held_null[stream]),
                        "mean_score": float(np.mean(held_positive[stream])), "threshold": threshold,
                    })
            completed += 1
            context.progress(stage, completed, total, f"T={duration:g}s K={k} ({time.perf_counter()-cell_started:.1f}s)")
    score_path = context.table_dir / "heldout_scores.csv"
    calibration_path = context.table_dir / "heldout_calibration.csv"
    model_path = context.table_dir / "learned_models.csv"
    cv_path = context.table_dir / "inner_cv_scores.csv"
    pd.DataFrame(rows).to_csv(score_path, index=False)
    pd.DataFrame(calibration_rows).to_csv(calibration_path, index=False)
    pd.DataFrame(model_rows).to_csv(model_path, index=False)
    pd.DataFrame(cv_rows).to_csv(cv_path, index=False)
    summary = {
        "branch": "public_set_fdt", "participants": len(participants), "outer_folds": len(participants),
        "inner_group_folds": config.inner_group_folds, "algorithms": list(config.algorithms),
        "threshold_training_only": True, "scaling_training_only": True, "runtime_seconds": time.perf_counter() - started,
        "ml_calibration_null_cap_per_participant": config.calibration_null_cap_per_participant,
    }
    report_path = context.run_dir / "nested_benchmark.json"
    atomic_write_json(report_path, summary)
    outputs = [score_path, calibration_path, model_path, cv_path, report_path]
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
