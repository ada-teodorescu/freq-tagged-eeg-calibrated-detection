"""Real ablation: 5-region regional detector vs a single whole-scalp global-average
detector, holding everything else fixed (Hilbert estimator, K=10 static look,
participant-held-out nonnegative logistic fusion, 5% empirical-threshold FPR
calibration). Reuses the exact same functions as the frozen confirmatory pipeline
(eeg_confirmatory_study.stages.hilbert_condition) so the regional numbers should
reproduce \\HilbertStaticSensitivity / \\HilbertStaticFPRPrecise as a sanity check.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from eeg_confirmatory_study.config import load_config
from eeg_accuracy_study.features import region_indices
from eeg_advanced_study.algorithms import phase_trajectories, temporal_shift_multipliers
from eeg_algorithm_study.algorithms import fit_monotone_logistic
from eeg_detection.evaluation import empirical_threshold, auc_from_scores

CONFIG_PATH = "confirmatory_study_config.toml"
RUN_DIR = Path("outputs/confirmatory_study/4d6a54f6f902")
METHOD_KEY = "hilbert"  # matches H2_Hilbert4s_Regional4H, the paper's primary detector


def build_rotations(config, participant_index: int, stream: int, seed_salt: int = 0) -> np.ndarray:
    blocks = []
    for seed in config.surrogate_seeds:
        sequence = np.random.SeedSequence([seed, participant_index, stream, seed_salt])
        blocks.append(temporal_shift_multipliers(
            config.trials_per_stream, tuple(range(1, len(config.harmonics_hz) + 1)),
            config.surrogates_per_seed, sequence,
        ))
    return np.concatenate(blocks, axis=0)


def main() -> None:
    config = load_config(CONFIG_PATH)
    lock = json.loads((RUN_DIR / "design_lock.json").read_text(encoding="utf-8"))
    participants = lock["participants"]
    channel_labels = tuple(lock["channel_labels"])
    regions = region_indices(channel_labels, config.regions)
    n_regions, n_harm = len(regions), len(config.harmonics_hz)
    conditions, streams_per = config.conditions, config.streams_per_condition
    scale = 2.0 * config.trials_per_stream

    pos_regional_all, pos_global_all = [], []
    null_regional_all, null_global_all = [], []

    for p_idx, participant in enumerate(participants):
        npz_path = RUN_DIR / "checkpoints" / "01_coefficients" / f"{participant}.npz"
        with np.load(npz_path) as archive:
            coefficients = archive[METHOD_KEY]  # streams x trials x channels x harmonics

        global_pos, regional_pos = phase_trajectories(coefficients, (config.trials_per_stream,), regions)

        null_global = np.empty((config.streams, config.surrogates, n_harm), np.float32)
        null_regional = np.empty((config.streams, config.surrogates, n_regions, n_harm), np.float32)
        for stream in range(config.streams):
            rotations = build_rotations(config, p_idx, stream)
            g_scores, r_scores = phase_trajectories(
                coefficients[stream:stream + 1], (config.trials_per_stream,), regions, rotations,
            )
            null_global[stream] = g_scores[0, :, 0]
            null_regional[stream] = r_scores[0, :, 0]

        pos_g = global_pos[:, 0].reshape(conditions, streams_per, n_harm).mean(axis=1)
        pos_r = regional_pos[:, 0].reshape(conditions, streams_per, n_regions, n_harm).mean(axis=1)
        null_g = null_global.reshape(conditions, streams_per, config.surrogates, n_harm).mean(axis=1)
        null_r = null_regional.reshape(conditions, streams_per, config.surrogates, n_regions, n_harm).mean(axis=1)

        pos_global_all.append((scale * pos_g).astype(np.float32))
        pos_regional_all.append((scale * pos_r).astype(np.float32))
        null_global_all.append((scale * null_g).astype(np.float32))
        null_regional_all.append((scale * null_r).astype(np.float32))
        print(f"  features computed: {participant} ({p_idx + 1}/{len(participants)})", file=sys.stderr)

    pos_global = np.stack(pos_global_all)      # participants x conditions x harmonics (4 features)
    pos_regional = np.stack(pos_regional_all)  # participants x conditions x regions x harmonics (20 features)
    null_global = np.stack(null_global_all)    # participants x conditions x surrogates x harmonics
    null_regional = np.stack(null_regional_all)

    def evaluate(pos: np.ndarray, null: np.ndarray, label: str) -> dict:
        pos_flat = pos.reshape(pos.shape[0], pos.shape[1], -1)
        null_flat = null.reshape(null.shape[0], null.shape[1], null.shape[2], -1)
        rows = []
        pooled_pos, pooled_null = [], []
        for held, participant in enumerate(participants):
            training = np.arange(len(participants)) != held
            train_pos = pos_flat[training].reshape(-1, pos_flat.shape[-1])
            train_null = null_flat[training].reshape(-1, null_flat.shape[-1])
            values = np.concatenate((train_pos, train_null))
            labels = np.concatenate((np.ones(len(train_pos), np.int8), np.zeros(len(train_null), np.int8)))
            model = fit_monotone_logistic(values, labels, config.locked_c, config.maximum_iterations, config.tolerance)

            positive_scores = model.decision_function(pos_flat[held])
            held_null_flat = null_flat[held].reshape(-1, null_flat.shape[-1])
            held_null_scores = model.decision_function(held_null_flat)
            threshold = empirical_threshold(model.decision_function(train_null), config.target_fpr)

            detected = positive_scores > threshold
            rows.append({
                "participant": participant,
                "sensitivity": float(detected.mean()),
                "fpr": float(np.mean(held_null_scores > threshold)),
                "auc": auc_from_scores(positive_scores, held_null_scores),
            })
            pooled_pos.append(positive_scores)
            pooled_null.append(held_null_scores)

        pooled_pos = np.concatenate(pooled_pos)
        pooled_null = np.concatenate(pooled_null)
        mean_sens = float(np.mean([r["sensitivity"] for r in rows]))
        mean_fpr = float(np.mean([r["fpr"] for r in rows]))
        mean_auc = float(np.mean([r["auc"] for r in rows]))
        pooled_auc = auc_from_scores(pooled_pos, pooled_null)
        print(f"\n=== {label} ({pos_flat.shape[-1]} features) ===")
        print(f"mean per-participant detection rate: {mean_sens*100:.1f}%")
        print(f"mean held-out surrogate FPR:          {mean_fpr*100:.2f}%")
        print(f"mean per-participant AUC:              {mean_auc:.3f}")
        print(f"pooled AUC (all held-out scores):      {pooled_auc:.3f}")
        return {"rows": rows, "mean_sensitivity": mean_sens, "mean_fpr": mean_fpr,
                "mean_auc": mean_auc, "pooled_auc": pooled_auc}

    regional_result = evaluate(pos_regional, null_regional, "Regional (5 regions x 4 harmonics)")
    global_result = evaluate(pos_global, null_global, "Global average (whole scalp x 4 harmonics)")

    gain = regional_result["mean_sensitivity"] - global_result["mean_sensitivity"]
    print(f"\nRegional-minus-global detection-rate gain: {gain*100:.1f} pp")
    print(f"Regional-minus-global AUC gain (mean-of-participant AUC): "
          f"{regional_result['mean_auc'] - global_result['mean_auc']:.3f}")
    print(f"Regional-minus-global AUC gain (pooled AUC): "
          f"{regional_result['pooled_auc'] - global_result['pooled_auc']:.3f}")

    out_path = Path(".tmp/region_vs_global_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"regional": regional_result, "global": global_result}, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to {out_path}")


if __name__ == "__main__":
    main()
