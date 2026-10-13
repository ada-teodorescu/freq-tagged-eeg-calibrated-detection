from __future__ import annotations

import numpy as np
import pandas as pd

from ..detectors import detector_score
from ..evaluation import binomial_interval, empirical_threshold
from ..runtime import RunContext, atomic_save_npz, atomic_write_json


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "02_null_validation"
    config = context.config
    detectors = [value for value in config.detectors if value != "D7"]
    batch_size = 50
    n_checkpoints = int(np.ceil(config.surrogate_batches / batch_size))
    context.start_stage(stage, n_checkpoints)
    outputs = []
    for checkpoint_index in range(n_checkpoints):
        start = checkpoint_index * batch_size
        stop = min(config.surrogate_batches, start + batch_size)
        path = context.checkpoint_dir / stage / f"batch_{start:04d}_{stop:04d}.npz"
        if not context.checkpoint_valid(path) or force:
            scores = np.empty((stop - start, len(config.k_values), len(detectors)), dtype=float)
            loaded_participant = None
            empirical = None
            for local, batch in enumerate(range(start, stop)):
                rng = np.random.default_rng(np.random.SeedSequence([config.seed, 2, batch]))
                participant = batch % 25
                stream = (batch // 25) % 45
                if participant != loaded_participant:
                    empirical = np.load(
                        context.cache_dir / "stored_matlab" / f"participant_{participant:02d}.npz"
                    )["signal"]
                    loaded_participant = participant
                amplitudes = np.abs(empirical[stream])
                coefficients = amplitudes * np.exp(1j * rng.uniform(-np.pi, np.pi, size=amplitudes.shape))
                for k_index, k in enumerate(config.k_values):
                    subset = coefficients[:k]
                    for detector_index, detector in enumerate(detectors):
                        values = subset if detector == "D8" else subset[..., :1]
                        scores[local, k_index, detector_index] = detector_score(detector, values)
            atomic_save_npz(path, scores=scores)
            context.mark_checkpoint(path, {"first_batch": start, "last_batch_exclusive": stop})
        outputs.append(path)
        context.progress(stage, checkpoint_index + 1, n_checkpoints, f"surrogates {start + 1}-{stop}")
    all_scores = np.concatenate([np.load(path)["scores"] for path in outputs], axis=0)
    rows = []
    flagged = []
    folds = np.arange(len(all_scores)) % 5
    for k_index, k in enumerate(config.k_values):
        for detector_index, detector in enumerate(detectors):
            positives = 0
            thresholds = []
            for fold in range(5):
                train = all_scores[folds != fold, k_index, detector_index]
                heldout = all_scores[folds == fold, k_index, detector_index]
                threshold = empirical_threshold(train, config.target_fpr)
                thresholds.append(threshold)
                positives += int(np.count_nonzero(heldout > threshold))
            n_heldout = len(all_scores)
            fpr = positives / n_heldout
            low, high = binomial_interval(positives, n_heldout)
            is_flagged = not (low <= config.target_fpr <= high)
            if is_flagged: flagged.append(f"{detector}:K{k}")
            rows.append({"branch": "stored_matlab_phase_randomized", "detector": detector, "k": k,
                         "threshold": float(np.mean(thresholds)),
                         "threshold_min": float(np.min(thresholds)),
                         "threshold_max": float(np.max(thresholds)),
                         "heldout_fpr": fpr, "ci_low": low, "ci_high": high,
                         "n_train_per_fold": int(np.count_nonzero(folds != 0)),
                         "n_heldout_total": n_heldout, "validation_folds": 5,
                         "flagged": is_flagged})
    table_path = context.table_dir / "surrogate_null_calibration.csv"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    report = {"surrogate_batches": config.surrogate_batches, "seed": config.seed,
              "amplitude_source": "stored_matlab per-trial coefficients",
              "phase_randomization": "independent uniform phases preserving every sampled coefficient magnitude",
              "heldout_fraction_per_fold": 0.2, "validation_folds": 5,
              "flagged_detector_k": flagged}
    report_path = context.run_dir / "surrogate_null_validation.json"
    atomic_write_json(report_path, report)
    outputs.extend([table_path, report_path])
    context.finish_stage(stage, outputs, report)
    return outputs, report
