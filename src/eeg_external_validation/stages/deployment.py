from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from eeg_algorithm_study.algorithms import fit_monotone_logistic
from eeg_advanced_study.algorithms import calibrate_pocock_boundaries
from eeg_core.runtime import atomic_save_npz
from eeg_detection.evaluation import empirical_threshold
from ..runtime import RunContext


METHOD = "H2_Hilbert4s_Regional4H"


def model_path(context: RunContext) -> Path:
    return context.checkpoint_dir / "05_deployment_model" / "deployment_models.npz"


def _fit(pos: np.ndarray, null: np.ndarray, config):
    values = np.concatenate((pos, null))
    labels = np.concatenate((np.ones(len(pos), np.int8), np.zeros(len(null), np.int8)))
    return fit_monotone_logistic(values, labels, config.locked_c,
                                 config.maximum_iterations, config.tolerance)


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    stage, config = "05_deployment_model", context.config
    context.start_stage(stage, len(config.k_values) + 1)
    destination = model_path(context)
    record_path = context.table_dir / "deployment_model_records.csv"
    if context.checkpoint_valid(destination) and record_path.exists() and not force:
        context.progress(stage, len(config.k_values) + 1, len(config.k_values) + 1, "deployment checkpoint")
        outputs = [destination, record_path]
        context.finish_stage(stage, outputs, {"reused": True}); return outputs, {"reused": True}
    static_path = config.baseline_confirmatory_run / "checkpoints" / "02_hilbert_condition" / "condition_features.npz"
    sequential_path = config.baseline_confirmatory_run / "checkpoints" / "03_hilbert_sequential" / "trajectories.npz"
    with np.load(static_path, allow_pickle=False) as archive:
        static_pos = archive[f"{METHOD}_pos"].reshape(-1, 20)
        static_null = archive[f"{METHOD}_null"].reshape(-1, 20)
    static_model = _fit(static_pos, static_null, config)
    static_threshold = empirical_threshold(static_model.decision_function(static_null), config.target_fpr)
    context.progress(stage, 1, len(config.k_values) + 1, "static K10 model")
    with np.load(sequential_path, allow_pickle=False) as archive:
        seq_pos = archive[f"{METHOD}_pos"]
        seq_null = archive[f"{METHOD}_null"]
    coefficients, intercepts, means, scales = [], [], [], []
    calibration_scores = np.empty((np.prod(seq_null.shape[:3]), len(config.k_values)), float)
    records = [{"look_k": 10, "endpoint": "static", "coefficients": json.dumps(static_model.coefficients.tolist()),
                "intercept": static_model.intercept, "standardizer_mean": json.dumps(static_model.standardizer.mean.tolist()),
                "standardizer_scale": json.dumps(static_model.standardizer.scale.tolist()),
                "threshold": static_threshold}]
    for look, k in enumerate(config.k_values):
        pos = seq_pos[:, :, look].reshape(-1, 20); null = seq_null[:, :, :, look].reshape(-1, 20)
        model = _fit(pos, null, config); coefficients.append(model.coefficients); intercepts.append(model.intercept)
        means.append(model.standardizer.mean); scales.append(model.standardizer.scale)
        calibration_scores[:, look] = model.decision_function(null)
        records.append({"look_k": k, "endpoint": "sequential", "coefficients": json.dumps(model.coefficients.tolist()),
                        "intercept": model.intercept, "standardizer_mean": json.dumps(model.standardizer.mean.tolist()),
                        "standardizer_scale": json.dumps(model.standardizer.scale.tolist()), "threshold": np.nan})
        context.progress(stage, look + 2, len(config.k_values) + 1, f"sequential K={k}")
    calibrated = calibrate_pocock_boundaries(calibration_scores, config.target_fpr)
    for row, boundary in zip(records[1:], calibrated.boundaries, strict=True): row["threshold"] = float(boundary)
    atomic_save_npz(destination, static_coefficients=static_model.coefficients, static_intercept=static_model.intercept,
                    static_mean=static_model.standardizer.mean, static_scale=static_model.standardizer.scale,
                    static_threshold=static_threshold, sequential_coefficients=np.stack(coefficients),
                    sequential_intercepts=np.asarray(intercepts), sequential_means=np.stack(means),
                    sequential_scales=np.stack(scales), sequential_boundaries=calibrated.boundaries,
                    k_values=np.asarray(config.k_values))
    context.mark_checkpoint(destination, {"training_source": str(config.baseline_confirmatory_run),
                                          "external_data_used": False})
    pd.DataFrame(records).to_csv(record_path, index=False)
    outputs = [destination, record_path]
    summary = {"training_source": str(config.baseline_confirmatory_run), "external_data_used": False,
               "static_threshold": float(static_threshold), "sequential_training_fwer": calibrated.familywise_fpr}
    context.finish_stage(stage, outputs, summary); return outputs, summary
