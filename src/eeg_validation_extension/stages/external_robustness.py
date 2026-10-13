from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import signal

from eeg_accuracy_study.features import region_indices
from eeg_advanced_study.algorithms import calibrate_pocock_boundaries, first_crossings
from eeg_algorithm_study.algorithms import fit_monotone_logistic, MonotoneModel
from eeg_core.algorithms import Standardizer
from eeg_core.runtime import atomic_save_npz, atomic_write_json
from eeg_detection.evaluation import empirical_threshold
from eeg_external_validation.stages.scores import condition_trajectories
from eeg_external_validation.stages.features import _prepare_raw
from eeg_external_validation.config import load_config as load_external_config
from eeg_external_validation.statistics import participant_cluster_interval
from ..features import condition_features
from ..runtime import RunContext


def _fit(pos: np.ndarray, null: np.ndarray, config):
    values = np.concatenate((pos, null)); labels = np.concatenate((np.ones(len(pos), np.int8), np.zeros(len(null), np.int8)))
    return fit_monotone_logistic(values, labels, config.locked_c, config.maximum_iterations, config.tolerance)


def _model(coefficients, intercept, mean, scale, c_value) -> MonotoneModel:
    return MonotoneModel(np.asarray(coefficients, float), float(intercept),
                         Standardizer(np.asarray(mean, float), np.asarray(scale, float)), c_value, True)


def _shared_training(context: RunContext, participants: list[str], shared_indices: np.ndarray,
                     shared_regions: tuple[np.ndarray, ...], force: bool) -> tuple[Path, dict]:
    config = context.config; destination = context.checkpoint_dir / "05_external_robustness" / "shared28_models.npz"
    if context.checkpoint_valid(destination) and not force:
        return destination, {"reused": True}
    positives, nulls = [], []
    for participant_index, participant in enumerate(participants):
        source = config.baseline_confirmatory_run / "checkpoints/01_coefficients" / f"{participant}.npz"
        with np.load(source, allow_pickle=False) as archive: coefficients = archive["hilbert"][:, :, shared_indices, :]
        positive, null = condition_features(coefficients, config, shared_regions, participant_index, config.k_values)
        positives.append(positive); nulls.append(null)
    positive, null = np.stack(positives), np.stack(nulls)
    static_pos = positive[:, :, -1].reshape(-1, positive.shape[-1]); static_null = null[:, :, :, -1].reshape(-1, null.shape[-1])
    static_model = _fit(static_pos, static_null, config); static_threshold = empirical_threshold(static_model.decision_function(static_null), config.target_fpr)
    coefficients, intercepts, means, scales = [], [], [], []
    calibration = np.empty((np.prod(null.shape[:3]), len(config.k_values)))
    for look in range(len(config.k_values)):
        pos = positive[:, :, look].reshape(-1, positive.shape[-1]); nul = null[:, :, :, look].reshape(-1, null.shape[-1])
        model = _fit(pos, nul, config); coefficients.append(model.coefficients); intercepts.append(model.intercept)
        means.append(model.standardizer.mean); scales.append(model.standardizer.scale); calibration[:, look] = model.decision_function(nul)
    calibrated = calibrate_pocock_boundaries(calibration, config.target_fpr)
    atomic_save_npz(destination, static_coefficients=static_model.coefficients, static_intercept=static_model.intercept,
                    static_mean=static_model.standardizer.mean, static_scale=static_model.standardizer.scale,
                    static_threshold=static_threshold, sequential_coefficients=np.stack(coefficients),
                    sequential_intercepts=np.asarray(intercepts), sequential_means=np.stack(means),
                    sequential_scales=np.stack(scales), sequential_boundaries=calibrated.boundaries,
                    shared_channels=np.asarray(config.shared_channels))
    context.mark_checkpoint(destination, {"external_data_used_for_training": False, "channels": 28})
    return destination, {"static_threshold": float(static_threshold), "sequential_training_fwer": calibrated.familywise_fpr}


def _load_models(path: Path, config):
    archive = np.load(path, allow_pickle=False)
    static = _model(archive["static_coefficients"], archive["static_intercept"], archive["static_mean"], archive["static_scale"], config.locked_c)
    sequential = [_model(c, i, m, s, config.locked_c) for c, i, m, s in zip(
        archive["sequential_coefficients"], archive["sequential_intercepts"], archive["sequential_means"], archive["sequential_scales"], strict=True)]
    return archive, static, sequential


def _score_coefficients(coefficients: np.ndarray, regions: tuple[np.ndarray, ...], config,
                        static_model, sequential_models, static_threshold: float, boundaries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = condition_trajectories(coefficients, config.k_values, regions)
    static_scores = static_model.decision_function(features[:, -1])
    sequential_scores = np.column_stack([model.decision_function(features[:, look]) for look, model in enumerate(sequential_models)])
    first = first_crossings(sequential_scores, boundaries, config.k_values)
    return static_scores > static_threshold, first <= config.k_values[-1]


def _summary_rows(scores: pd.DataFrame, config, analysis: str, seed_salt: int) -> list[dict]:
    rows = []
    for task_index, (task, group) in enumerate(scores.groupby("task", sort=True)):
        for endpoint_index, column in enumerate(("static_false_positive", "sequential_false_positive")):
            rates = group.groupby("participant")[column].mean().to_numpy(float)
            interval = participant_cluster_interval(rates, config.bootstrap_iterations,
                                                    config.seed + seed_salt + 10 * task_index + endpoint_index,
                                                    config.one_sided_confidence)
            rows.append({"analysis": analysis, "task": task, "endpoint": column.replace("_false_positive", ""),
                         "participants": len(rates), "units": len(group),
                         "false_positives": int(group[column].sum()), **interval})
    return rows


def randomized_nonoverlapping(onsets: np.ndarray, duration: float, recording_duration: float,
                              count: int, resamples: int, seed: int) -> list[np.ndarray]:
    """Choose random event windows while preserving greedy maximum-cardinality feasibility."""
    candidates = np.sort(np.asarray(onsets, dtype=float))
    candidates = candidates[(candidates >= 0) & (candidates + duration <= recording_duration)]
    if len(candidates) < count: raise ValueError(f"only {len(candidates)} valid event-anchored windows")
    next_index = np.searchsorted(candidates, candidates + duration, side="left")
    capacity = np.zeros(len(candidates) + 1, dtype=int)
    for index in range(len(candidates) - 1, -1, -1):
        capacity[index] = max(capacity[index + 1], 1 + capacity[next_index[index]])
    if capacity[0] < count:
        raise ValueError(f"event sequence provides at most {capacity[0]} non-overlapping windows, need {count}")
    rng = np.random.default_rng(seed); outputs = []
    for _ in range(resamples):
        chosen: list[float] = []; start_index = 0
        for remaining in range(count, 0, -1):
            feasible = []
            for index in range(start_index, len(candidates)):
                if 1 + capacity[next_index[index]] >= remaining: feasible.append(index)
                elif capacity[index] < remaining: break
            if not feasible: raise RuntimeError("window-selection feasibility invariant failed")
            selected = feasible[int(rng.integers(0, len(feasible)))]
            chosen.append(float(candidates[selected])); start_index = int(next_index[selected])
        outputs.append(np.asarray(chosen))
    return outputs


def rest_selections(recording_duration: float, duration: float, count: int,
                    resamples: int, seed: int, edge_seconds: float = 5.0) -> list[np.ndarray]:
    usable = recording_duration - 2 * edge_seconds - count * duration
    if usable < 0: raise ValueError("recording is too short for rest-window resampling")
    rng = np.random.default_rng(seed)
    return [edge_seconds + rng.uniform(0, usable) + np.arange(count) * duration for _ in range(resamples)]


def continuous_window_coefficients(raw, starts_seconds: np.ndarray, frequencies_hz: tuple[float, ...],
                                   epoch_seconds: float, retained_edge_seconds: float,
                                   order: int, half_bandwidth_hz: float) -> np.ndarray:
    """Extract Hilbert coefficients after filtering the continuous recording once."""
    sampling_rate = int(round(raw.info["sfreq"])); data = raw.get_data().astype(np.float64)
    starts = np.rint(np.asarray(starts_seconds) * sampling_rate).astype(int)
    left = starts + int(round(retained_edge_seconds * sampling_rate))
    right = starts + int(round((epoch_seconds - retained_edge_seconds) * sampling_rate))
    if np.any(left < 0) or np.any(right > data.shape[-1]) or np.any(right <= left):
        raise ValueError("selected external window lies outside the continuous recording")
    sample_index = np.arange(data.shape[-1], dtype=float); output = []
    for frequency in frequencies_hz:
        sos = signal.butter(order, [frequency - half_bandwidth_hz, frequency + half_bandwidth_hz],
                            btype="bandpass", fs=sampling_rate, output="sos")
        filtered = signal.sosfiltfilt(sos, data, axis=-1)
        analytic = signal.hilbert(filtered, axis=-1)
        demodulated = (analytic * np.exp(-2j * np.pi * frequency * sample_index / sampling_rate)).astype(np.complex64)
        cumulative = np.concatenate((np.zeros((data.shape[0], 1), np.complex64),
                                     np.cumsum(demodulated, axis=-1, dtype=np.complex64)), axis=-1)
        output.append(((cumulative[:, right] - cumulative[:, left]) / (right - left)[None, :]).T)
    return np.stack(output, axis=-1).astype(np.complex64)


def _raw_window_selection_scores(context: RunContext, static_model, sequential_models,
                                 static_threshold: float, boundaries: np.ndarray,
                                 regions: tuple[np.ndarray, ...]) -> pd.DataFrame:
    config = context.config; external_config = load_external_config(config.project_root / "external_validation_config.toml")
    inventory = pd.read_csv(config.baseline_external_run / "tables/recording_inventory.csv")
    eligibility = pd.read_csv(config.baseline_external_run / "tables/preprocessing_eligibility.csv")
    membership = pd.read_csv(config.baseline_external_run / "tables/negative_unit_membership.csv")
    eligible_keys = membership[["participant", "task"]].drop_duplicates()
    records = inventory.merge(eligibility, on=["participant", "task"], suffixes=("", "_eligibility"))
    records = records.merge(eligible_keys, on=["participant", "task"])
    rows = []
    for record_index, record in enumerate(records.itertuples(index=False)):
        bad_targets = tuple(json.loads(record.dataset_bad_target_channels))
        raw, _, _ = _prepare_raw(Path(record.vhdr_path), external_config, bad_targets)
        count = external_config.auditory_windows_per_participant if record.task == external_config.primary_task else external_config.rest_windows_per_participant
        seed = int(np.random.SeedSequence([config.seed, record_index]).generate_state(1)[0])
        if record.task == external_config.primary_task:
            events = pd.read_csv(Path(record.event_path), sep="\t")
            selections = randomized_nonoverlapping(events["onset"].to_numpy(float), external_config.epoch_seconds,
                float(record.duration_seconds), count, config.window_resamples, seed)
        else:
            selections = rest_selections(float(record.duration_seconds), external_config.epoch_seconds,
                                         count, config.window_resamples, seed)
        all_samples = np.unique(np.concatenate([np.rint(values * external_config.sampling_rate).astype(np.int64) for values in selections]))
        all_starts = all_samples / external_config.sampling_rate
        coefficients = continuous_window_coefficients(raw, all_starts, external_config.harmonics_hz,
            external_config.epoch_seconds, 0.5, order=4, half_bandwidth_hz=0.5)
        lookup = {int(sample): index for index, sample in enumerate(all_samples)}
        for resample, selected in enumerate(selections):
            selected_samples = np.rint(selected * external_config.sampling_rate).astype(np.int64)
            selected_coefficients = coefficients[[lookup[int(sample)] for sample in selected_samples]]
            units = selected_coefficients.reshape(-1, external_config.trials_per_stream,
                                                  external_config.streams_per_unit,
                                                  len(external_config.target_channels), len(external_config.harmonics_hz)).transpose(0, 2, 1, 3, 4)
            static_fp, sequential_fp = _score_coefficients(units, regions, config, static_model, sequential_models,
                                                            static_threshold, boundaries)
            rows.append({"participant": record.participant, "task": record.task, "resample": resample,
                         "units": len(static_fp), "static_fpr": float(np.mean(static_fp)),
                         "sequential_fpr": float(np.mean(sequential_fp))})
        del raw
    return pd.DataFrame(rows)


def run(context: RunContext, force: bool = False) -> tuple[list[Path], dict]:
    stage, config = "05_external_robustness", context.config
    participants = json.loads((context.run_dir / "design_lock.json").read_text(encoding="utf-8"))["participants"]
    sample_path = config.baseline_confirmatory_run / "checkpoints/01_coefficients" / f"{participants[0]}.npz"
    with np.load(sample_path, allow_pickle=False) as sample: labels = tuple(sample["channel_labels"].astype(str))
    shared_indices = np.asarray([labels.index(channel) for channel in config.shared_channels])
    pruned_regions = tuple(tuple(ch for ch in region if ch in set(config.shared_channels)) for region in config.regions)
    shared_regions = region_indices(config.shared_channels, pruned_regions)
    external_feature_paths = sorted((config.baseline_external_run / "checkpoints/04_hilbert_features").glob("*.npz"))
    context.start_stage(stage, len(external_feature_paths) + 2)
    model_path, training_summary = _shared_training(context, participants, shared_indices, shared_regions, force)
    context.progress(stage, 1, len(external_feature_paths) + 2, "shared 28-channel model")
    archive28, static28, sequential28 = _load_models(model_path, config)
    rows28 = []
    for index, path in enumerate(external_feature_paths, 2):
        with np.load(path, allow_pickle=False) as source:
            source_labels = tuple(source["channel_labels"].astype(str)); indices = np.asarray([source_labels.index(ch) for ch in config.shared_channels])
            coefficients = source["coefficients"][:, :, :, indices, :]
            participant, task = str(source["participant"]), str(source["task"])
        static_fp, sequential_fp = _score_coefficients(coefficients, shared_regions, config, static28, sequential28,
            float(archive28["static_threshold"]), archive28["sequential_boundaries"])
        for unit, (sfp, qfp) in enumerate(zip(static_fp, sequential_fp, strict=True), 1):
            rows28.append({"participant": participant, "task": task, "unit": unit,
                           "static_false_positive": bool(sfp), "sequential_false_positive": bool(qfp)})
        context.progress(stage, index, len(external_feature_paths) + 2, f"{participant} {task}")
    archive28.close()
    scores28 = pd.DataFrame(rows28); path28 = context.table_dir / "external_shared28_scores.csv"; scores28.to_csv(path28, index=False)

    # Generate genuinely different non-overlapping windows. Filtering/Hilbert transforms are
    # performed once per continuous recording; cumulative demodulated sums make extraction cheap.
    deployment = config.baseline_external_run / "checkpoints/05_deployment_model/deployment_models.npz"
    archive32, static32, sequential32 = _load_models(deployment, config)
    regions32 = region_indices(labels, config.regions)
    selection_scores = _raw_window_selection_scores(context, static32, sequential32,
        float(archive32["static_threshold"]), archive32["sequential_boundaries"], regions32)
    archive32.close()
    selection_path = context.table_dir / "external_window_selection_robustness.csv"
    selection_scores.to_csv(selection_path, index=False)
    context.progress(stage, len(external_feature_paths) + 2, len(external_feature_paths) + 2, "raw window-selection resamples")

    baseline_scores = pd.read_csv(config.baseline_external_run / "tables/external_negative_scores.csv")
    artifacts = pd.read_csv(config.baseline_external_run / "tables/epoch_artifact_sensitivity.csv")
    unit_rejected = artifacts.groupby(["participant", "task", "unit"])["sensitivity_rejected"].any().rename("unit_rejected").reset_index()
    artifact_complete = baseline_scores.merge(unit_rejected, on=["participant", "task", "unit"], how="left")
    artifact_complete = artifact_complete[~artifact_complete["unit_rejected"].fillna(True)]
    summaries = _summary_rows(baseline_scores, config, "locked_primary", 0)
    summaries += _summary_rows(artifact_complete, config, "artifact_complete_case", 1000)
    summaries += _summary_rows(scores28, config, "shared_28_channel", 2000)
    summary_path = context.table_dir / "external_robustness_summary.csv"; pd.DataFrame(summaries).to_csv(summary_path, index=False)
    summary = {"shared_channel_count": 28, "window_selection_resamples": config.window_resamples,
               "window_selection_method": "continuous filtering plus cumulative complex demodulation",
               **training_summary}
    report = context.run_dir / "external_robustness.json"; atomic_write_json(report, summary)
    outputs = [model_path, path28, selection_path, summary_path, report]
    context.finish_stage(stage, outputs, summary); return outputs, summary
