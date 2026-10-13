from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd

from cucu_detection.detectors import deterministic_subsets
from cucu_detection.evaluation import auc_from_scores, empirical_threshold
from ..runtime import RunContext, atomic_write_json


METHODS = {
    "F1_D1_4Hz": ("F1", "phase", (0,)),
    "F1_D8_4_8": ("F1", "phase", (0, 1)),
    "F2_D1_4Hz": ("F2", "phase", (0,)),
    "F2_D8_4_8": ("F2", "phase", (0, 1)),
    "F3_line_D1_4Hz": ("F3_line", "phase", (0,)),
    "F3_line_D8_4_8": ("F3_line", "phase", (0, 1)),
    "F3_Thomson_F_4Hz": ("F3_f", "f_statistic", (0,)),
}


def _condition_streams(condition: int) -> range: return range(3 * condition + 1, 3 * condition + 4)


def _archive_path(context: RunContext, participant: int, stream: int):
    return context.cache_dir / "public_set_fdt" / f"participant_{participant:02d}" / f"stream_{stream:02d}.npz"


def _phase_score(values: np.ndarray, subsets: np.ndarray, harmonics: tuple[int, ...], channel: int | None) -> np.ndarray:
    selected = np.asarray(values)[subsets][..., list(harmonics), :]
    magnitude = np.abs(selected)
    units = np.divide(selected, magnitude, out=np.zeros_like(selected), where=magnitude > 0)
    r2 = np.abs(units.mean(axis=1)) ** 2
    if len(harmonics) == 1: statistic = r2[..., 0, :]
    else: statistic = np.sum(2 * subsets.shape[1] * r2, axis=-2)
    return statistic.mean(axis=1) if channel is None else statistic[:, channel]


def _phase_target(values: np.ndarray, subsets: np.ndarray, harmonics: tuple[int, ...], channel: int | None) -> np.ndarray:
    return _phase_score(np.asarray(values)[..., None], subsets, harmonics, channel)[..., 0]


def _phase_surrogate(
    values: np.ndarray, subsets: np.ndarray, harmonics: tuple[int, ...], channel: int | None,
    seed: np.random.SeedSequence, batches: int = 18,
) -> np.ndarray:
    selected = np.asarray(values)[subsets][..., list(harmonics)]
    rng = np.random.default_rng(seed)
    rotations = np.exp(1j * rng.uniform(
        -np.pi, np.pi, size=(len(subsets), subsets.shape[1], 1, len(harmonics), batches)
    ))
    randomized = selected[..., None] * rotations
    magnitude = np.abs(randomized)
    units = np.divide(randomized, magnitude, out=np.zeros_like(randomized), where=magnitude > 0)
    r2 = np.abs(units.mean(axis=1)) ** 2
    if len(harmonics) == 1: statistic = r2[..., 0, :]
    else: statistic = np.sum(2 * subsets.shape[1] * r2, axis=-2)
    return statistic.mean(axis=1) if channel is None else statistic[:, channel]


def _f_score(values: np.ndarray, subsets: np.ndarray, channel: int | None) -> np.ndarray:
    selected = np.log1p(np.asarray(values)[subsets][..., 0, :]).mean(axis=1)
    return selected.mean(axis=1) if channel is None else selected[:, channel]


def _f_target(values: np.ndarray, subsets: np.ndarray, channel: int | None) -> np.ndarray:
    return _f_score(np.asarray(values)[..., None], subsets, channel)[..., 0]


def _channel_quality(archive, method: str) -> np.ndarray:
    prefix, detector, harmonics = METHODS[method]
    if detector == "f_statistic": return np.mean(np.log1p(archive["F3_f_target"][..., 0]), axis=0)
    values = archive[f"{prefix}_target"][..., list(harmonics)]
    units = np.divide(values, np.abs(values), out=np.zeros_like(values), where=np.abs(values) > 0)
    r2 = np.abs(units.mean(axis=0)) ** 2
    return r2[:, 0] if len(harmonics) == 1 else np.sum(r2, axis=-1)


def _qualities(context: RunContext, participant: int) -> dict[str, np.ndarray]:
    result = {method: np.empty((15, 32), dtype=float) for method in METHODS}
    for condition in range(15):
        by_method = {method: [] for method in METHODS}
        for stream in _condition_streams(condition):
            with np.load(_archive_path(context, participant, stream)) as archive:
                for method in METHODS: by_method[method].append(_channel_quality(archive, method))
        for method in METHODS: result[method][condition] = np.mean(by_method[method], axis=0)
    return result


def select_train_only_channel(condition_quality: np.ndarray, training_conditions: list[int]) -> int:
    """Select S1 from explicitly supplied training conditions only."""
    return int(np.argmax(np.asarray(condition_quality)[training_conditions].mean(axis=0)))


def _scores(
    archive, method: str, subsets: np.ndarray, channel: int | None,
    seed: np.random.SeedSequence | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    prefix, detector, harmonics = METHODS[method]
    if detector == "f_statistic":
        return _f_target(archive["F3_f_target"], subsets, channel), _f_score(archive["F3_f_neighbors"], subsets, channel)
    target = archive[f"{prefix}_target"]
    positive = _phase_target(target, subsets, harmonics, channel)
    if prefix == "F2":
        if seed is None: raise ValueError("F2 phase-surrogate calibration requires a deterministic seed")
        negative = _phase_surrogate(target, subsets, harmonics, channel, seed)
    else:
        negative = _phase_score(archive[f"{prefix}_neighbors"], subsets, harmonics, channel)
    return positive, negative


def _calibration_source(method: str) -> str:
    return "phase_randomized_target_coefficients" if method.startswith("F2_") else "neighboring_frequency_bins"


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "03_public_focused"; total = 25 * 15; context.start_stage(stage, total)
    outputs = []; completed = 0; channel_rows = []
    for participant in range(25):
        qualities = _qualities(context, participant)
        for outer in range(15):
            checkpoint = context.checkpoint_dir / stage / f"participant_{participant:02d}_condition_{outer:02d}.csv"
            channel_checkpoint = checkpoint.with_name(checkpoint.stem + "_channels.csv")
            if not context.checkpoint_valid(checkpoint) or force:
                training_conditions = [condition for condition in range(15) if condition != outer]
                calibration: dict[tuple[str, str, int], list[np.ndarray]] = defaultdict(list)
                inner_channels: dict[tuple[str, int], int] = {}
                for inner in training_conditions:
                    inner_train = [condition for condition in training_conditions if condition != inner]
                    for method in METHODS:
                        inner_channels[(method, inner)] = select_train_only_channel(qualities[method], inner_train)
                    for stream in _condition_streams(inner):
                        with np.load(_archive_path(context, participant, stream)) as archive:
                            for method in METHODS:
                                for spatial in context.config.spatial_methods:
                                    channel = None if spatial == "S0" else inner_channels[(method, inner)]
                                    for k in context.config.k_values:
                                        subsets = deterministic_subsets(10, k, context.config.subset_cap, context.config.seed)
                                        seed = np.random.SeedSequence([context.config.seed, participant, outer, inner, stream,
                                                                       list(METHODS).index(method), k])
                                        _, negative = _scores(archive, method, subsets, channel, seed)
                                        calibration[(method, spatial, k)].append(negative.reshape(-1))
                final_channels = {method: select_train_only_channel(qualities[method], training_conditions) for method in METHODS}
                local_channel_rows = [{"participant": participant, "outer_condition": outer, "method": method,
                    "spatial": "S1", "selected_channel_zero_based": channel, "fit_conditions": 14,
                    "heldout_condition_excluded": True} for method, channel in final_channels.items()]
                rows = []
                for stream in _condition_streams(outer):
                    with np.load(_archive_path(context, participant, stream)) as archive:
                        for method in METHODS:
                            for spatial in context.config.spatial_methods:
                                channel = None if spatial == "S0" else final_channels[method]
                                for k in context.config.k_values:
                                    subsets = deterministic_subsets(10, k, context.config.subset_cap, context.config.seed)
                                    training_null = np.concatenate(calibration[(method, spatial, k)])
                                    threshold = empirical_threshold(training_null, context.config.target_fpr)
                                    seed = np.random.SeedSequence([context.config.seed, participant, outer, stream,
                                                                   list(METHODS).index(method), k, 999])
                                    positive, held_null = _scores(archive, method, subsets, channel, seed)
                                    rows.append({"branch": "public_set_fdt", "participant": participant,
                                        "outer_condition": outer, "stream": stream - 1, "method": method,
                                        "spatial": spatial, "k": k, "threshold": threshold,
                                        "detection_rate": float(np.mean(positive > threshold)),
                                        "heldout_fpr": float(np.mean(held_null > threshold)),
                                        "auc": auc_from_scores(positive, held_null.reshape(-1)),
                                        "mean_score": float(np.mean(positive)),
                                        "calibration_source": _calibration_source(method),
                                        "inner_cross_fitted": True, "selected_channel_zero_based": channel})
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
                pd.DataFrame(local_channel_rows).to_csv(channel_checkpoint, index=False)
                context.mark_checkpoint(checkpoint, {"participant": participant, "outer_condition": outer})
                context.mark_checkpoint(channel_checkpoint, {"participant": participant, "outer_condition": outer})
            outputs.extend([checkpoint, channel_checkpoint]); completed += 1
            context.progress(stage, completed, total, f"participant {participant + 1}, condition {outer + 1}")
    score_paths = [path for path in outputs if path.name.endswith(".csv") and not path.name.endswith("_channels.csv")]
    channel_paths = [path for path in outputs if path.name.endswith("_channels.csv")]
    scores_path = context.table_dir / "public_focused_scores.csv"
    channels_path = context.table_dir / "public_s1_channels.csv"
    pd.concat((pd.read_csv(path) for path in score_paths), ignore_index=True).to_csv(scores_path, index=False)
    pd.concat((pd.read_csv(path) for path in channel_paths), ignore_index=True).to_csv(channels_path, index=False)
    report = {"source": "public_set_fdt", "methods": list(METHODS), "spatial_methods": list(context.config.spatial_methods),
              "multitaper_calibration": "18 neighboring bins with inner leave-one-condition-out S1 fitting",
              "hilbert_calibration": "deterministic phase randomization with inner leave-one-condition-out S1 fitting",
              "excluded_spatial_methods": ["RESS", "DSS", "CCA"], "cross_source_equivalence_claim": False}
    report_path = context.run_dir / "public_focused_screen.json"; atomic_write_json(report_path, report)
    outputs.extend([scores_path, channels_path, report_path]); context.finish_stage(stage, outputs, report); return outputs, report
