from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class StudyConfig:
    config_path: Path
    project_root: Path
    data_root: Path
    baseline_article_run: Path
    output_root: Path
    export_root: Path
    seed: int
    workers: int
    sampling_rate: int
    onset_crop_samples: int
    fft_pad_samples: int
    trials_per_stream: int
    streams: int
    conditions: int
    channels: int
    harmonics_hz: tuple[float, ...]
    durations_seconds: tuple[float, ...]
    k_values: tuple[int, ...]
    subset_cap: int
    surrogates_per_subset: int
    target_fpr: float
    target_detection: float
    c_grid: tuple[float, ...]
    inner_group_folds: int
    maximum_iterations: int
    tolerance: float
    calibration_null_cap_per_participant: int
    harmonic_subsets: tuple[tuple[int, ...], ...]
    bootstrap_iterations: int
    sign_flip_iterations: int
    fpr_acceptance_low: float
    fpr_acceptance_high: float
    minimum_progress_interval_seconds: float
    minimum_free_disk_gb: float


def load_config(path: str | Path = "algorithm_study_config.toml") -> StudyConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle: raw = tomllib.load(handle)
    root = config_path.parent; paths = raw["paths"]; analysis = raw["analysis"]
    ml, statistics, runtime = raw["machine_learning"], raw["statistics"], raw["runtime"]
    workers = int(analysis.get("workers", 1))
    if not 1 <= workers <= 4: raise ValueError("analysis.workers must be in 1..4")
    harmonics = tuple(float(value) for value in analysis["harmonics_hz"])
    subsets = tuple(tuple(int(index) for index in subset) for subset in ml["harmonic_subsets"])
    if not subsets or any(not subset or tuple(sorted(set(subset))) != subset or max(subset) >= len(harmonics) for subset in subsets):
        raise ValueError("machine_learning.harmonic_subsets are invalid")
    k_values = tuple(int(value) for value in analysis["k_values"])
    trials = int(analysis["trials_per_stream"])
    if min(k_values) < 2 or max(k_values) > trials: raise ValueError("invalid K values")
    durations = tuple(float(value) for value in analysis["durations_seconds"])
    if min(durations) <= 0 or max(durations) * int(analysis["sampling_rate"]) > int(analysis["fft_pad_samples"]):
        raise ValueError("invalid durations")
    low, high = float(statistics["fpr_acceptance_low"]), float(statistics["fpr_acceptance_high"])
    target_fpr = float(analysis["target_fpr"])
    if not 0 < low <= target_fpr <= high < 1: raise ValueError("invalid FPR interval")
    resolve = lambda name: (root / paths[name]).resolve()
    return StudyConfig(
        config_path=config_path, project_root=root, data_root=resolve("data_root"),
        baseline_article_run=resolve("baseline_article_run"), output_root=resolve("output_root"),
        export_root=resolve("export_root"), seed=int(analysis["seed"]), workers=workers,
        sampling_rate=int(analysis["sampling_rate"]), onset_crop_samples=int(analysis["onset_crop_samples"]),
        fft_pad_samples=int(analysis["fft_pad_samples"]), trials_per_stream=trials,
        streams=int(analysis["streams"]), conditions=int(analysis["conditions"]), channels=int(analysis["channels"]),
        harmonics_hz=harmonics, durations_seconds=durations, k_values=k_values,
        subset_cap=int(analysis["subset_cap"]), surrogates_per_subset=int(analysis["surrogates_per_subset"]),
        target_fpr=target_fpr, target_detection=float(analysis["target_detection"]),
        c_grid=tuple(float(value) for value in ml["c_grid"]), inner_group_folds=int(ml["inner_group_folds"]),
        maximum_iterations=int(ml["maximum_iterations"]), tolerance=float(ml["tolerance"]),
        calibration_null_cap_per_participant=int(ml["calibration_null_cap_per_participant"]),
        harmonic_subsets=subsets, bootstrap_iterations=int(statistics["bootstrap_iterations"]),
        sign_flip_iterations=int(statistics["sign_flip_iterations"]), fpr_acceptance_low=low,
        fpr_acceptance_high=high, minimum_progress_interval_seconds=float(runtime["minimum_progress_interval_seconds"]),
        minimum_free_disk_gb=float(runtime["minimum_free_disk_gb"]),
    )
