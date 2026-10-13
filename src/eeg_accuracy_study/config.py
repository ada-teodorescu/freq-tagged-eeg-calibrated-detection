from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class AccuracyConfig:
    config_path: Path
    project_root: Path
    data_root: Path
    baseline_algorithm_run: Path
    output_root: Path
    export_root: Path
    seed: int
    workers: int
    sampling_rate: int
    onset_crop_samples: int
    trials_per_stream: int
    streams_per_condition: int
    conditions: int
    channels: int
    harmonics_hz: tuple[float, ...]
    primary_duration_seconds: float
    target_fpr: float
    target_sensitivity: float
    operating_fprs: tuple[float, ...]
    surrogate_seeds: tuple[int, ...]
    surrogates_per_seed: int
    reliability_harmonic_indices: tuple[int, ...]
    reliability_lower_quantile: float
    reliability_upper_quantile: float
    regions: tuple[tuple[str, ...], ...]
    region_names: tuple[str, ...]
    c_grid: tuple[float, ...]
    inner_group_folds: int
    maximum_iterations: int
    tolerance: float
    training_nulls_per_condition: int
    onset_window_seconds: float
    onset_starts_seconds: tuple[float, ...]
    onset_surrogates: int
    bootstrap_iterations: int
    sign_flip_iterations: int
    fpr_acceptance_low: float
    fpr_acceptance_high: float
    minimum_progress_interval_seconds: float
    minimum_free_disk_gb: float

    @property
    def participants_expected(self) -> int:
        return 25

    @property
    def streams(self) -> int:
        return self.conditions * self.streams_per_condition

    @property
    def primary_surrogates(self) -> int:
        return len(self.surrogate_seeds) * self.surrogates_per_seed


def load_config(path: str | Path = "accuracy_study_config.toml") -> AccuracyConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = config_path.parent
    paths = raw["paths"]
    analysis = raw["analysis"]
    reliability = raw["reliability"]
    spatial = raw["spatial"]
    machine = raw["machine_learning"]
    onset = raw["onset_control"]
    statistics = raw["statistics"]
    runtime = raw["runtime"]

    resolve = lambda key: (root / paths[key]).resolve()
    harmonics = tuple(float(value) for value in analysis["harmonics_hz"])
    operating = tuple(float(value) for value in analysis["operating_fprs"])
    target_fpr = float(analysis["target_fpr"])
    if target_fpr not in operating or tuple(sorted(set(operating))) != operating:
        raise ValueError("operating_fprs must be unique, sorted, and contain target_fpr")
    if not all(0 < value < 0.5 for value in operating):
        raise ValueError("operating FPR values must lie in (0, 0.5)")
    regions = tuple(tuple(str(channel) for channel in group) for group in spatial["regions"])
    region_names = tuple(str(value) for value in spatial["region_names"])
    flattened = [channel for group in regions for channel in group]
    channels = int(analysis["channels"])
    if len(regions) != len(region_names) or len(flattened) != channels or len(set(flattened)) != channels:
        raise ValueError("spatial regions must partition all configured channels exactly once")
    starts = tuple(float(value) for value in onset["starts_seconds"])
    window = float(onset["window_seconds"])
    duration = float(analysis["primary_duration_seconds"])
    if any(value < 0 or value + window > duration + 1e-12 for value in starts):
        raise ValueError("onset-control windows exceed the primary duration")
    lower, upper = float(reliability["lower_quantile"]), float(reliability["upper_quantile"])
    if not 0 <= lower < upper <= 1:
        raise ValueError("invalid reliability clipping quantiles")
    workers = int(analysis.get("workers", 1))
    if not 1 <= workers <= 4:
        raise ValueError("workers must lie in 1..4")
    low, high = float(statistics["fpr_acceptance_low"]), float(statistics["fpr_acceptance_high"])
    if not 0 < low <= target_fpr <= high < 1:
        raise ValueError("invalid FPR acceptance interval")
    seeds = tuple(int(value) for value in analysis["surrogate_seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("surrogate seeds must be non-empty and unique")
    return AccuracyConfig(
        config_path=config_path, project_root=root, data_root=resolve("data_root"),
        baseline_algorithm_run=resolve("baseline_algorithm_run"), output_root=resolve("output_root"),
        export_root=resolve("export_root"), seed=int(analysis["seed"]), workers=workers,
        sampling_rate=int(analysis["sampling_rate"]), onset_crop_samples=int(analysis["onset_crop_samples"]),
        trials_per_stream=int(analysis["trials_per_stream"]),
        streams_per_condition=int(analysis["streams_per_condition"]), conditions=int(analysis["conditions"]),
        channels=channels, harmonics_hz=harmonics, primary_duration_seconds=duration,
        target_fpr=target_fpr, target_sensitivity=float(analysis["target_sensitivity"]),
        operating_fprs=operating, surrogate_seeds=seeds,
        surrogates_per_seed=int(analysis["surrogates_per_seed"]),
        reliability_harmonic_indices=tuple(int(value) for value in reliability["harmonic_indices"]),
        reliability_lower_quantile=lower, reliability_upper_quantile=upper,
        regions=regions, region_names=region_names, c_grid=tuple(float(value) for value in machine["c_grid"]),
        inner_group_folds=int(machine["inner_group_folds"]), maximum_iterations=int(machine["maximum_iterations"]),
        tolerance=float(machine["tolerance"]),
        training_nulls_per_condition=int(machine["training_nulls_per_condition"]),
        onset_window_seconds=window, onset_starts_seconds=starts, onset_surrogates=int(onset["surrogates"]),
        bootstrap_iterations=int(statistics["bootstrap_iterations"]),
        sign_flip_iterations=int(statistics["sign_flip_iterations"]), fpr_acceptance_low=low,
        fpr_acceptance_high=high,
        minimum_progress_interval_seconds=float(runtime["minimum_progress_interval_seconds"]),
        minimum_free_disk_gb=float(runtime["minimum_free_disk_gb"]),
    )
