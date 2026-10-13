from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ConfirmatoryConfig:
    config_path: Path
    project_root: Path
    data_root: Path
    baseline_advanced_run: Path
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
    k_values: tuple[int, ...]
    primary_duration_seconds: float
    matched_window_start_seconds: float
    matched_window_end_seconds: float
    full_window_start_seconds: float
    full_window_end_seconds: float
    hilbert_order: int
    hilbert_half_bandwidth_hz: float
    hilbert_edge_crop_seconds: float
    target_fpr: float
    fpr_acceptance_low: float
    fpr_acceptance_high: float
    surrogate_seeds: tuple[int, ...]
    surrogates_per_seed: int
    c_grid: tuple[float, ...]
    locked_c: float
    maximum_iterations: int
    tolerance: float
    trca_regularization: tuple[float, ...]
    regions: tuple[tuple[str, ...], ...]
    region_names: tuple[str, ...]
    bootstrap_iterations: int
    sign_flip_iterations: int
    minimum_progress_interval_seconds: float
    minimum_free_disk_gb: float

    @property
    def streams(self) -> int:
        return self.conditions * self.streams_per_condition

    @property
    def participants_expected(self) -> int:
        return 25

    @property
    def surrogates(self) -> int:
        return len(self.surrogate_seeds) * self.surrogates_per_seed


def load_config(path: str | Path = "confirmatory_study_config.toml") -> ConfirmatoryConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = config_path.parent
    paths, analysis = raw["paths"], raw["analysis"]
    windows, hilbert = raw["windows"], raw["hilbert"]
    calibration, machine, trca = raw["calibration"], raw["machine_learning"], raw["trca"]
    spatial, statistics, runtime = raw["spatial"], raw["statistics"], raw["runtime"]
    resolve = lambda value: (root / value).resolve()

    harmonics = tuple(float(value) for value in analysis["harmonics_hz"])
    if harmonics != (4.0, 8.0, 12.0, 16.0):
        raise ValueError("confirmatory study requires the locked 4, 8, 12, and 16 Hz harmonics")
    trials = int(analysis["trials_per_stream"])
    k_values = tuple(int(value) for value in analysis["k_values"])
    if k_values != tuple(range(2, trials + 1)):
        raise ValueError("sequential K values must be every prefix from 2 through trials_per_stream")

    matched_start, matched_end = float(windows["matched_window_start_seconds"]), float(windows["matched_window_end_seconds"])
    full_start, full_end = float(windows["full_window_start_seconds"]), float(windows["full_window_end_seconds"])
    if not (0 <= matched_start < matched_end <= full_end) or full_start != matched_start:
        raise ValueError("matched and full windows must share a start and matched must not exceed full")
    if round(matched_end - matched_start, 6) != 4.0:
        raise ValueError("the matched fair-comparison window must be exactly 4.0 seconds, per the locked design")

    seeds = tuple(int(value) for value in analysis["surrogate_seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("surrogate seeds must be unique and non-empty")

    target_fpr = float(calibration["target_fpr"])
    low, high = float(calibration["fpr_acceptance_low"]), float(calibration["fpr_acceptance_high"])
    if not 0 < low <= target_fpr <= high < 1:
        raise ValueError("invalid FPR target or acceptance interval")

    c_grid = tuple(float(value) for value in machine["c_grid"])
    locked_c = float(machine["locked_c"])
    if locked_c not in c_grid:
        raise ValueError("locked_c must be a member of c_grid")

    regions = tuple(tuple(str(channel) for channel in group) for group in spatial["regions"])
    names = tuple(str(value) for value in spatial["region_names"])
    flattened = [channel for region in regions for channel in region]
    channels = int(analysis["channels"])
    if len(regions) != len(names) or len(flattened) != channels or len(set(flattened)) != channels:
        raise ValueError("regions must partition all configured channels")

    workers = int(analysis.get("workers", 1))
    if not 1 <= workers <= 4:
        raise ValueError("workers must lie in 1..4")

    return ConfirmatoryConfig(
        config_path=config_path, project_root=root, data_root=resolve(paths["data_root"]),
        baseline_advanced_run=resolve(paths["baseline_advanced_run"]),
        output_root=resolve(paths["output_root"]), export_root=resolve(paths["export_root"]),
        seed=int(analysis["seed"]), workers=workers, sampling_rate=int(analysis["sampling_rate"]),
        onset_crop_samples=int(analysis["onset_crop_samples"]), trials_per_stream=trials,
        streams_per_condition=int(analysis["streams_per_condition"]), conditions=int(analysis["conditions"]),
        channels=channels, harmonics_hz=harmonics, k_values=k_values,
        primary_duration_seconds=float(analysis["primary_duration_seconds"]),
        matched_window_start_seconds=matched_start, matched_window_end_seconds=matched_end,
        full_window_start_seconds=full_start, full_window_end_seconds=full_end,
        hilbert_order=int(hilbert["order"]), hilbert_half_bandwidth_hz=float(hilbert["half_bandwidth_hz"]),
        hilbert_edge_crop_seconds=float(hilbert["edge_crop_seconds"]),
        target_fpr=target_fpr, fpr_acceptance_low=low, fpr_acceptance_high=high,
        surrogate_seeds=seeds, surrogates_per_seed=int(analysis["surrogates_per_seed"]),
        c_grid=c_grid, locked_c=locked_c,
        maximum_iterations=int(machine["maximum_iterations"]), tolerance=float(machine["tolerance"]),
        trca_regularization=tuple(float(value) for value in trca["regularization_grid"]),
        regions=regions, region_names=names,
        bootstrap_iterations=int(statistics["bootstrap_iterations"]),
        sign_flip_iterations=int(statistics["sign_flip_iterations"]),
        minimum_progress_interval_seconds=float(runtime["minimum_progress_interval_seconds"]),
        minimum_free_disk_gb=float(runtime["minimum_free_disk_gb"]),
    )
