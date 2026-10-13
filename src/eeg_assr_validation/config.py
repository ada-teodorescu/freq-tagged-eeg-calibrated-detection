from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ASSRValidationConfig:
    config_path: Path
    project_root: Path
    dataset_root: Path
    output_root: Path
    export_root: Path
    dataset_id: str
    dataset_version: str
    group: str
    task: str
    seed: int
    workers: int
    sampling_rate: int
    frequencies_hz: tuple[float, ...]
    primary_frequency_hz: float
    trials_per_block: int
    standards_per_frequency: tuple[int, ...]
    k_values: tuple[int, ...]
    surrogates_per_block: int
    epoch_start_seconds: float
    epoch_end_seconds: float
    l_freq: float
    h_freq: float
    notch_hz: float
    average_reference: bool
    hilbert_order: int
    hilbert_half_bandwidth_hz: float
    artifact_peak_to_peak_uv: float
    minimum_clean_trials: int
    target_fpr: float
    fpr_acceptance_high: float
    minimum_participants: int
    bootstrap_iterations: int
    sign_flip_iterations: int
    minimum_progress_interval_seconds: float
    minimum_free_disk_gb: float

    def standard_limit(self, frequency_hz: float) -> int:
        try:
            return self.standards_per_frequency[self.frequencies_hz.index(float(frequency_hz))]
        except ValueError as error:
            raise KeyError(f"unconfigured frequency {frequency_hz}") from error


def load_config(path: str | Path = "assr_validation_config.toml") -> ASSRValidationConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = config_path.parent
    resolve = lambda value: (root / value).resolve()
    paths, dataset = raw["paths"], raw["dataset"]
    analysis, preprocessing = raw["analysis"], raw["preprocessing"]
    estimators, validation, runtime = raw["estimators"], raw["validation"], raw["runtime"]
    frequencies = tuple(float(value) for value in analysis["frequencies_hz"])
    limits = tuple(int(value) for value in analysis["standards_per_frequency"])
    k_values = tuple(int(value) for value in analysis["k_values"])
    trials = int(analysis["trials_per_block"])
    if frequencies != (40.0, 27.0):
        raise ValueError("ASSR frequencies must remain locked to primary 40 Hz and secondary 27 Hz")
    if len(limits) != len(frequencies) or limits != (170, 160):
        raise ValueError("standard limits must remain locked to 170 and 160")
    if any(limit % trials for limit in limits):
        raise ValueError("standard limits must form complete trial blocks")
    if k_values != tuple(range(2, trials + 1)):
        raise ValueError("k_values must contain every prefix from 2 through trials_per_block")
    if not 0 <= float(preprocessing["l_freq"]) < float(preprocessing["h_freq"]) < int(analysis["sampling_rate"]) / 2:
        raise ValueError("invalid preprocessing passband")
    if not 0 < float(validation["target_fpr"]) <= float(validation["fpr_acceptance_high"]) < 1:
        raise ValueError("invalid FPR limits")
    workers = int(analysis.get("workers", 1))
    if not 1 <= workers <= 4:
        raise ValueError("workers must lie in 1..4")
    start, end = float(analysis["epoch_start_seconds"]), float(analysis["epoch_end_seconds"])
    if not 0 <= start < end <= 0.5:
        raise ValueError("retained interval must lie within the 0.5-second stimulus")
    return ASSRValidationConfig(
        config_path=config_path, project_root=root,
        dataset_root=resolve(paths["dataset_root"]), output_root=resolve(paths["output_root"]),
        export_root=resolve(paths["export_root"]), dataset_id=str(dataset["dataset_id"]),
        dataset_version=str(dataset["dataset_version"]), group=str(dataset["group"]), task=str(dataset["task"]),
        seed=int(analysis["seed"]), workers=workers, sampling_rate=int(analysis["sampling_rate"]),
        frequencies_hz=frequencies, primary_frequency_hz=float(analysis["primary_frequency_hz"]),
        trials_per_block=trials, standards_per_frequency=limits, k_values=k_values,
        surrogates_per_block=int(analysis["surrogates_per_block"]),
        epoch_start_seconds=start, epoch_end_seconds=end,
        l_freq=float(preprocessing["l_freq"]), h_freq=float(preprocessing["h_freq"]),
        notch_hz=float(preprocessing["notch_hz"]), average_reference=bool(preprocessing["average_reference"]),
        hilbert_order=int(estimators["hilbert_order"]),
        hilbert_half_bandwidth_hz=float(estimators["hilbert_half_bandwidth_hz"]),
        artifact_peak_to_peak_uv=float(preprocessing["artifact_peak_to_peak_uv"]),
        minimum_clean_trials=int(preprocessing["minimum_clean_trials"]),
        target_fpr=float(validation["target_fpr"]),
        fpr_acceptance_high=float(validation["fpr_acceptance_high"]),
        minimum_participants=int(validation["minimum_participants"]),
        bootstrap_iterations=int(validation["bootstrap_iterations"]),
        sign_flip_iterations=int(validation["sign_flip_iterations"]),
        minimum_progress_interval_seconds=float(runtime["minimum_progress_interval_seconds"]),
        minimum_free_disk_gb=float(runtime["minimum_free_disk_gb"]),
    )
