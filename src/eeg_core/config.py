from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ArticleConfig:
    config_path: Path
    project_root: Path
    data_root: Path
    replication_results: Path
    detection_results: Path
    improvement_results: Path
    raw_replication_run: Path
    output_root: Path
    export_root: Path
    manuscript_root: Path
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
    neighbor_bins_per_harmonic: int
    target_fpr: float
    target_detection: float
    algorithms: tuple[str, ...]
    c_grid: tuple[float, ...]
    inner_group_folds: int
    maximum_iterations: int
    tolerance: float
    calibration_null_cap_per_participant: int
    bootstrap_iterations: int
    sign_flip_iterations: int
    fpr_acceptance_low: float
    fpr_acceptance_high: float
    minimum_progress_interval_seconds: float
    minimum_free_disk_gb: float


def load_config(path: str | Path = "article_config.toml") -> ArticleConfig:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    root = config_path.parent
    paths = raw["paths"]
    analysis = raw["analysis"]
    ml = raw["machine_learning"]
    statistics = raw["statistics"]
    runtime = raw["runtime"]
    workers = int(analysis.get("workers", 1))
    if not 1 <= workers <= 4:
        raise ValueError("analysis.workers must be in 1..4")
    trials = int(analysis["trials_per_stream"])
    k_values = tuple(int(value) for value in analysis["k_values"])
    if not k_values or min(k_values) < 2 or max(k_values) > trials:
        raise ValueError("analysis.k_values must lie in 2..trials_per_stream")
    durations = tuple(float(value) for value in analysis["durations_seconds"])
    usable_seconds = (int(analysis["fft_pad_samples"]) / int(analysis["sampling_rate"]))
    if not durations or min(durations) <= 0 or max(durations) > usable_seconds:
        raise ValueError("analysis.durations_seconds exceed the usable epoch")
    harmonics = tuple(float(value) for value in analysis["harmonics_hz"])
    if len(harmonics) != 4:
        raise ValueError("exactly four harmonic features are required")
    algorithms = tuple(str(value) for value in analysis["algorithms"])
    expected = ("A1_ITPC", "A2_Hotelling", "A3_JointHarmonic", "A4_LearnedFusion")
    if algorithms != expected:
        raise ValueError(f"analysis.algorithms must be {expected}")
    folds = int(ml["inner_group_folds"])
    if folds < 2 or folds >= 25:
        raise ValueError("machine_learning.inner_group_folds must lie in 2..24")
    calibration_cap = int(ml["calibration_null_cap_per_participant"])
    if calibration_cap <= 0:
        raise ValueError("machine_learning.calibration_null_cap_per_participant must be positive")
    low, high = float(statistics["fpr_acceptance_low"]), float(statistics["fpr_acceptance_high"])
    target_fpr = float(analysis["target_fpr"])
    if not 0 < low <= target_fpr <= high < 1:
        raise ValueError("invalid FPR acceptance interval")

    def resolve(name: str) -> Path:
        return (root / paths[name]).resolve()

    return ArticleConfig(
        config_path=config_path,
        project_root=root,
        data_root=resolve("data_root"),
        replication_results=resolve("replication_results"),
        detection_results=resolve("detection_results"),
        improvement_results=resolve("improvement_results"),
        raw_replication_run=resolve("raw_replication_run"),
        output_root=resolve("output_root"),
        export_root=resolve("export_root"),
        manuscript_root=resolve("manuscript_root"),
        seed=int(analysis["seed"]), workers=workers,
        sampling_rate=int(analysis["sampling_rate"]),
        onset_crop_samples=int(analysis["onset_crop_samples"]),
        fft_pad_samples=int(analysis["fft_pad_samples"]),
        trials_per_stream=trials,
        streams=int(analysis["streams"]), conditions=int(analysis["conditions"]),
        channels=int(analysis["channels"]), harmonics_hz=harmonics,
        durations_seconds=durations, k_values=k_values,
        subset_cap=int(analysis["subset_cap"]),
        neighbor_bins_per_harmonic=int(analysis["neighbor_bins_per_harmonic"]),
        target_fpr=target_fpr, target_detection=float(analysis["target_detection"]),
        algorithms=algorithms, c_grid=tuple(float(value) for value in ml["c_grid"]),
        inner_group_folds=folds, maximum_iterations=int(ml["maximum_iterations"]),
        tolerance=float(ml["tolerance"]),
        calibration_null_cap_per_participant=calibration_cap,
        bootstrap_iterations=int(statistics["bootstrap_iterations"]),
        sign_flip_iterations=int(statistics["sign_flip_iterations"]),
        fpr_acceptance_low=low, fpr_acceptance_high=high,
        minimum_progress_interval_seconds=float(runtime["minimum_progress_interval_seconds"]),
        minimum_free_disk_gb=float(runtime["minimum_free_disk_gb"]),
    )
