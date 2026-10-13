from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import signal

from eeg_assr_validation.algorithms import (heldout_sequential, heldout_static,
    hilbert_interval_coefficients, phase_rotations, score_trajectories, sinusoidal_ls_coefficients)
from eeg_assr_validation.cli import _descendants
from eeg_assr_validation.config import load_config
from eeg_assr_validation.runtime import RunContext, atomic_write_text
from eeg_assr_validation.stages import stage_registry
from eeg_assr_validation.stages.inventory import select_standard_events
from eeg_assr_validation.stages.statistics import primary_gate
from eeg_assr_validation.stages.trajectories import build_rotations
from eeg_validation_extension.algorithms import kernel_metrics


ROOT = Path(__file__).resolve().parents[2]


def test_locked_config_and_stage_graph() -> None:
    config = load_config(ROOT / "assr_validation_config.toml")
    assert config.frequencies_hz == (40.0, 27.0)
    assert config.standards_per_frequency == (170, 160)
    assert config.k_values == tuple(range(2, 11))
    stages = stage_registry()
    assert tuple(stages) == ("00_freeze", "01_inventory", "02_coefficients", "03_null_trajectories",
                             "04_static_validation", "05_sequential_validation",
                             "06_sensitivity_statistics", "07_report")
    assert _descendants("03_null_trajectories", stages) == {
        "03_null_trajectories", "04_static_validation", "05_sequential_validation",
        "06_sensitivity_statistics", "07_report",
    }


def test_checkpoint_validation_corruption_and_stage_invalidation(tmp_path: Path) -> None:
    config = replace(load_config(ROOT / "assr_validation_config.toml"),
                     output_root=tmp_path / "outputs", export_root=tmp_path / "exports")
    context = RunContext.create(config, 1)
    checkpoint = context.checkpoint_dir / "stage" / "P1.txt"
    atomic_write_text(checkpoint, "complete")
    context.mark_checkpoint(checkpoint, {"participant": "P1"})
    assert context.checkpoint_valid(checkpoint)
    checkpoint.with_suffix(".txt.meta.json").write_text("not-json", encoding="utf-8")
    assert not context.checkpoint_valid(checkpoint)
    context.update_stage("04_static_validation", status="completed")
    context.invalidate({"04_static_validation"}, "test")
    assert context.read_state()["stages"]["04_static_validation"]["status"] == "stale"


def test_primary_gate_is_strict_and_checks_all_three_rules() -> None:
    config = load_config(ROOT / "assr_validation_config.toml")
    assert primary_gate(0.001, 0.055, 35, config)
    assert not primary_gate(0.0, 0.05, 37, config)
    assert not primary_gate(0.01, 0.056, 37, config)
    assert not primary_gate(0.01, 0.05, 34, config)


@pytest.mark.parametrize("frequency", [27.0, 40.0])
def test_exact_frequency_ls_recovers_phase(frequency: float) -> None:
    sfreq, start, samples = 512, 26, 204
    phase = 0.63
    time = (start + np.arange(samples)) / sfreq
    values = 3.0 + 2.5 * np.cos(2 * np.pi * frequency * time + phase)
    rectangular = sinusoidal_ls_coefficients(values, sfreq, frequency, start_sample=start)
    hann = sinusoidal_ls_coefficients(values, sfreq, frequency, start_sample=start, hann=True)
    assert np.angle(rectangular) == pytest.approx(phase, abs=1e-10)
    assert np.angle(hann) == pytest.approx(phase, abs=1e-10)
    assert abs(rectangular) == pytest.approx(2.5, rel=1e-10)
    assert abs(hann) == pytest.approx(2.5, rel=1e-10)


def test_hilbert_interval_uses_exact_absolute_samples() -> None:
    sfreq, frequency, start, samples = 512, 27.0, 26, 204
    phase = -0.42
    time = (start + np.arange(samples)) / sfreq
    analytic = 1.7 * np.exp(1j * (2 * np.pi * frequency * time + phase))
    coefficient = hilbert_interval_coefficients(analytic, sfreq, frequency, start)
    assert abs(coefficient) == pytest.approx(1.7, rel=1e-12)
    assert np.angle(coefficient) == pytest.approx(phase, abs=1e-12)


def test_standard_selection_excludes_nontargets_and_caps_chronologically() -> None:
    events = pd.DataFrame({
        "onset": [3.0, 1.0, 2.0, 4.0, 5.0, 6.0],
        "sample": [30, 10, 20, 40, 50, 60],
        "trial_type": ["40_Hz_Standard", "Response_button", "40_Hz_Standard",
                       "40_Hz_Oddball", "40_Hz_Standard", "40_Hz_Standard"],
    })
    selected = select_standard_events(events, 40.0, 4, 2)
    assert selected["sample"].tolist() == [20, 30, 50, 60]
    assert selected["block"].tolist() == [1, 1, 2, 2]
    assert selected["trial_in_block"].tolist() == [1, 2, 1, 2]


def test_shared_rotations_are_deterministic_and_shared_over_estimators() -> None:
    config = load_config(ROOT / "assr_validation_config.toml")
    first = build_rotations(config, 2, 0, 3)
    second = build_rotations(config, 2, 0, 3)
    other = build_rotations(config, 3, 0, 3)
    assert first.shape == (3, config.surrogates_per_block, config.trials_per_block)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_phase_score_separates_locked_signal_from_rotation_null() -> None:
    rng = np.random.default_rng(4)
    phases = rng.normal(0, 0.04, size=(5, 10, 32))
    coefficients = np.exp(1j * phases)
    rotations = np.stack([phase_rotations(500, 10, seed) for seed in range(5)])
    positive = score_trajectories(coefficients, (2, 5, 10))
    null = score_trajectories(coefficients, (2, 5, 10), rotations)
    assert positive[:, -1].mean() > 19.0
    assert 1.6 < null[:, :, -1].mean() < 2.4


def test_static_calibration_excludes_held_participant() -> None:
    participants = ("P1", "P2", "P3")
    positive = np.ones((3, 2))
    null = np.zeros((3, 2, 100))
    null[0] = 100.0
    rows = pd.DataFrame(heldout_static(positive, null, participants, 0.05))
    assert rows.loc[rows["participant"] == "P1", "threshold"].iloc[0] == 0.0
    assert rows.loc[rows["participant"] == "P2", "threshold"].iloc[0] == 100.0


def test_sequential_calibration_controls_synthetic_familywise_fpr() -> None:
    rng = np.random.default_rng(22)
    participants = tuple(f"P{index}" for index in range(5))
    null = rng.normal(size=(5, 8, 300, 9)).astype(np.float32)
    positive = null[:, :, 0, :].copy()
    _, summaries = heldout_sequential(positive, null, participants, tuple(range(2, 11)), 0.05)
    mean_fpr = np.mean([row["familywise_fpr"] for row in summaries])
    assert mean_fpr <= 0.075


def test_synthetic_estimator_to_detection_flow() -> None:
    rng = np.random.default_rng(7)
    participants = tuple(f"P{index}" for index in range(4))
    positives, nulls = [], []
    for participant in range(4):
        phase = rng.normal(0, 0.08, size=(6, 10, 16))
        coefficients = np.exp(1j * phase)
        rotations = np.stack([phase_rotations(250, 10, 1000 + participant * 10 + block) for block in range(6)])
        positives.append(score_trajectories(coefficients, tuple(range(2, 11))))
        nulls.append(score_trajectories(coefficients, tuple(range(2, 11)), rotations))
    rows = pd.DataFrame(heldout_static(np.stack(positives)[..., -1], np.stack(nulls)[..., -1], participants, 0.05))
    assert rows.groupby("participant")["detected"].mean().min() == 1.0
    _, summaries = heldout_sequential(np.stack(positives), np.stack(nulls), participants, tuple(range(2, 11)), 0.05)
    assert all(row["detection"] == 1.0 for row in summaries)


def test_kernel_characterization_finds_dominant_negative_lobe() -> None:
    sfreq, frequency, samples = 512, 4.0, 2048
    time = np.arange(samples) / sfreq
    kernel = np.exp(-2j * np.pi * frequency * time)
    metrics = kernel_metrics(kernel, sfreq, frequency, 100, samples - 100)
    assert metrics["peak_frequency_hz"] == pytest.approx(-frequency, abs=0.01)
    assert metrics["bandwidth_3db_hz"] > 0
    assert metrics["equivalent_noise_bandwidth_hz"] > 0


def test_optional_real_bdf_header_contract() -> None:
    bdfs = sorted((ROOT / "data" / "external" / "ds006780").glob("sub-*/eeg/*task-ASSR*_eeg.bdf"))
    bdfs = [path for path in bdfs if path.stat().st_size > 1024 * 1024]
    if not bdfs:
        pytest.skip("downloaded ds006780 ASSR payload unavailable")
    import mne
    raw = mne.io.read_raw_bdf(bdfs[0], preload=False, verbose="ERROR")
    assert raw.info["sfreq"] == 512
    assert len(raw.ch_names) == 73
