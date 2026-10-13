from pathlib import Path

import numpy as np
import pytest

from cucu_external_validation.config import load_config
from cucu_external_validation.dataset import greedy_nonoverlapping, group_condition_units, rest_windows
from cucu_external_validation.stages import stage_registry
from cucu_external_validation.stages.preprocess import eligibility
from cucu_external_validation.stages.features import _prepare_raw
from cucu_external_validation.stages.scores import condition_trajectories
from cucu_external_validation.statistics import participant_cluster_interval


def test_locked_config_and_stage_graph():
    config = load_config(Path(__file__).parents[2] / "external_validation_config.toml")
    assert config.target_channels[0] == "Fp1"
    assert len(config.target_channels) == 32
    assert config.structural_interpolation_targets == ("TP9", "TP10", "PO9", "PO10")
    assert tuple(stage_registry()) == (
        "00_freeze", "01_inventory", "02_preprocess", "03_negative_units", "04_hilbert_features",
        "05_deployment_model", "06_external_scores", "07_statistics", "08_report",
    )


def test_greedy_windows_are_deterministic_nonoverlapping_and_in_bounds():
    onsets = np.arange(0, 200, 1.25)
    first = greedy_nonoverlapping(onsets, duration=5.0, recording_duration=200.0, count=30)
    second = greedy_nonoverlapping(onsets[::-1], duration=5.0, recording_duration=200.0, count=30)
    np.testing.assert_array_equal(first, second)
    assert np.all(np.diff(first) >= 5.0)
    assert first[-1] + 5.0 <= 200.0


def test_greedy_windows_fail_when_recording_is_short():
    with pytest.raises(ValueError, match="needed 30"):
        greedy_nonoverlapping(np.arange(10.0), 5.0, 20.0, 30)


def test_condition_grouping_preserves_natural_triplets():
    grouped = group_condition_units(np.arange(90.0), streams=3, trials=10)
    assert grouped.shape == (3, 3, 10)
    np.testing.assert_array_equal(grouped[0, :, 0], [0, 1, 2])
    np.testing.assert_array_equal(grouped[0, 0], np.arange(0, 30, 3))
    assert np.all(np.diff(grouped, axis=2) > 0)


def test_rest_windows_respect_both_edges():
    starts = rest_windows(240.0, 5.0, 30)
    assert starts[0] == 5.0
    assert starts[-1] + 5.0 <= 235.0
    with pytest.raises(ValueError):
        rest_windows(150.0, 5.0, 30)


def test_channel_eligibility_allows_only_declared_structural_missing():
    targets = ("Fp1", "Fz", "TP9", "PO9")
    ok, reason, bad = eligibility(("Fp1", "Fz"), (), targets, ("TP9", "PO9"), 1)
    assert ok and reason == "eligible" and bad == []
    ok, reason, missing = eligibility(("Fp1",), (), targets, ("TP9", "PO9"), 1)
    assert not ok and reason == "undeclared missing target channels" and missing == ["Fz"]
    ok, reason, bad = eligibility(("Fp1", "Fz"), ("Fp1", "Fz"), targets, ("TP9", "PO9"), 1)
    assert not ok and reason == "too many dataset-marked bad target channels"


def test_condition_trajectories_contract_and_phase_locking():
    trials, channels, harmonics = 10, 4, 4
    locked = np.ones((1, 3, trials, channels, harmonics), dtype=np.complex64)
    regions = (np.asarray([0, 1]), np.asarray([2, 3]))
    trajectories = condition_trajectories(locked, tuple(range(2, 11)), regions)
    assert trajectories.shape == (1, 9, 8)
    for look, k in enumerate(range(2, 11)):
        np.testing.assert_allclose(trajectories[0, look], 2.0 * k)


def test_participant_cluster_interval_is_deterministic_and_clusters_rates():
    rates = np.asarray([0.0, 0.0, 0.1, 0.1, 0.2])
    first = participant_cluster_interval(rates, 1000, 19)
    second = participant_cluster_interval(rates, 1000, 19)
    assert first == second
    assert first["point_estimate"] == pytest.approx(0.08)
    assert first["ci95_low"] <= first["point_estimate"] <= first["ci95_high"]
    assert first["one_sided_upper"] <= first["ci95_high"]


def test_external_feature_preparation_excludes_coordinate_less_auxiliaries():
    config = load_config(Path(__file__).parents[2] / "external_validation_config.toml")
    header = (config.external_data_root / "sub-01" / "eeg" / "sub-01_task-rest_acq-ec_eeg.vhdr")
    if not header.exists() or header.stat().st_size < 1024:
        pytest.skip("external BrainVision payload unavailable")
    raw, structural, interpolated = _prepare_raw(header, config, ())
    assert tuple(raw.ch_names) == config.target_channels
    assert tuple(structural) == config.structural_interpolation_targets
    assert set(interpolated) == set(config.structural_interpolation_targets)
    assert np.isfinite(raw.get_data(start=0, stop=10)).all()
