from __future__ import annotations

import numpy as np

from eeg_accuracy_study.features import (
    amplitude_reliability_weights,
    condition_feature_bundle,
    exact_harmonic_coefficients,
    matched_condition_observations,
    region_indices,
)


def test_exact_harmonic_coefficient_selects_known_frequency() -> None:
    sampling_rate = 100
    time = np.arange(200) / sampling_rate
    trials = np.tile(np.sin(2 * np.pi * 4 * time), (3, 2, 1))
    coefficients = exact_harmonic_coefficients(trials, sampling_rate, (4.0, 8.0), 0.0, 2.0)
    assert coefficients.shape == (3, 2, 2)
    assert np.mean(np.abs(coefficients[..., 0])) > 100 * np.mean(np.abs(coefficients[..., 1]))


def test_reliability_weights_depend_only_on_amplitude() -> None:
    rng = np.random.default_rng(4)
    coefficients = rng.normal(size=(10, 4, 3)) + 1j * rng.normal(size=(10, 4, 3))
    rotations = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(10, 1, 3)))
    first = amplitude_reliability_weights(coefficients)
    second = amplitude_reliability_weights(coefficients * rotations)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first.mean(), 1.0)


def test_condition_bundle_preserves_locking_and_shapes() -> None:
    coefficients = np.ones((6, 10, 4, 4), dtype=np.complex64)
    regions = (np.asarray([0, 1]), np.asarray([2, 3]))
    bundle = condition_feature_bundle(coefficients, regions, (10, 11), 40, seed_salt=3)
    assert bundle["positive_global"].shape == (2, 4)
    assert bundle["null_global"].shape == (2, 80, 4)
    assert bundle["positive_regional"].shape == (2, 2, 4)
    assert bundle["null_regional"].shape == (2, 80, 2, 4)
    np.testing.assert_allclose(bundle["positive_global"], 1.0)
    assert bundle["null_global"].mean() < 0.2


def test_matched_condition_observations_use_same_units_and_groups() -> None:
    positive = np.ones((4, 3, 5))
    null = np.zeros((4, 3, 20, 5))
    values, labels, groups = matched_condition_observations(
        positive, null, np.arange(4), nulls_per_condition=4, seed=9,
    )
    assert values.shape == (4 * 3 + 4 * 3 * 4, 5)
    assert int(labels.sum()) == 12
    assert set(groups) == set(range(4))
    for participant in range(4):
        assert np.count_nonzero(groups == participant) == 15


def test_regions_must_exactly_partition_channels() -> None:
    labels = ("A", "B", "C", "D")
    regions = region_indices(labels, (("A", "C"), ("B", "D")))
    np.testing.assert_array_equal(regions[0], [0, 2])
    np.testing.assert_array_equal(regions[1], [1, 3])
