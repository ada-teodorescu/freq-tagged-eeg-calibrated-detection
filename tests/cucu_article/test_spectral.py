from __future__ import annotations

import numpy as np

from cucu_article.spectral import frequency_indices, hann_coefficients, phase_features, score_bundle
from cucu_detection.detectors import deterministic_subsets


def test_fixed_frequency_grid_and_duration_zero_padding() -> None:
    target, neighbors, offsets = frequency_indices(1000, 4500, (4, 8, 12, 16), 18)
    np.testing.assert_array_equal(target, [18, 36, 54, 72])
    assert neighbors.shape == (4, 18)
    assert 0 not in offsets
    time = np.arange(4500) / 1000
    trials = np.stack([np.cos(2 * np.pi * 4 * time)] * 10)[:, None, :]
    full, _ = hann_coefficients(trials, 1000, 4.5, 4500, (4, 8, 12, 16), 18)
    short, _ = hann_coefficients(trials, 1000, 0.5, 4500, (4, 8, 12, 16), 18)
    assert full.shape == short.shape == (10, 1, 4)
    assert np.abs(full[..., 0]).mean() > np.abs(full[..., 1:]).mean()


def test_phase_features_and_joint_statistic_have_expected_shapes() -> None:
    rng = np.random.default_rng(4)
    target = np.exp(1j * np.zeros((10, 3, 4)))
    neighbors = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(10, 3, 4, 18)))
    subsets = deterministic_subsets(10, 5, 20, 99)
    phase = phase_features(target, subsets)
    bundle = score_bundle(target, neighbors, subsets)
    assert phase.shape == (20, 4)
    np.testing.assert_allclose(phase, 1)
    np.testing.assert_allclose(bundle["positive_joint"], 4 * 2 * 5)
    assert bundle["null_phase"].shape == (20, 18, 4)
    assert bundle["null_hotelling"].shape == (20, 18)


def test_duration_scores_are_deterministic() -> None:
    rng = np.random.default_rng(7)
    trials = rng.normal(size=(10, 2, 4500))
    first = hann_coefficients(trials, 1000, 2.0, 4500, (4, 8, 12, 16), 18)
    second = hann_coefficients(trials, 1000, 2.0, 4500, (4, 8, 12, 16), 18)
    np.testing.assert_array_equal(first[0], second[0]); np.testing.assert_array_equal(first[1], second[1])
