from __future__ import annotations

import numpy as np

from cucu_advanced_study.algorithms import (
    calibrate_pocock_boundaries, crossing_rate, first_crossings, phase_trajectories,
    ridge_complex_trajectories, select_shrinkage_weight, temporal_shift_multipliers,
)


def test_temporal_shift_is_deterministic_and_preserves_locked_invariants():
    first = temporal_shift_multipliers(500, (1, 2, 3, 4), 3, 17)
    second = temporal_shift_multipliers(500, (1, 2, 3, 4), 3, 17)
    assert np.array_equal(first, second)
    coefficients = np.ones((3, 500, 4, 4), dtype=np.complex64)
    coefficients[..., 1, :] *= np.exp(0.7j)
    randomized = coefficients[:, :, :, :] * first[:, :, None, :]
    # A shared channel rotation preserves spatial phase differences exactly.
    assert np.allclose(randomized[..., 1, :] / randomized[..., 0, :], np.exp(0.7j), atol=1e-6)
    # phi_2 - 2*phi_1 is invariant under the coupled harmonic shift.
    before = np.angle(coefficients[..., 1]) - 2 * np.angle(coefficients[..., 0])
    after = np.angle(randomized[..., 1]) - 2 * np.angle(randomized[..., 0])
    assert np.allclose(np.exp(1j * before), np.exp(1j * after), atol=1e-5)
    assert np.max(np.abs(first[..., 0].mean(axis=1))) < 0.12


def test_independent_harmonic_sensitivity_does_not_preserve_cross_harmonic_phase():
    rotations = temporal_shift_multipliers(200, (1, 2), 2, 91, independent_harmonics=True)
    invariant = np.exp(1j * (np.angle(rotations[..., 1]) - 2 * np.angle(rotations[..., 0])))
    assert np.std(np.angle(invariant)) > 1.0


def test_phase_trajectories_are_nested_prefixes():
    rng = np.random.default_rng(4)
    coefficients = np.exp(1j * rng.normal(size=(2, 10, 3, 2))).astype(np.complex64)
    k_values = (2, 5, 10)
    global_scores, _ = phase_trajectories(coefficients, k_values)
    for look, k in enumerate(k_values):
        manual = np.abs(coefficients[:, :k].mean(axis=1)) ** 2
        assert np.allclose(global_scores[:, look], manual.mean(axis=1), atol=1e-6)


def test_pocock_boundaries_control_complete_trajectory_fpr():
    rng = np.random.default_rng(8)
    innovations = rng.normal(size=(80_000, 9))
    trajectories = np.cumsum(innovations, axis=1) / np.sqrt(np.arange(1, 10))
    calibration = calibrate_pocock_boundaries(trajectories[:40_000], 0.05)
    heldout_fpr = crossing_rate(trajectories[40_000:], calibration.boundaries)
    assert calibration.familywise_fpr <= 0.05
    assert 0.043 <= heldout_fpr <= 0.057
    first = first_crossings(trajectories[:10], calibration.boundaries, tuple(range(2, 11)))
    assert np.all(np.isin(first, (*range(2, 11), 11)))


def test_ridge_complex_handles_k2_singular_and_zero_variance_for_every_lock_value():
    rng = np.random.default_rng(12)
    singular = np.ones((3, 10, 5, 2), dtype=np.complex64)
    singular[:, 1::2] *= np.exp(0.2j)
    rotations = temporal_shift_multipliers(10, (1, 2), 7, 22)
    for ridge in (0.05, 0.1, 0.25, 0.5, 1.0):
        positive = ridge_complex_trajectories(singular, (2, 10), ridge)
        null = ridge_complex_trajectories(singular, (2, 10), ridge, rotations)
        assert positive.shape == (3, 2)
        assert null.shape == (3, 7, 2)
        assert np.all(np.isfinite(positive)) and np.all(positive >= 0)
        assert np.all(np.isfinite(null)) and np.all(null >= 0)
    zeros = np.zeros((1, 2, 2, 2), dtype=np.complex64)
    assert np.all(ridge_complex_trajectories(zeros, (2,), 0.05) == 0)


def test_shrinkage_selection_uses_null_error_and_ties_to_less_adaptation():
    population = np.array([1.0, 1.0])
    local = np.ones((4, 2))
    validation = np.zeros((4, 20, 2))
    weight, errors = select_shrinkage_weight(population, local, validation, (0.0, .25, .5, .75, 1.0), .05)
    assert weight == 0.0
    assert len(set(errors.values())) == 1


def test_shrinkage_has_population_fallback_for_invalid_local_boundaries():
    population = np.array([1.0, 1.0])
    local = np.full((3, 2), np.nan)
    validation = np.zeros((3, 10, 2))
    weight, errors = select_shrinkage_weight(population, local, validation, (0.0, .5, 1.0), .05)
    assert weight == 0.0
    assert np.isfinite(errors[0.0])
    assert np.isinf(errors[0.5]) and np.isinf(errors[1.0])
