import numpy as np

from eeg_detection.detectors import (
    debiased_ppc,
    detector_score,
    deterministic_subsets,
    rayleigh_statistic,
    squared_itpc,
)
from eeg_detection.stages.detector_screen import _vectorized_scores
from eeg_detection.stages.spatial_screen import _score_subsets


def test_d1_d2_d3_are_exact_fixed_k_transformations():
    rng = np.random.default_rng(4)
    coefficients = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(7, 5, 4)))
    d1 = squared_itpc(coefficients)
    np.testing.assert_allclose(debiased_ppc(coefficients), (7 * d1 - 1) / 6, atol=1e-15)
    np.testing.assert_allclose(rayleigh_statistic(coefficients), 7 * d1, atol=1e-15)


def test_phase_locked_exceeds_phase_random_and_joint_harmonic_gains():
    rng = np.random.default_rng(5)
    locked = np.exp(1j * rng.normal(scale=0.08, size=(10, 8, 4)))
    random = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(10, 8, 4)))
    assert detector_score("D1", locked[..., :1]) > detector_score("D1", random[..., :1])
    assert detector_score("D8", locked) > 3 * detector_score("D3", locked[..., :1])


def test_trial_subsets_are_complete_or_deterministically_capped():
    complete = deterministic_subsets(10, 2, 200, 17)
    assert complete.shape == (45, 2)
    first = deterministic_subsets(10, 5, 200, 17)
    second = deterministic_subsets(10, 5, 200, 17)
    assert first.shape == (200, 5)
    np.testing.assert_array_equal(first, second)


def test_amplitude_weighted_and_known_phase_detectors_are_finite():
    rng = np.random.default_rng(6)
    coefficients = rng.lognormal(size=(10, 3, 1)) * np.exp(1j * rng.normal(0.4, 0.2, (10, 3, 1)))
    for detector in ("D4", "D5", "D6"):
        score = detector_score(detector, coefficients, expected_phase=0.4)
        assert np.isfinite(score)


def test_vectorized_subset_screen_matches_scalar_detector_definitions():
    rng = np.random.default_rng(12)
    coefficients = rng.normal(size=(6, 5, 4, 4)) + 1j * rng.normal(size=(6, 5, 4, 4))
    for detector in ("D1", "D2", "D3", "D4", "D5", "D6", "D8"):
        vector = _vectorized_scores(detector, coefficients, np.asarray([0.2]))
        scalar = np.asarray([
            detector_score(detector, value if detector == "D8" else value[..., :1], expected_phase=0.2)
            for value in coefficients
        ])
        np.testing.assert_allclose(vector, scalar, rtol=1e-10, atol=1e-10)


def test_public_subset_vectorization_preserves_subset_order():
    rng = np.random.default_rng(13)
    coefficients = rng.normal(size=(10, 3, 4)) + 1j * rng.normal(size=(10, 3, 4))
    subsets = deterministic_subsets(10, 3, 20, 8)
    positive, null = _score_subsets(coefficients, None, "D8", subsets, 3, rng)
    expected = np.asarray([detector_score("D8", coefficients[subset]) for subset in subsets])
    np.testing.assert_allclose(positive, expected, atol=1e-12)
    assert null.shape == (20, 3)
