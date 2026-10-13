from types import SimpleNamespace

import numpy as np

from cucu_detection.detectors import deterministic_subsets
from cucu_improvement.stages.public_coefficients import _spectral_coefficients
from cucu_improvement.stages.public_focused import (
    _phase_score, _phase_surrogate, select_train_only_channel,
)
from cucu_improvement.stages.stored_d8 import (
    _auc_against_sorted_null, _empirical_threshold_fast, _method_scores, fit_train_only_weights,
)
from cucu_detection.evaluation import auc_from_scores, empirical_threshold


def test_train_only_weights_favor_informative_harmonic_without_held_data():
    rng = np.random.default_rng(4)
    negative = rng.normal(0.1, 0.01, size=(24, 5, 4, 18))
    positive = rng.normal(0.1, 0.01, size=(24, 5, 4))
    positive[..., 1] += 0.25
    weights = fit_train_only_weights(positive, negative)
    assert np.argmax(weights) == 1
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_4_8_d8_ignores_upper_harmonics():
    positive = np.zeros((2, 3, 4)); negative = np.zeros((2, 3, 4, 18))
    weights = np.full(4, 0.25)
    first, first_null = _method_scores("D8_4_8_equal", positive, negative, 5, weights)
    positive[..., 2:] = 100; negative[..., 2:, :] = 100
    second, second_null = _method_scores("D8_4_8_equal", positive, negative, 5, weights)
    np.testing.assert_array_equal(first, second); np.testing.assert_array_equal(first_null, second_null)


def test_corrected_public_spectral_cache_has_matched_neighbor_axes():
    config = SimpleNamespace(
        fft_crop_samples=100, harmonics_hz=(4.0, 8.0, 12.0, 16.0), sampling_rate=200,
        neighbor_bins_per_harmonic=18, dpss_time_bandwidth=2.5, dpss_tapers=4,
    )
    time = np.arange(1000) / config.sampling_rate
    trials = np.stack([np.cos(2 * np.pi * 4 * time + phase) for phase in (0.2, 0.3)])[:, None, :]
    result = _spectral_coefficients(trials, config)
    assert result["F1_target"].shape == result["F3_line_target"].shape == (2, 1, 4)
    assert result["F1_neighbors"].shape == result["F3_line_neighbors"].shape == (2, 1, 4, 18)
    assert result["F3_f_neighbors"].shape == (2, 1, 4, 18)
    assert np.mean(result["F3_f_target"][..., 0]) > np.mean(result["F3_f_neighbors"][..., 0, :])


def test_phase_surrogate_is_deterministic_and_breaks_locking():
    coefficients = np.ones((10, 2, 2), dtype=np.complex128)
    subsets = deterministic_subsets(10, 5, 20, 7)
    positive = _phase_score(coefficients[..., None], subsets, (0, 1), None)[..., 0]
    first = _phase_surrogate(coefficients, subsets, (0, 1), None, np.random.SeedSequence([7]))
    second = _phase_surrogate(coefficients, subsets, (0, 1), None, np.random.SeedSequence([7]))
    np.testing.assert_array_equal(first, second)
    assert positive.mean() > first.mean()


def test_fast_calibration_and_auc_match_reference_implementations():
    rng = np.random.default_rng(11)
    null = rng.normal(size=1001); positive = rng.normal(0.5, 1.0, size=37)
    assert _empirical_threshold_fast(null, 0.05) == empirical_threshold(null, 0.05)
    np.testing.assert_allclose(_auc_against_sorted_null(positive, np.sort(null)), auc_from_scores(positive, null))


def test_s1_selection_cannot_see_held_condition():
    quality = np.zeros((15, 32)); quality[:14, 3] = 1.0
    selected = select_train_only_channel(quality, list(range(14)))
    quality[14, 20] = 1e9
    assert selected == 3
    assert select_train_only_channel(quality, list(range(14))) == selected
