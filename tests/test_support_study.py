"""Support isolation and static-fold invariants; no source EEG required."""
from types import SimpleNamespace

import numpy as np
import pytest

from eeg_exposure_study import analysis as fitting
from eeg_support_study.analysis import _verify_coefficients
from eeg_support_study.estimators import restricted_hilbert


@pytest.mark.parametrize("reflection", ["odd", "even"])
def test_outside_samples_cannot_affect_restricted_estimator(reflection):
    raw = np.random.default_rng(17).normal(size=(2, 3, 5000))
    changed = raw.copy()
    # Even invalid values outside the retained interval must not enter processing.
    changed[..., :500] = np.nan
    changed[..., 4500:] = np.inf
    expected = restricted_hilbert(raw, 1000, (4.0, 8.0), reflection=reflection)
    actual = restricted_hilbert(changed, 1000, (4.0, 8.0), reflection=reflection)
    assert actual.shape == (2, 3, 2)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("reflection", ["odd", "even"])
def test_retained_signal_affects_estimate_and_preserves_linearity(reflection):
    time = np.arange(5000) / 1000
    raw = np.cos(2 * np.pi * 8 * time)[None, None, :]
    estimate = restricted_hilbert(raw, 1000, (8.0,), reflection=reflection)
    doubled = restricted_hilbert(2 * raw, 1000, (8.0,), reflection=reflection)
    assert abs(estimate.item()) > 0.1
    np.testing.assert_allclose(doubled, 2 * estimate, rtol=1e-10, atol=1e-10)


def test_invalid_interval_and_nonfinite_retained_data_fail():
    raw = np.ones((1, 5000))
    with pytest.raises(ValueError, match="interval"):
        restricted_hilbert(raw, 1000, (4.0,), stop=6000)
    raw[..., 1000] = np.nan
    with pytest.raises(ValueError, match="Nonfinite retained"):
        restricted_hilbert(raw, 1000, (4.0,))


def test_source_coefficient_mismatch_stops_comparison():
    reference = np.ones((2, 3), dtype=complex)
    assert _verify_coefficients(reference.copy(), reference, "test") == 0
    with pytest.raises(ValueError, match="mismatch"):
        _verify_coefficients(2 * reference, reference, "test")


def test_static_fold_excludes_held_participant_and_calibrates_training_nulls(monkeypatch):
    positive = np.arange(12, dtype=float).reshape(3, 2, 2)
    null = -np.arange(48, dtype=float).reshape(3, 2, 4, 2)
    positive[1] = 999
    null[1] = -999
    captured = {}

    def fit(values, labels, *args):
        captured["training"] = values.copy()
        return SimpleNamespace(
            converged=True, coefficients=np.ones(2), intercept=0,
            standardizer=SimpleNamespace(mean=np.zeros(2), scale=np.ones(2)),
            decision_function=lambda x: x.sum(axis=-1),
        )

    def threshold(scores, target):
        captured["calibration"] = scores.copy()
        return -5.0

    monkeypatch.setattr(fitting, "fit_monotone_logistic", fit)
    monkeypatch.setattr(fitting, "empirical_threshold", threshold)
    config = SimpleNamespace(source=SimpleNamespace(locked_c=100, target_fpr=0.05),
                             maximum_iterations=150, tolerance=1e-8)
    result = fitting.fit_fold(positive, null, 1, config)
    assert not np.isin(captured["training"], [-999, 999]).any()
    assert not np.isin(captured["calibration"], [-1998, 1998]).any()
    assert result["positive_scores"].shape == (2,)
    assert result["null_scores"].shape == (2, 4)
    np.testing.assert_array_equal(result["training_indices"], [0, 2])
