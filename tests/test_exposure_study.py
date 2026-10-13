"""Focused scientific invariants; no source EEG or full analysis required."""
from types import SimpleNamespace

import numpy as np
import pytest

from eeg_exposure_study import analysis
from eeg_exposure_study.runtime import Run


def test_halves_use_disjoint_equal_trial_sets_and_paired_surrogates():
    source = SimpleNamespace(streams=3, conditions=1, surrogates=4,
                             surrogate_seeds=(41,), surrogates_per_seed=4)
    coefficients = np.ones((3, 10, 2, 4), dtype=np.complex64)
    coefficients[:, 5:] = np.exp(2j * np.pi * np.arange(5) / 5)[None, :, None, None]
    regions = (np.array([0]), np.array([1]))
    positive, null, r2 = analysis.half_features(coefficients, regions, source, 0)
    assert positive.shape == (2, 1, 2, 4)
    assert null.shape == (2, 1, 4, 2, 4)
    np.testing.assert_allclose(positive[0], 10, atol=1e-6)
    np.testing.assert_allclose(positive[1], 0, atol=1e-6)
    np.testing.assert_allclose(r2[0], 1, atol=1e-6)
    np.testing.assert_allclose(r2[1], 0, atol=1e-6)
    duplicate = analysis.half_features(coefficients.copy(), regions, source, 0)
    np.testing.assert_array_equal(null, duplicate[1])


def test_fold_excludes_both_held_halves_from_fit_and_threshold(monkeypatch):
    positive = np.arange(12, dtype=float).reshape(3, 2, 2, 1)
    null = -np.arange(24, dtype=float).reshape(3, 2, 2, 2, 1)
    positive[1] = 999
    null[1] = -999
    calls = {}

    def fit(values, labels, *args):
        calls["fit"] = values.copy()
        return SimpleNamespace(
            converged=True, coefficients=np.ones(1), intercept=0,
            standardizer=SimpleNamespace(mean=np.zeros(1), scale=np.ones(1)),
            decision_function=lambda x: x[..., 0],
        )

    def threshold(values, target):
        calls["calibration"] = values.copy()
        return 12.0

    monkeypatch.setattr(analysis, "fit_monotone_logistic", fit)
    monkeypatch.setattr(analysis, "empirical_threshold", threshold)
    config = SimpleNamespace(source=SimpleNamespace(locked_c=100, target_fpr=0.05),
                             maximum_iterations=150, tolerance=1e-8)
    result = analysis.fit_fold(positive, null, 1, config)
    assert not np.isin(calls["fit"], [-999, 999]).any()
    assert not np.isin(calls["calibration"], [-999, 999]).any()
    np.testing.assert_array_equal(result["training_indices"], [0, 2])
    assert result["positive_scores"].shape == (2, 2)
    assert result["threshold"].ndim == 0  # shared across halves
    assert np.all(result["positive_scores"] == 999)


def test_failed_optimizer_is_not_accepted(monkeypatch):
    monkeypatch.setattr(analysis, "fit_monotone_logistic",
                        lambda *args: SimpleNamespace(converged=False))
    config = SimpleNamespace(source=SimpleNamespace(locked_c=100),
                             maximum_iterations=1, tolerance=1e-8)
    with pytest.raises(RuntimeError, match="did not converge"):
        analysis.fit_fold(np.ones((3, 2, 1, 1)), np.ones((3, 2, 1, 2, 1)), 0, config)


def test_checkpoint_detects_corruption_and_run_identity(tmp_path):
    run = Run.__new__(Run)
    run.identifier = "first"
    path = tmp_path / "unit.npz"
    run.save(path, value=np.arange(3))
    assert run.valid(path)
    run.identifier = "second"
    assert not run.valid(path)
    run.identifier = "first"
    path.write_bytes(b"interrupted or corrupted checkpoint")
    assert not run.valid(path)
