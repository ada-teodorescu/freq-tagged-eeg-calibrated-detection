from __future__ import annotations

import numpy as np

from eeg_core.algorithms import (
    fit_logistic_l2, grouped_folds, make_training_observations, select_regularization,
)
from eeg_core.stages.benchmark import empirical_threshold_fast, leave_one_participant_thresholds


def test_logistic_model_learns_separable_harmonic_features() -> None:
    rng = np.random.default_rng(5)
    negative = rng.normal(-1, 0.3, size=(200, 4))
    positive = rng.normal(1, 0.3, size=(200, 4))
    values = np.vstack((negative, positive)); labels = np.r_[np.zeros(200), np.ones(200)]
    model = fit_logistic_l2(values, labels, 10.0)
    assert model.converged
    assert model.decision_function(positive).mean() > model.decision_function(negative).mean()
    np.testing.assert_allclose(model.standardizer.mean, values.mean(axis=0))


def test_grouped_regularization_is_deterministic_and_group_safe() -> None:
    rng = np.random.default_rng(8)
    groups = np.repeat(np.arange(12), 20)
    labels = np.tile(np.r_[np.zeros(10), np.ones(10)], 12)
    values = rng.normal(size=(len(labels), 4)) + labels[:, None] * 0.5
    folds = grouped_folds(groups, 6, 91)
    assert sorted(np.concatenate(folds).tolist()) == list(range(12))
    assert all(not set(left) & set(right) for i, left in enumerate(folds) for right in folds[i + 1:])
    result1 = select_regularization(values, labels, groups, (0.01, 0.1, 1.0), 6, 91, 100, 1e-8)
    result2 = select_regularization(values, labels, groups, (0.01, 0.1, 1.0), 6, 91, 100, 1e-8)
    assert result1 == result2


def test_training_observations_preserve_participant_groups() -> None:
    positive = np.ones((3, 6, 2, 4))
    null = np.zeros((3, 6, 2, 5, 4))
    values, labels, groups = make_training_observations(positive, null, np.array([10, 20, 30]))
    assert values.shape == (3 * 2 + 3 * 2 * 5, 4)
    assert labels.sum() == 6
    assert set(groups) == {10, 20, 30}
    assert np.all(values[labels == 1] == 1)
    assert np.all(values[labels == 0] == 0)


def test_fast_leave_one_participant_thresholds_are_exact() -> None:
    rng = np.random.default_rng(22)
    values = rng.normal(size=(7, 5, 11))
    thresholds = leave_one_participant_thresholds(values, 0.05)
    for heldout in range(7):
        expected = empirical_threshold_fast(values[np.arange(7) != heldout], 0.05)
        assert thresholds[heldout] == expected
