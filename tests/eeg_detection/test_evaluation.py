import numpy as np

from eeg_detection.evaluation import (
    auc_from_scores,
    crossed_bootstrap_gain,
    empirical_threshold,
    holm_adjust,
)


def test_empirical_five_percent_calibration_and_auc():
    train = np.arange(1_000, dtype=float)
    threshold = empirical_threshold(train, 0.05)
    rate = np.mean(train > threshold)
    assert 0.045 <= rate <= 0.05
    assert auc_from_scores(np.arange(10, 20), np.arange(10)) == 1.0


def test_crossed_bootstrap_is_deterministic_and_holm_is_ordered():
    gains = np.full((8, 5), 0.2)
    assert crossed_bootstrap_gain(gains, 100, 9) == crossed_bootstrap_gain(gains, 100, 9)
    adjusted = holm_adjust(np.asarray([0.01, 0.04, 0.03]))
    assert np.all(adjusted >= np.asarray([0.01, 0.04, 0.03]))
