from __future__ import annotations

import numpy as np
import pandas as pd

from cucu_validation_extension.algorithms import (
    causal_demodulation_coefficients, causal_settling_samples, equal_weight_score, linear_effective_kernel, mcnemar_exact,
    rectangular_ls_coefficients, simultaneous_crossed_band,
)
from cucu_validation_extension.config import load_config
from cucu_validation_extension.stages import stage_registry
from cucu_validation_extension.stages.chronological import completion_event
from cucu_validation_extension.stages.double_holdout import _training_arrays
from cucu_validation_extension.stages.external_robustness import randomized_nonoverlapping


def test_config_and_stage_graph():
    config = load_config("validation_extension_config.toml")
    assert config.shared_channels == tuple(ch for ch in config.target_channels if ch not in {"TP9", "TP10", "PO9", "PO10"})
    assert len(config.shared_channels) == 28
    stages = stage_registry()
    assert tuple(stages) == ("00_freeze", "01_estimators", "02_benchmarks", "03_double_holdout",
                             "04_chronological", "05_external_robustness", "06_statistics", "07_report")
    assert stages["07_report"].dependencies == ("06_statistics",)


def test_rectangular_ls_recovers_known_complex_coefficient():
    sampling_rate, frequency, samples = 1000, 4.0, 5000
    time = np.arange(samples) / sampling_rate
    phase = 0.37; signal = 2.5 * np.cos(2 * np.pi * frequency * time + phase)
    coefficient = rectangular_ls_coefficients(signal[None, :], sampling_rate, np.asarray([frequency]), 500, 4500)[0, 0]
    assert np.isclose(abs(coefficient), 2.5, atol=1e-10)
    assert np.isclose(np.angle(coefficient), phase, atol=1e-10)


def test_effective_kernel_reconstructs_linear_estimator():
    weights = np.asarray([1 + 2j, -0.5j, 3 - 1j, 0.25 + 0.75j])
    kernel = linear_effective_kernel(lambda values: values @ weights, len(weights), chunk_size=2)
    signal = np.asarray([0.5, -2.0, 1.5, 4.0])
    assert np.allclose(kernel, weights)
    assert np.allclose(signal @ kernel, signal @ weights)


def test_causal_demodulator_leaves_usable_support_and_recovers_line():
    sampling_rate, frequency = 1000, 4.0
    time = np.arange(5500) / sampling_rate
    values = np.cos(2 * np.pi * frequency * time)[None, :]
    settling = causal_settling_samples(sampling_rate, 4, 0.5)
    coefficient = causal_demodulation_coefficients(values, sampling_rate, np.asarray([frequency]),
                                                    4, 0.5, settling)[0, 0]
    assert 5500 - settling >= 2000
    assert np.isclose(abs(coefficient), 0.5, atol=0.05)


def test_equal_weight_score_harmonic_subsets():
    features = np.arange(1, 21, dtype=float)
    assert np.isclose(equal_weight_score(features, 4), features.mean())
    expected = features.reshape(5, 4)[:, :2].mean()
    assert np.isclose(equal_weight_score(features, 2), expected)


def test_double_holdout_excludes_participant_and_condition():
    positive = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    null = np.arange(3 * 4 * 5 * 2, dtype=float).reshape(3, 4, 5, 2)
    train_positive, train_null = _training_arrays(positive, null, held_p=1, held_c=2)
    expected = positive[np.ix_([True, False, True], [True, True, False, True])].reshape(-1, 2)
    assert np.array_equal(train_positive, expected)
    assert train_positive.shape == (6, 2)
    assert train_null.shape == (30, 2)


def test_chronological_completion_is_last_kth_stream_event():
    frame = pd.DataFrame({
        "stream_id": [1, 2, 3, 1, 3, 2],
        "trial_index": [2, 3, 5, 7, 8, 10],
        "time_seconds": [2., 3., 5., 7., 8., 10.],
    })
    event = completion_event(frame, 2)
    assert int(event["trial_index"]) == 10


def test_simultaneous_band_and_exact_mcnemar_are_deterministic():
    gains = np.zeros((4, 3, 2)); gains[..., 0] = 0.1; gains[..., 1] = -0.2
    low, high = simultaneous_crossed_band(gains, 100, 7)
    assert np.allclose(low, [0.1, -0.2]) and np.allclose(high, [0.1, -0.2])
    result = mcnemar_exact(np.asarray([1, 1, 0, 0]), np.asarray([1, 0, 1, 0]))
    assert result["first_only"] == 1 and result["second_only"] == 1 and result["p_value"] == 1.0


def test_random_window_selection_is_reproducible_and_nonoverlapping():
    first = randomized_nonoverlapping(np.arange(20, dtype=float), 2.0, 21.0, 6, 5, 17)
    second = randomized_nonoverlapping(np.arange(20, dtype=float), 2.0, 21.0, 6, 5, 17)
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)
        assert len(left) == 6
        assert np.all(np.diff(left) >= 2.0)
