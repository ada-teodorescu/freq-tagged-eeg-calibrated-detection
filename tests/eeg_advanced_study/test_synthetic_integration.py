from __future__ import annotations

import numpy as np

from eeg_advanced_study.algorithms import (
    calibrate_pocock_boundaries, crossing_rate, first_crossings, fixed_stream_scores,
    phase_trajectories, temporal_shift_multipliers,
)


def test_coupled_surrogate_to_sequential_decision_chain():
    rng = np.random.default_rng(911)
    streams, trials, channels = 12, 10, 6
    base_phase = rng.normal(0, 0.18, size=(streams, trials, 1))
    spatial = rng.normal(0, 0.1, size=(streams, 1, channels))
    coefficients = np.empty((streams, trials, channels, 2), np.complex64)
    coefficients[..., 0] = np.exp(1j * (base_phase + spatial))
    coefficients[..., 1] = 0.6 * np.exp(1j * (2 * base_phase + 0.5 * spatial))
    rotations = temporal_shift_multipliers(trials, (1, 2), 2_000, 73)
    positive_r2, _ = phase_trajectories(coefficients, tuple(range(2, 11)))
    null_r2, _ = phase_trajectories(coefficients, tuple(range(2, 11)), rotations=rotations)
    positive = fixed_stream_scores(positive_r2, tuple(range(2, 11)))[..., 1]
    null = fixed_stream_scores(null_r2, tuple(range(2, 11)))[..., 1]
    training = null[:6].reshape(-1, 9)
    heldout = null[6:].reshape(-1, 9)
    calibration = calibrate_pocock_boundaries(training, .05)
    assert crossing_rate(heldout, calibration.boundaries) < .065
    first = first_crossings(positive[6:], calibration.boundaries, tuple(range(2, 11)))
    assert np.mean(first <= 10) > .9
