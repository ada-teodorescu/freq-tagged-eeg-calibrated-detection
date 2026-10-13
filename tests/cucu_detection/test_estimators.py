import numpy as np

from cucu_detection.estimators import (
    dpss_line_estimates,
    dpss_multitaper_coefficients,
    hilbert_coefficients,
    least_squares_coefficients,
    published_hann_dft,
)


def test_arbitrary_frequency_least_squares_recovers_phase():
    sampling_rate = 200
    time = np.arange(1000) / sampling_rate
    phase = 0.73
    trials = np.stack([np.cos(2 * np.pi * 4.13 * time + phase)] * 5)[:, None, :]
    coefficients = least_squares_coefficients(trials, sampling_rate, np.asarray([4.13]), crop_samples=0)
    assert coefficients.shape == (5, 1, 1)
    np.testing.assert_allclose(np.angle(coefficients[:, 0, 0]), phase, atol=2e-3)


def test_published_dft_and_multitaper_detect_harmonic_line():
    sampling_rate = 200
    time = np.arange(900) / sampling_rate
    trials = np.stack([np.sin(2 * np.pi * 4 * time)] * 4)[:, None, :]
    dft = published_hann_dft(trials, sampling_rate, np.asarray([4.0, 7.0]), crop_samples=0)
    assert np.mean(np.abs(dft[..., 0])) > 100 * np.mean(np.abs(dft[..., 1]))
    multitaper = dpss_multitaper_coefficients(trials, sampling_rate, np.asarray([4.0]),
                                              crop_samples=0, tapers=3)
    assert multitaper.shape == (4, 1, 1, 3)


def test_hilbert_demodulation_recovers_stationary_phase():
    sampling_rate = 200
    time = np.arange(1200) / sampling_rate
    phase = 0.61
    trials = np.stack([np.cos(2 * np.pi * 4 * time + phase)] * 4)[:, None, :]
    coefficients = hilbert_coefficients(
        trials, sampling_rate, np.asarray([4.0]), edge_crop_seconds=0.5,
    )
    assert np.mean(np.abs(coefficients)) > 0.9
    np.testing.assert_allclose(np.angle(coefficients[:, 0, 0]), phase, atol=2e-2)


def test_thomson_line_estimate_recovers_phase_and_rejects_off_line():
    sampling_rate = 200
    time = np.arange(1000) / sampling_rate
    phase = -0.42
    trials = np.stack([np.cos(2 * np.pi * 4 * time + phase)] * 3)[:, None, :]
    line, f_statistic, ratios = dpss_line_estimates(
        trials, sampling_rate, np.asarray([4.0, 6.0]), crop_samples=0, tapers=4,
    )
    assert line.shape == f_statistic.shape == (3, 1, 2)
    assert ratios.shape == (4,)
    np.testing.assert_allclose(np.angle(line[..., 0]), phase, atol=2e-2)
    assert np.min(f_statistic[..., 0]) > 100 * np.max(f_statistic[..., 1])
