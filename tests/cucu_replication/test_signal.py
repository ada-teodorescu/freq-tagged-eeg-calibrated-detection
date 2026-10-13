import numpy as np

from cucu_replication.signal import (
    hann_scalloping_loss,
    hilbert_itpc,
    phase_randomize_trials,
    published_fft_itpc,
    squared_itpc,
    weighted_sinusoid_coefficients,
)


def test_squared_itpc_extremes():
    same = np.exp(1j * np.full((10, 3), 0.4))
    assert np.allclose(squared_itpc(same, axis=0), 1.0)
    balanced = np.exp(1j * np.linspace(0, 2 * np.pi, 10, endpoint=False))
    assert np.allclose(squared_itpc(balanced), 0.0, atol=1e-14)


def test_published_fft_detects_phase_locked_four_hz():
    sampling_rate = 1000
    time = np.arange(5000) / sampling_rate
    trial = np.sin(2 * np.pi * 4.0 * time)
    trials = np.tile(trial, (10, 2, 1))
    result = published_fft_itpc(trials, sampling_rate, (4.0, 8.0), crop_samples=500)
    assert result.shape == (2, 2)
    assert np.allclose(result[:, 0], 1.0)


def test_hilbert_reports_mean_r_squared_not_square_of_mean_r():
    sampling_rate = 100
    time = np.arange(500) / sampling_rate
    trials = np.stack([
        np.sin(2 * np.pi * 4 * time + phase) for phase in np.linspace(0, 0.2, 10)
    ])[:, None, :]
    mean_r, mean_squared, trace = hilbert_itpc(trials, sampling_rate, 4.0, 0.5, 4, 0.5)
    assert mean_r.shape == mean_squared.shape == (1,)
    assert trace.shape == (400,)
    assert np.all(mean_squared >= np.square(mean_r) - 1e-12)


def test_phase_randomization_preserves_amplitude_spectrum():
    rng = np.random.default_rng(3)
    values = rng.normal(size=(3, 2, 100))
    randomized = phase_randomize_trials(values, np.random.default_rng(8))
    np.testing.assert_allclose(
        np.abs(np.fft.rfft(values, axis=-1)),
        np.abs(np.fft.rfft(randomized, axis=-1)),
        rtol=1e-10,
        atol=1e-10,
    )


def test_arbitrary_frequency_fit_recovers_off_bin_line():
    sampling_rate = 1000
    time = np.arange(5000) / sampling_rate
    frequency = 4.037
    values = np.cos(2 * np.pi * frequency * time + 0.3)
    trials = np.tile(values, (10, 2, 1))
    correct = weighted_sinusoid_coefficients(trials, sampling_rate, frequency)
    wrong = weighted_sinusoid_coefficients(trials, sampling_rate, 4.0)
    assert np.mean(np.abs(correct)) > np.mean(np.abs(wrong))
    assert np.allclose(squared_itpc(correct, axis=0), 1.0)


def test_hann_scalloping_is_one_at_zero_and_symmetric():
    assert hann_scalloping_loss(0.0, 4.5) == 1.0
    assert np.allclose(hann_scalloping_loss(0.1, 4.5), hann_scalloping_loss(-0.1, 4.5))
