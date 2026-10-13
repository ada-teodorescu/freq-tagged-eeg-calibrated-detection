import numpy as np

from eeg_raw_replication.spectral import analyse_epochs, harmonic_coefficients, squared_itpc
from eeg_raw_replication.statistics import (
    agreement_metrics, classify_agreement, covariance_pca, project_pca,
)


def test_hann_fft_and_itpc_recover_phase_locked_harmonics():
    sampling_rate = 1000
    time = np.arange(5000) / sampling_rate
    trial = np.sin(2 * np.pi * 4 * time)
    trials = np.repeat(trial[None, None, :], 10, axis=0)
    coefficients = harmonic_coefficients(trials, sampling_rate, (4.0, 8.0), 500)
    assert coefficients.shape == (10, 1, 2)
    assert squared_itpc(coefficients, axis=0)[0, 0] > 0.999999


def test_analyse_epochs_preserves_stream_channel_harmonic_dimensions():
    time = np.arange(5000) / 1000
    epochs = []
    ids = []
    for stream in range(1, 4):
        for trial in range(2):
            epochs.append(np.stack([
                np.sin(2 * np.pi * 4 * time + trial * 0.1),
                np.cos(2 * np.pi * 8 * time + trial * 0.1),
            ]))
            ids.append(stream)
    result = analyse_epochs(np.asarray(epochs), np.asarray(ids), 1000, (4.0, 8.0), 500, streams=3, trials_per_stream=2)
    assert result.coefficients.shape == (3, 2, 2, 2)
    assert result.channel_itpc.shape == (3, 2, 2)
    assert result.mean_itpc.shape == (3, 2)


def test_pca_projection_and_orientation_are_deterministic():
    values = np.asarray([[1, 2, 3, 4], [2, 3, 5, 7], [4, 6, 8, 9], [8, 7, 6, 5]], dtype=float)
    pca = covariance_pca(values)
    assert pca.loadings[0, 0] >= 0
    assert np.allclose(project_pca(values, pca), pca.scores)
    assert np.isclose(pca.explained_variance_ratio.sum(), 1.0)


def test_agreement_classification_uses_declared_thresholds():
    metrics = agreement_metrics(np.arange(10.0), np.arange(10.0))
    assert classify_agreement(metrics, 0.999, 0.05, 0.9, 0.5) == "matches"
    approximate = {"pearson_r": 0.95, "normalized_rmse": 0.2}
    assert classify_agreement(approximate, 0.999, 0.05, 0.9, 0.5) == "approximately_agrees"
    poor = {"pearson_r": 0.4, "normalized_rmse": 1.2}
    assert classify_agreement(poor, 0.999, 0.05, 0.9, 0.5) == "does_not_agree"
