from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import linalg


@dataclass(frozen=True)
class SpatialModel:
    method: str
    weights: np.ndarray
    metadata: dict


def apply_spatial(data: np.ndarray, model: SpatialModel) -> np.ndarray:
    """Project arrays whose penultimate axis is channel."""
    return np.einsum("...ct,c->...t", np.asarray(data), model.weights)


def best_channel(training_scores: np.ndarray) -> SpatialModel:
    scores = np.asarray(training_scores)
    channel = int(np.nanargmax(np.mean(scores, axis=tuple(range(scores.ndim - 1)))))
    weights = np.zeros(scores.shape[-1]); weights[channel] = 1.0
    return SpatialModel("S1", weights, {"selected_channel": channel, "oracle": False})


def _regularized_covariance(data: np.ndarray, regularization: float) -> np.ndarray:
    covariance = np.cov(np.asarray(data), rowvar=False)
    target = np.trace(covariance) / covariance.shape[0] * np.eye(covariance.shape[0])
    return (1 - regularization) * covariance + regularization * target


def fit_ress(training: np.ndarray, frequency: float, sampling_rate: int, regularization: float = 0.1) -> SpatialModel:
    """Fit RESS on training-only trial x channel x time data."""
    values = np.asarray(training, dtype=float)
    time = np.arange(values.shape[-1]) / sampling_rate
    reference = np.exp(-2j * np.pi * frequency * time)
    Fourier = np.einsum("nct,t->nc", values, reference)
    signal_cov = _regularized_covariance(np.vstack((Fourier.real, Fourier.imag)), regularization)
    broadband = np.moveaxis(values, 1, -1).reshape(-1, values.shape[1])
    reference_cov = _regularized_covariance(broadband, regularization)
    eigenvalues, eigenvectors = linalg.eigh(signal_cov, reference_cov)
    weights = eigenvectors[:, np.argmax(eigenvalues)]
    weights /= np.linalg.norm(weights)
    return SpatialModel("S2", weights, {"frequency_hz": frequency, "training_trials": values.shape[0]})


def fit_dss(training: np.ndarray, regularization: float = 0.1) -> SpatialModel:
    """Fit DSS using repeatable trial-average activity as the bias function."""
    values = np.asarray(training, dtype=float)
    broadband = np.moveaxis(values, 1, -1).reshape(-1, values.shape[1])
    total_cov = _regularized_covariance(broadband, regularization)
    evoked = values.mean(axis=0).T
    repeatable_cov = _regularized_covariance(evoked, regularization)
    eigenvalues, eigenvectors = linalg.eigh(repeatable_cov, total_cov)
    weights = eigenvectors[:, np.argmax(eigenvalues)]
    weights /= np.linalg.norm(weights)
    return SpatialModel("S3", weights, {"training_trials": values.shape[0]})


def lagged_stimulus(stimulus: np.ndarray, lags_samples: tuple[int, ...]) -> np.ndarray:
    signal = np.asarray(stimulus, dtype=float)
    columns = []
    for lag in lags_samples:
        shifted = np.zeros_like(signal)
        if lag == 0: shifted[:] = signal
        elif lag > 0: shifted[lag:] = signal[:-lag]
        else: shifted[:lag] = signal[-lag:]
        columns.append(shifted)
    return np.column_stack(columns)


def fit_regularized_cca(
    training: np.ndarray, stimulus_design: np.ndarray, regularization: float = 0.1,
) -> SpatialModel:
    """Fit the EEG side of ridge CCA using training observations only."""
    eeg = np.asarray(training, dtype=float)
    stim = np.asarray(stimulus_design, dtype=float)
    eeg -= eeg.mean(axis=0); stim -= stim.mean(axis=0)
    cxx = eeg.T @ eeg / max(len(eeg) - 1, 1) + regularization * np.eye(eeg.shape[1])
    cyy = stim.T @ stim / max(len(stim) - 1, 1) + regularization * np.eye(stim.shape[1])
    cxy = eeg.T @ stim / max(len(eeg) - 1, 1)
    matrix = np.linalg.pinv(cxx) @ cxy @ np.linalg.pinv(cyy) @ cxy.T
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    weights = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])
    weights /= np.linalg.norm(weights)
    return SpatialModel("S4", weights, {"canonical_correlation_squared": float(np.max(np.real(eigenvalues)))})
