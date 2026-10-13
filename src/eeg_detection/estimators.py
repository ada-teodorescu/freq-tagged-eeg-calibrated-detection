from __future__ import annotations

import numpy as np
from scipy import signal


def published_hann_dft(
    trials: np.ndarray, sampling_rate: int, frequencies_hz: np.ndarray,
    crop_samples: int = 500,
) -> np.ndarray:
    values = np.asarray(trials, dtype=np.float64)[..., crop_samples:]
    window = np.hanning(values.shape[-1])
    bins = np.rint(np.asarray(frequencies_hz) * values.shape[-1] / sampling_rate).astype(int)
    return np.fft.rfft(values * window, axis=-1)[..., bins]


def hilbert_coefficients(
    trials: np.ndarray, sampling_rate: int, frequencies_hz: np.ndarray,
    order: int = 4, half_bandwidth_hz: float = 0.5, edge_crop_seconds: float = 0.5,
) -> np.ndarray:
    """Bandpass/Hilbert coefficients after complex demodulation to baseband."""
    values = np.asarray(trials, dtype=np.float64)
    crop = int(round(edge_crop_seconds * sampling_rate))
    coefficients = []
    for frequency in np.asarray(frequencies_hz):
        sos = signal.butter(order, [frequency - half_bandwidth_hz, frequency + half_bandwidth_hz],
                            btype="bandpass", fs=sampling_rate, output="sos")
        analytic = signal.hilbert(signal.sosfiltfilt(sos, values, axis=-1), axis=-1)
        if crop:
            analytic = analytic[..., crop:-crop]
            sample_indices = np.arange(crop, values.shape[-1] - crop)
        else:
            sample_indices = np.arange(values.shape[-1])
        demodulator = np.exp(-2j * np.pi * frequency * sample_indices / sampling_rate)
        coefficients.append(np.mean(analytic * demodulator, axis=-1))
    return np.stack(coefficients, axis=-1)


def least_squares_coefficients(
    trials: np.ndarray, sampling_rate: int, frequencies_hz: np.ndarray, crop_samples: int = 500,
) -> np.ndarray:
    values = np.asarray(trials, dtype=np.float64)[..., crop_samples:]
    n_times = values.shape[-1]
    time = np.arange(n_times) / sampling_rate
    window_root = np.sqrt(np.hanning(n_times))
    flattened = values.reshape(-1, n_times) * window_root
    result = []
    for frequency in np.asarray(frequencies_hz):
        design = np.column_stack((np.ones(n_times), np.cos(2 * np.pi * frequency * time),
                                  np.sin(2 * np.pi * frequency * time))) * window_root[:, None]
        beta = np.linalg.pinv(design) @ flattened.T
        result.append((beta[1] - 1j * beta[2]).reshape(values.shape[:-1]))
    return np.stack(result, axis=-1)


def dpss_multitaper_coefficients(
    trials: np.ndarray, sampling_rate: int, frequencies_hz: np.ndarray,
    crop_samples: int = 500, time_bandwidth: float = 2.5, tapers: int = 4,
) -> np.ndarray:
    values = np.asarray(trials, dtype=np.float64)[..., crop_samples:]
    windows = signal.windows.dpss(
        values.shape[-1], time_bandwidth, Kmax=tapers, sym=False, norm=2
    )
    bins = np.rint(np.asarray(frequencies_hz) * values.shape[-1] / sampling_rate).astype(int)
    spectra = np.fft.rfft(values[..., None, :] * windows, axis=-1)[..., bins]
    # input-leading axes x taper x frequency -> input-leading axes x frequency x taper
    return np.swapaxes(spectra, -1, -2)


def thomson_line_from_spectra(
    spectra: np.ndarray, taper_response: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a complex sinusoidal line to DPSS tapered Fourier spectra.

    ``spectra`` must end in the taper dimension. The returned line estimate and
    F statistic have that final dimension removed.
    """
    values = np.asarray(spectra, dtype=np.complex128)
    response = np.asarray(taper_response, dtype=np.float64)
    if values.shape[-1] != response.size:
        raise ValueError("spectra and taper_response have incompatible shapes")
    response_power = float(np.sum(np.square(response)))
    if response_power <= np.finfo(float).tiny:
        raise ValueError("DPSS tapers have zero aggregate line response")
    line = np.einsum("...k,k->...", values, response, optimize=True) / response_power
    residual = values - line[..., None] * response
    residual_power = np.sum(np.abs(residual) ** 2, axis=-1)
    numerator = (response.size - 1.0) * np.abs(line) ** 2 * response_power
    f_statistic = np.divide(
        numerator,
        residual_power,
        out=np.full_like(numerator, np.inf, dtype=float),
        where=residual_power > np.finfo(float).tiny,
    )
    return line, f_statistic


def dpss_line_estimates(
    trials: np.ndarray,
    sampling_rate: int,
    frequencies_hz: np.ndarray,
    crop_samples: int = 500,
    time_bandwidth: float = 2.5,
    tapers: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Thomson line coefficients, harmonic F statistics, and DPSS ratios.

    Tapered Fourier coefficients are regressed on each taper's response at zero
    offset. The F statistic has nominal degrees of freedom (2, 2*K-2).
    """
    values = np.asarray(trials, dtype=np.float64)[..., crop_samples:]
    n_times = values.shape[-1]
    windows, ratios = signal.windows.dpss(
        n_times,
        time_bandwidth,
        Kmax=tapers,
        sym=False,
        norm=2,
        return_ratios=True,
    )
    time = np.arange(n_times, dtype=float) / sampling_rate
    basis = np.exp(-2j * np.pi * np.asarray(frequencies_hz, dtype=float)[:, None] * time)
    tapered = values[..., None, :] * windows
    spectra = np.einsum("...kt,ft->...fk", tapered, basis, optimize=True)
    line, f_statistic = thomson_line_from_spectra(spectra, windows.sum(axis=-1))
    return line, f_statistic, np.asarray(ratios, dtype=float)
