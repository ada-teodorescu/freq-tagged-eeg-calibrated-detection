from __future__ import annotations

import numpy as np
from scipy import signal, stats

from cucu_detection.evaluation import empirical_threshold


def rectangular_ls_coefficients(trials: np.ndarray, sampling_rate: int,
                                frequencies_hz: np.ndarray, start_sample: int,
                                stop_sample: int) -> np.ndarray:
    """Unweighted sinusoidal least-squares coefficients with an intercept."""
    values = np.asarray(trials, dtype=np.float64)[..., start_sample:stop_sample]
    time = np.arange(start_sample, stop_sample, dtype=float) / sampling_rate
    flat = values.reshape(-1, values.shape[-1])
    output = []
    for frequency in np.asarray(frequencies_hz, dtype=float):
        design = np.column_stack((np.ones(len(time)), np.cos(2 * np.pi * frequency * time),
                                  np.sin(2 * np.pi * frequency * time)))
        beta = np.linalg.pinv(design) @ flat.T
        output.append((beta[1] - 1j * beta[2]).reshape(values.shape[:-1]))
    return np.stack(output, axis=-1)


def filtered_ls_coefficients(trials: np.ndarray, sampling_rate: int,
                             frequencies_hz: np.ndarray, start_sample: int,
                             stop_sample: int, order: int, half_bandwidth_hz: float) -> np.ndarray:
    """Center-frequency LS after the same zero-phase narrowband filter as Hilbert."""
    values = np.asarray(trials, dtype=np.float64)
    output = []
    for frequency in np.asarray(frequencies_hz, dtype=float):
        sos = signal.butter(order, [frequency - half_bandwidth_hz, frequency + half_bandwidth_hz],
                            btype="bandpass", fs=sampling_rate, output="sos")
        filtered = signal.sosfiltfilt(sos, values, axis=-1)
        coefficient = rectangular_ls_coefficients(filtered, sampling_rate, np.asarray([frequency]),
                                                   start_sample, stop_sample)[..., 0]
        output.append(coefficient)
    return np.stack(output, axis=-1)


def linear_effective_kernel(estimator, n_times: int, chunk_size: int = 64) -> np.ndarray:
    """Recover a real-input complex linear functional by chunked impulses."""
    if n_times < 2 or chunk_size < 1: raise ValueError("invalid kernel dimensions")
    kernel = np.empty(n_times, dtype=np.complex128)
    for start in range(0, n_times, chunk_size):
        stop = min(n_times, start + chunk_size)
        impulses = np.zeros((stop - start, n_times), dtype=np.float64)
        impulses[np.arange(stop - start), np.arange(start, stop)] = 1.0
        values = np.asarray(estimator(impulses)).reshape(stop - start)
        kernel[start:stop] = values
    return kernel


def kernel_response(kernel: np.ndarray, sampling_rate: int, n_fft: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(kernel, dtype=np.complex128)
    response = np.fft.fftshift(np.fft.fft(values, n=max(n_fft, len(values))))
    frequencies = np.fft.fftshift(np.fft.fftfreq(len(response), 1.0 / sampling_rate))
    return frequencies, np.abs(response)


def kernel_metrics(kernel: np.ndarray, sampling_rate: int, target_frequency: float,
                   retained_start: int, retained_stop: int) -> dict[str, float]:
    frequencies, magnitude = kernel_response(kernel, sampling_rate)
    candidates = [int(np.argmin(np.abs(frequencies - target_frequency))),
                  int(np.argmin(np.abs(frequencies + target_frequency)))]
    peak_index = max(candidates, key=lambda index: magnitude[index])
    peak = float(magnitude[peak_index])
    threshold = peak / np.sqrt(2.0)
    left = peak_index
    while left > 0 and magnitude[left] >= threshold: left -= 1
    right = peak_index
    while right + 1 < len(magnitude) and magnitude[right] >= threshold: right += 1
    if left < peak_index and magnitude[left + 1] != magnitude[left]:
        left_crossing = frequencies[left] + ((threshold - magnitude[left]) /
                                              (magnitude[left + 1] - magnitude[left])) * (frequencies[left + 1] - frequencies[left])
    else:
        left_crossing = frequencies[left]
    if right + 1 < len(magnitude) and magnitude[right + 1] != magnitude[right]:
        right_crossing = frequencies[right] + ((threshold - magnitude[right]) /
                                                (magnitude[right + 1] - magnitude[right])) * (frequencies[right + 1] - frequencies[right])
    else:
        right_crossing = frequencies[right]
    power = np.square(magnitude)
    enbw = float(np.trapezoid(power, frequencies) / max(peak * peak, np.finfo(float).tiny))
    energy = np.abs(kernel) ** 2
    outside = energy.copy(); outside[retained_start:retained_stop] = 0
    return {"peak_gain": peak, "peak_frequency_hz": float(frequencies[peak_index]),
            "bandwidth_3db_hz": float(right_crossing - left_crossing),
            "equivalent_noise_bandwidth_hz": enbw,
            "outside_retained_energy_fraction": float(outside.sum() / max(energy.sum(), np.finfo(float).tiny))}


def causal_settling_samples(sampling_rate: int, order: int,
                            half_bandwidth_hz: float, fraction: float = 0.99,
                            maximum_seconds: float = 5.5) -> int:
    """Settling time of the causal low-pass used after complex demodulation."""
    sos = signal.butter(order, half_bandwidth_hz, btype="lowpass", fs=sampling_rate, output="sos")
    impulse = np.zeros(int(round(maximum_seconds * sampling_rate))); impulse[0] = 1.0
    response = signal.sosfilt(sos, impulse)
    cumulative = np.cumsum(response ** 2)
    return int(np.searchsorted(cumulative, fraction * cumulative[-1]))


def causal_demodulation_coefficients(trials: np.ndarray, sampling_rate: int,
                                     frequencies_hz: np.ndarray, order: int,
                                     half_bandwidth_hz: float, settling_samples: int) -> np.ndarray:
    """Causal I/Q demodulation followed by a causal low-pass and fixed burn-in."""
    values = np.asarray(trials, dtype=np.float64)
    if not 0 <= settling_samples < values.shape[-1]: raise ValueError("invalid causal settling interval")
    time = np.arange(values.shape[-1], dtype=float) / sampling_rate
    sos = signal.butter(order, half_bandwidth_hz, btype="lowpass", fs=sampling_rate, output="sos")
    outputs = []
    for frequency in np.asarray(frequencies_hz, dtype=float):
        demodulated = values * np.exp(-2j * np.pi * frequency * time)
        filtered = signal.sosfilt(sos, demodulated, axis=-1)
        outputs.append(filtered[..., settling_samples:].mean(axis=-1))
    return np.stack(outputs, axis=-1)


def crossed_bootstrap_draws(values: np.ndarray, iterations: int, seed: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2: raise ValueError("values must be participant x condition")
    rng = np.random.default_rng(seed); draws = np.empty(iterations)
    for index in range(iterations):
        participants = rng.integers(0, matrix.shape[0], matrix.shape[0])
        conditions = rng.integers(0, matrix.shape[1], matrix.shape[1])
        draws[index] = matrix[np.ix_(participants, conditions)].mean()
    return draws


def simultaneous_crossed_band(gains: np.ndarray, iterations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(gains, dtype=float)
    if values.ndim != 3: raise ValueError("gains must be participant x condition x look")
    estimate = values.mean(axis=(0, 1)); rng = np.random.default_rng(seed)
    draws = np.empty((iterations, values.shape[-1]))
    for index in range(iterations):
        participants = rng.integers(0, values.shape[0], values.shape[0])
        conditions = rng.integers(0, values.shape[1], values.shape[1])
        draws[index] = values[np.ix_(participants, conditions)].mean(axis=(0, 1))
    errors = np.max(np.abs(draws - estimate), axis=1)
    radius = float(np.quantile(errors, 0.95))
    return estimate - radius, estimate + radius


def paired_sign_flip(differences: np.ndarray, iterations: int, seed: int) -> float:
    values = np.asarray(differences, dtype=float).reshape(-1)
    observed = abs(float(values.mean())); rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(iterations):
        exceed += abs(float(np.mean(values * rng.choice((-1.0, 1.0), len(values))))) >= observed
    return float((exceed + 1) / (iterations + 1))


def mcnemar_exact(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    a, b = np.asarray(first, bool), np.asarray(second, bool)
    first_only = int(np.sum(a & ~b)); second_only = int(np.sum(~a & b)); discordant = first_only + second_only
    p = 1.0 if discordant == 0 else float(stats.binomtest(min(first_only, second_only), discordant, 0.5).pvalue)
    return {"first_only": first_only, "second_only": second_only, "discordant": discordant, "p_value": p}


def equal_weight_score(features: np.ndarray, harmonics: int = 4) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    if values.shape[-1] % 4: raise ValueError("features must contain four harmonics per region")
    reshaped = values.reshape(*values.shape[:-1], values.shape[-1] // 4, 4)
    if not 1 <= harmonics <= 4: raise ValueError("harmonics must lie in 1..4")
    return reshaped[..., :harmonics].mean(axis=(-2, -1))


def simple_threshold(train_null_features: np.ndarray, target_fpr: float, harmonics: int = 4) -> float:
    return empirical_threshold(equal_weight_score(train_null_features, harmonics), target_fpr)
