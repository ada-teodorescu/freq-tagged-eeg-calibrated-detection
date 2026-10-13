"""Crop before every restricted operation; padding never reads outside the crop."""
from __future__ import annotations

import numpy as np
from scipy import signal


def restricted_hilbert(
    trials: np.ndarray,
    sampling_rate: int,
    frequencies: tuple[float, ...],
    *,
    start: int = 500,
    stop: int = 4500,
    padding: int = 500,
    reflection: str = "odd",
    sos_padlen: int = 27,
    order: int = 4,
    half_bandwidth: float = 0.5,
) -> np.ndarray:
    """Estimate over [start, stop), with only synthetic surrounding context.

    1. Crop recorded EEG before padding, filtering, or analytic transformation.
    2. Reflect the cropped signal (odd primary, even sensitivity).
    3. Apply a Butterworth SOS bandpass forward/backward, explicit odd internal
       extension and padlen; scipy's endpoint-scaled steady-state initialization.
    4. FFT Hilbert transform over exactly the padded buffer (periodic FFT boundary,
       no extra zero padding), discard synthetic margins, and demodulate at the
       original sample coordinates. All retained samples contribute equally.
    """
    raw = np.asarray(trials)
    if raw.ndim < 1 or not 0 <= start < stop <= raw.shape[-1]:
        raise ValueError("Invalid retained interval")
    if reflection not in ("odd", "even") or not 0 < padding < stop - start:
        raise ValueError("Use odd/even reflection and padding shorter than the crop")
    if not 0 <= sos_padlen < stop - start + 2 * padding:
        raise ValueError("Invalid SOS pad length")
    values = np.asarray(raw[..., start:stop], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite retained EEG")
    widths = [(0, 0)] * values.ndim
    widths[-1] = (padding, padding)
    padded = np.pad(values, widths, mode="reflect", reflect_type=reflection)
    sample = np.arange(start, stop, dtype=float)
    output = []
    for frequency in frequencies:
        sos = signal.butter(order, [frequency - half_bandwidth, frequency + half_bandwidth],
                            btype="bandpass", fs=sampling_rate, output="sos")
        filtered = signal.sosfiltfilt(sos, padded, axis=-1, padtype="odd", padlen=sos_padlen)
        analytic = signal.hilbert(filtered, N=padded.shape[-1], axis=-1)
        retained = analytic[..., padding:padding + stop - start]
        demodulator = np.exp(-2j * np.pi * frequency * sample / sampling_rate)
        output.append(np.mean(retained * demodulator, axis=-1))
    return np.stack(output, axis=-1)
