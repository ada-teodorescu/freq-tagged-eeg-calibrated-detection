from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import linalg

from eeg_detection.spatial import SpatialModel


def _reproducibility_matrices(training_by_stream: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return (Q_raw, S) for grouped TRCA.

    ``Q_raw`` is the sum of per-trial channel Gram matrices, pooled across ALL
    training trials regardless of stream (a variance denominator, for which
    pooling across streams is fine). ``S`` is the reproducibility numerator,
    restricted to WITHIN-stream cross-trial products only: trials from
    different streams (different syllable sequences) must never contribute a
    cross product to S, or the filter stops measuring genuine repeat-to-repeat
    reproducibility of one stimulus.
    """
    if not training_by_stream:
        raise ValueError("at least one training stream is required")
    channels = {int(np.asarray(group).shape[1]) for group in training_by_stream}
    if len(channels) != 1:
        raise ValueError("all training streams must share the same channel count")
    q_raw = None
    s = None
    for group in training_by_stream:
        values = np.asarray(group, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("each training stream must be trial x channel x time")
        if values.shape[0] < 2:
            raise ValueError("every training stream needs at least two trials")
        centered = values - values.mean(axis=-1, keepdims=True)
        trial_gram = np.einsum("nct,ndt->cd", centered, centered, optimize=True)
        total = centered.sum(axis=0)
        stream_s = total @ total.T - trial_gram
        q_raw = trial_gram if q_raw is None else q_raw + trial_gram
        s = stream_s if s is None else s + stream_s
    return q_raw, s


def fit_trca(training_by_stream: Sequence[np.ndarray], regularization: float = 0.1) -> SpatialModel:
    """Fit grouped TRCA on training-only, stream-grouped trial x channel x time data.

    Unlike ``eeg_detection.spatial``'s ``fit_ress``/``fit_dss``/``fit_regularized_cca``,
    which each take one flat pooled array, TRCA needs stream identity preserved: it
    maximizes the ratio of between-trial reproducibility *within a repeated stimulus*
    to total per-trial variance, so ``training_by_stream`` is a sequence of one array
    per training stream rather than a single concatenated array.

    Regularization is applied only to the variance denominator ``Q``, never to the
    reproducibility numerator ``S`` -- shrinking ``S`` would contaminate the
    cross-trial reproducibility measure with the denominator's shrinkage target.
    """
    if not 0.0 <= regularization <= 1.0:
        raise ValueError("regularization must lie in [0, 1]")
    q_raw, s = _reproducibility_matrices(training_by_stream)
    target = np.trace(q_raw) / q_raw.shape[0] * np.eye(q_raw.shape[0])
    q_regularized = (1.0 - regularization) * q_raw + regularization * target
    eigenvalues, eigenvectors = linalg.eigh(s, q_regularized)
    weights = eigenvectors[:, np.argmax(eigenvalues)]
    weights = weights / np.linalg.norm(weights)
    pivot = int(np.argmax(np.abs(weights)))
    if weights[pivot] < 0:
        weights = -weights
    return SpatialModel(
        "TRCA_grouped",
        weights,
        {
            "training_streams": len(training_by_stream),
            "training_trials": int(sum(np.asarray(group).shape[0] for group in training_by_stream)),
            "regularization": float(regularization),
        },
    )
