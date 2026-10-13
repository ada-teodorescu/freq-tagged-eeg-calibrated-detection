from __future__ import annotations

import numpy as np

from cucu_advanced_study.algorithms import phase_trajectories, temporal_shift_multipliers


def rotations(config, participant_index: int, stream_index: int, seed_salt: int = 0) -> np.ndarray:
    blocks = []
    for seed in config.surrogate_seeds:
        sequence = np.random.SeedSequence([seed, participant_index, stream_index, seed_salt])
        blocks.append(temporal_shift_multipliers(config.trials_per_stream,
            tuple(range(1, len(config.harmonics_hz) + 1)), config.surrogates_per_seed, sequence))
    return np.concatenate(blocks, axis=0)


def condition_features(coefficients: np.ndarray, config, regions: tuple[np.ndarray, ...],
                       participant_index: int, k_values: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return scaled positive/null regional features at requested trial prefixes."""
    _, positive = phase_trajectories(coefficients, k_values, regions)
    null = np.empty((config.streams, config.surrogates, len(k_values), len(regions),
                     len(config.harmonics_hz)), dtype=np.float32)
    for stream in range(config.streams):
        _, regional = phase_trajectories(coefficients[stream:stream + 1], k_values, regions,
                                         rotations(config, participant_index, stream))
        null[stream] = regional[0]
    c, s, looks = config.conditions, config.streams_per_condition, len(k_values)
    positive = positive.reshape(c, s, looks, len(regions), -1).mean(axis=1)
    null = null.reshape(c, s, config.surrogates, looks, len(regions), -1).mean(axis=1)
    scale = 2.0 * np.asarray(k_values, dtype=float)
    positive = positive * scale[None, :, None, None]
    null = null * scale[None, None, :, None, None]
    return positive.reshape(c, looks, -1).astype(np.float32), null.reshape(c, config.surrogates, looks, -1).astype(np.float32)
