from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from ..algorithms import METHODS, phase_rotations, score_trajectories
from ..runtime import RunContext, atomic_save_npz
from .coefficients import coefficient_path


def trajectory_path(context: RunContext, participant: str) -> Path:
    return context.checkpoint_dir / "03_null_trajectories" / f"{participant}.npz"


def build_rotations(config, participant_index: int, frequency_index: int, blocks: int,
                    seed_salt: int = 0) -> np.ndarray:
    return np.stack([phase_rotations(config.surrogates_per_block, config.trials_per_block,
                                    np.random.SeedSequence([config.seed, participant_index, frequency_index, block, seed_salt]))
                     for block in range(blocks)])


def _participant(context: RunContext, participant: str, participant_index: int, force: bool) -> tuple[str, Path]:
    config = context.config
    destination = trajectory_path(context, participant)
    if context.checkpoint_valid(destination) and not force:
        return participant, destination
    with np.load(coefficient_path(context, participant), allow_pickle=False) as archive:
        coefficients = archive["coefficients"]
    max_blocks = max(limit // config.trials_per_block for limit in config.standards_per_frequency)
    positive = np.full((len(METHODS), len(config.frequencies_hz), max_blocks, len(config.k_values)), np.nan, np.float32)
    null = np.full((len(METHODS), len(config.frequencies_hz), max_blocks,
                    config.surrogates_per_block, len(config.k_values)), np.nan, np.float32)
    for f_index, frequency in enumerate(config.frequencies_hz):
        limit = config.standard_limit(frequency)
        blocks = limit // config.trials_per_block
        rotations = build_rotations(config, participant_index, f_index, blocks)
        for m_index, _ in enumerate(METHODS):
            values = coefficients[m_index, f_index, :limit].reshape(blocks, config.trials_per_block, 64)
            positive[m_index, f_index, :blocks] = score_trajectories(values, config.k_values)
            null[m_index, f_index, :blocks] = score_trajectories(values, config.k_values, rotations)
    atomic_save_npz(destination, positive=positive, null=null, participant=np.asarray(participant),
                    methods=np.asarray(METHODS), frequencies_hz=np.asarray(config.frequencies_hz),
                    k_values=np.asarray(config.k_values))
    source = coefficient_path(context, participant)
    context.mark_checkpoint(destination, {"participant": participant,
                                          "input_fingerprint": sha256(source.read_bytes()).hexdigest(),
                                          "shape_positive": list(positive.shape),
                                          "shape_null": list(null.shape), "surrogates": config.surrogates_per_block,
                                          "rotation_contract": "shared_over_channels_and_estimators"})
    return participant, destination


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage, config = "03_null_trajectories", context.config
    participants = tuple(pd.read_csv(context.table_dir / "recording_inventory.csv")["participant"].astype(str))
    context.start_stage(stage, len(participants)); outputs = []
    if context.workers == 1:
        for index, participant in enumerate(participants, 1):
            _, path = _participant(context, participant, index - 1, force)
            outputs.append(path); context.progress(stage, index, len(participants), participant)
    else:
        with ThreadPoolExecutor(max_workers=context.workers) as executor:
            futures = [executor.submit(_participant, context, participant, index, force)
                       for index, participant in enumerate(participants)]
            for completed, future in enumerate(as_completed(futures), 1):
                participant, path = future.result(); outputs.append(path)
                context.progress(stage, completed, len(participants), participant)
    summary = {"participants": len(participants), "surrogates_per_block": config.surrogates_per_block,
               "score": "2K_times_equal_channel_ITPC_squared"}
    context.finish_stage(stage, outputs, summary)
    return outputs, summary
