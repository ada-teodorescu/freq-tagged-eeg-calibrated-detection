from __future__ import annotations

import numpy as np

from eeg_core.spectral import _hotelling_vectorized, hotelling_scores, phase_features


def same_frequency_scores(
    target:np.ndarray,subsets:np.ndarray,surrogates:int,seed_sequence:np.random.SeedSequence,
)->dict[str,np.ndarray]:
    """Target scores and same-frequency phase-randomized nulls.

    Rotations are shared across channels for each trial/harmonic, preserving the
    instantaneous spatial pattern while destroying repeat-to-repeat phase locking.
    """
    values=np.asarray(target); selected=values[subsets]; draws,k,channels,harmonics=selected.shape
    positive_phase=phase_features(values,subsets); positive_hotelling=hotelling_scores(values[...,0],subsets)
    null_phase=np.empty((draws,surrogates,harmonics),dtype=np.float32)
    null_hotelling=np.empty((draws,surrogates),dtype=np.float32)
    children=seed_sequence.spawn(surrogates)
    for surrogate,child in enumerate(children):
        rng=np.random.default_rng(child)
        rotations=np.exp(1j*rng.uniform(-np.pi,np.pi,size=(draws,k,1,harmonics)))
        randomized=selected*rotations
        units=np.divide(randomized,np.abs(randomized),out=np.zeros_like(randomized),where=np.abs(randomized)>0)
        null_phase[:,surrogate]=np.mean(np.abs(units.mean(axis=1))**2,axis=1)
        null_hotelling[:,surrogate]=_hotelling_vectorized(randomized[...,0]).mean(axis=1)
    return {"positive_phase":positive_phase.astype(np.float32),"null_phase":null_phase,
            "positive_hotelling":positive_hotelling.astype(np.float32),"null_hotelling":null_hotelling}
