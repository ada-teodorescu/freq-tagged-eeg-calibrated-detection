from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json
import numpy as np

from cucu_article.spectral import hann_coefficients
from cucu_detection.detectors import deterministic_subsets
from cucu_replication.io import inspect_set,load_stream_trials,natural_subject_key
from ..runtime import RunContext,atomic_save_npz,atomic_write_json
from ..surrogates import same_frequency_scores


def feature_path(context:RunContext,participant:str,duration:float,k:int)->Path:
    return context.checkpoint_dir/"01_surrogate_features"/participant/f"duration_{str(duration).replace('.','p')}_k{k:02d}.npz"


def _participant(context:RunContext,set_path:Path,participant_index:int,force:bool)->tuple[str,list[Path]]:
    config=context.config; epoch=inspect_set(set_path)
    paths=[feature_path(context,epoch.participant,duration,k) for duration in config.durations_seconds for k in config.k_values]
    if not force and all(context.checkpoint_valid(path) for path in paths): return epoch.participant,paths
    subsets={k:deterministic_subsets(config.trials_per_stream,k,config.subset_cap,config.seed) for k in config.k_values}
    buffers={}
    for duration in config.durations_seconds:
        for k in config.k_values:
            draws=len(subsets[k]); buffers[(duration,k)]={
                "positive_phase":np.empty((config.streams,draws,4),np.float32),
                "null_phase":np.empty((config.streams,draws,config.surrogates_per_subset,4),np.float32),
                "positive_hotelling":np.empty((config.streams,draws),np.float32),
                "null_hotelling":np.empty((config.streams,draws,config.surrogates_per_subset),np.float32)}
    for stream in range(1,config.streams+1):
        trials=load_stream_trials(epoch,stream)[...,config.onset_crop_samples:]
        for duration_index,duration in enumerate(config.durations_seconds):
            target,_=hann_coefficients(trials,config.sampling_rate,duration,config.fft_pad_samples,config.harmonics_hz,18)
            for k in config.k_values:
                scores=same_frequency_scores(target,subsets[k],config.surrogates_per_subset,
                    np.random.SeedSequence([config.seed,participant_index,stream,duration_index,k]))
                for name,values in scores.items(): buffers[(duration,k)][name][stream-1]=values
    for (duration,k),arrays in buffers.items():
        path=feature_path(context,epoch.participant,duration,k)
        atomic_save_npz(path,**arrays,subsets=subsets[k],duration_seconds=np.asarray(duration),k=np.asarray(k))
        context.mark_checkpoint(path,{"participant":epoch.participant,"duration_seconds":duration,"k":k,
            "null":"same-frequency independent trial-phase rotations shared across channels","surrogates":config.surrogates_per_subset})
    return epoch.participant,paths


def run(context:RunContext,force:bool=False)->tuple[list,dict]:
    stage="01_surrogate_features"; config=context.config
    set_paths=sorted((config.data_root/"Pre-processed EEGLAB datasets").glob("*.set"),key=natural_subject_key)
    context.start_stage(stage,len(set_paths)); outputs=[]
    if context.workers==1:
        for index,path in enumerate(set_paths):
            participant,files=_participant(context,path,index,force); outputs.extend(files); context.progress(stage,index+1,len(set_paths),participant)
    else:
        with ThreadPoolExecutor(max_workers=context.workers) as executor:
            futures={executor.submit(_participant,context,path,index,force):path for index,path in enumerate(set_paths)}
            for completed,future in enumerate(as_completed(futures),1):
                participant,files=future.result(); outputs.extend(files); context.progress(stage,completed,len(set_paths),participant)
    summary={"participants":len(set_paths),"feature_files":len(outputs),"null":"same_frequency_phase_randomization",
             "surrogates_per_subset":config.surrogates_per_subset,"spatial_phase_preserved":True}
    report=context.run_dir/"surrogate_features.json"; atomic_write_json(report,summary); outputs.append(report)
    context.finish_stage(stage,outputs,summary); return outputs,summary
