from __future__ import annotations

import json
import numpy as np
import pandas as pd

from cucu_replication.io import inspect_set, natural_subject_key
from ..runtime import RunContext, atomic_write_json


def run(context:RunContext,force:bool=False)->tuple[list,dict]:
    del force; stage="00_diagnosis"; context.start_stage(stage,4); config=context.config
    required=[config.baseline_article_run/"claim_status.json",config.baseline_article_run/"tables"/"algorithm_summary.csv",
              config.baseline_article_run/"tables"/"learned_models.csv"]
    missing=[str(path) for path in required if not path.exists()]
    if missing: raise FileNotFoundError("Missing baseline article artifacts: "+", ".join(missing))
    context.progress(stage,1,4,"baseline artifacts")
    summary=pd.read_csv(required[1]); models=pd.read_csv(required[2])
    short=summary[(summary.algorithm=="A4_LearnedFusion")&(summary.k==10)&(summary.duration_seconds<=2)]
    short_models=models[(models.k==10)&(models.duration_seconds<=2)]
    coefficient_columns=[f"coefficient_{int(f)}hz" for f in config.harmonics_hz]
    negative_counts={column:int((short_models[column]<0).sum()) for column in coefficient_columns}
    calibration=pd.read_csv(config.baseline_article_run/"tables"/"heldout_calibration.csv")
    primary=calibration[(calibration.k==10)&np.isclose(calibration.duration_seconds,4.5)]
    participant_range=[float(primary.heldout_fpr.min()),float(primary.heldout_fpr.max())]
    context.progress(stage,2,4,"baseline failures quantified")
    set_paths=sorted((config.data_root/"Pre-processed EEGLAB datasets").glob("*.set"),key=natural_subject_key)
    records=[inspect_set(path) for path in set_paths]
    if len(records)!=25 or any((r.n_channels,r.n_times,r.n_trials)!=(32,5000,450) for r in records):
        raise AssertionError("Public EEG structure failed")
    context.progress(stage,3,4,"public EEG verified")
    diagnosis={
        "baseline_analysis_id":json.loads((config.baseline_article_run/"claim_status.json").read_text())["analysis_id"],
        "failures":[
            {"id":"short_window_neighbor_null","evidence":"A4 high detection with negative harmonic coefficients at T<=2 s",
             "repair":"same-frequency phase-randomized nulls"},
            {"id":"unconstrained_direction","evidence":negative_counts,"repair":"non-negative logistic coefficients"},
            {"id":"participant_threshold_shift","evidence":{"primary_participant_fpr_range":participant_range},
             "repair":"leave-one-condition-out participant-null calibration sensitivity"},
            {"id":"hotelling_no_gain","evidence":"A2 approximately agrees with A1 at the locked endpoint",
             "repair":"retain as analytic diagnostic; do not optimize after outcomes"},
        ],
        "locked_methods":["B1_ITPC","B2_Hotelling","B3_EqualHarmonic","B4_CVSubsetFusion","B5_MonotoneFusion"],
        "harmonic_subset_candidates":[list(value) for value in config.harmonic_subsets],
        "primary_endpoint":{"k":10,"duration_seconds":4.5},"manuscript_outputs":False,
        "participants":[record.participant for record in records],
    }
    path=context.run_dir/"diagnosis_lock.json"; atomic_write_json(path,diagnosis); context.progress(stage,4,4,"repair locked")
    context.finish_stage(stage,[path],diagnosis); return [path],diagnosis
