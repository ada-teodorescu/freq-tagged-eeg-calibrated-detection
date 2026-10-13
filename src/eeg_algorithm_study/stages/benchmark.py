from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd

from eeg_core.stages.benchmark import auc_fast,empirical_threshold_fast,leave_one_participant_thresholds
from eeg_detection.evaluation import binomial_interval
from ..algorithms import fit_monotone_logistic,select_harmonic_subset,select_monotone_regularization,training_observations
from ..runtime import RunContext,atomic_write_json
from .features import feature_path


ALGORITHMS=("B1_ITPC","B2_Hotelling","B3_EqualHarmonic","B4_CVSubsetFusion","B5_MonotoneFusion")


def _load(context:RunContext,participants:list[str],duration:float,k:int)->dict[str,np.ndarray]:
    values={name:[] for name in ("positive_phase","null_phase","positive_hotelling","null_hotelling")}
    for participant in participants:
        with np.load(feature_path(context,participant,duration,k),allow_pickle=False) as archive:
            for name in values: values[name].append(archive[name])
    return {name:np.stack(items) for name,items in values.items()}


def _decision_chunks(model,values:np.ndarray,chunk_size:int=500_000)->np.ndarray:
    flat=np.asarray(values).reshape(-1,values.shape[-1]); result=np.empty(len(flat),np.float32)
    for start in range(0,len(flat),chunk_size): result[start:start+chunk_size]=model.decision_function(flat[start:start+chunk_size]).astype(np.float32)
    return result.reshape(values.shape[:-1])


def _stream_rows(participant:str,participant_index:int,algorithm:str,k:int,duration:float,mode:str,
                 positive:np.ndarray,null:np.ndarray,thresholds:float|np.ndarray)->list[dict]:
    rows=[]
    for stream in range(positive.shape[0]):
        threshold=float(thresholds[stream//3]) if np.ndim(thresholds) else float(thresholds)
        rows.append({"participant":participant,"participant_index":participant_index,"condition":stream//3,"stream":stream,
            "algorithm":algorithm,"k":k,"duration_seconds":duration,"calibration_mode":mode,
            "detection_rate":float(np.mean(positive[stream]>threshold)),"heldout_fpr":float(np.mean(null[stream]>threshold)),
            "auc":auc_fast(positive[stream],null[stream]),"mean_score":float(np.mean(positive[stream])),"threshold":threshold})
    return rows


def _adaptive_thresholds(null:np.ndarray,target_fpr:float)->np.ndarray:
    thresholds=np.empty(15)
    for condition in range(15):
        training_streams=np.r_[0:3*condition,3*condition+3:45]
        thresholds[condition]=empirical_threshold_fast(null[training_streams],target_fpr)
    return thresholds


def run(context:RunContext,force:bool=False)->tuple[list,dict]:
    del force; stage="02_repaired_benchmark"; config=context.config
    diagnosis=json.loads((context.run_dir/"diagnosis_lock.json").read_text(encoding="utf-8")); participants=list(diagnosis["participants"])
    participant_numbers=np.arange(len(participants)); total=len(config.durations_seconds)*len(config.k_values); context.start_stage(stage,total)
    rows=[]; calibration_rows=[]; model_rows=[]; selection_rows=[]; completed=0; started=time.perf_counter()
    for duration_index,duration in enumerate(config.durations_seconds):
        for k in config.k_values:
            features=_load(context,participants,duration,k); positive=2*k*features["positive_phase"]; null=2*k*features["null_phase"]
            fixed={"B1_ITPC":(positive[...,0],null[...,0]),"B2_Hotelling":(features["positive_hotelling"],features["null_hotelling"]),
                   "B3_EqualHarmonic":(positive.sum(axis=-1),null.sum(axis=-1))}
            fixed_thresholds={name:leave_one_participant_thresholds(pair[1],config.target_fpr) for name,pair in fixed.items()}
            candidate_arrays={candidate:(positive[...,list(candidate)].sum(axis=-1),null[...,list(candidate)].sum(axis=-1))
                              for candidate in config.harmonic_subsets}
            candidate_thresholds={candidate:leave_one_participant_thresholds(pair[1],config.target_fpr)
                                  for candidate,pair in candidate_arrays.items()}
            ml_calibration=[]
            for participant_index in participant_numbers:
                values=null[participant_index].reshape(-1,4); cap=min(config.calibration_null_cap_per_participant,len(values))
                rng=np.random.default_rng(np.random.SeedSequence([config.seed,duration_index,k,int(participant_index),991]))
                indices=np.sort(rng.choice(len(values),cap,replace=False)) if cap<len(values) else np.arange(len(values)); ml_calibration.append(values[indices])
            for heldout,participant in enumerate(participants):
                training=participant_numbers!=heldout
                evaluated={}
                for algorithm,(pos_values,null_values) in fixed.items():
                    threshold=float(fixed_thresholds[algorithm][heldout]); held_positive=pos_values[heldout]; held_null=null_values[heldout]
                    evaluated[algorithm]=(held_positive,held_null,threshold)
                subset,subset_scores=select_harmonic_subset(positive[training],null[training],config.harmonic_subsets,
                    participant_numbers[training],config.inner_group_folds,config.seed+duration_index*1000+k*25+heldout)
                subset_positive,subset_null=candidate_arrays[subset]; subset_threshold=float(candidate_thresholds[subset][heldout])
                evaluated["B4_CVSubsetFusion"]=(subset_positive[heldout],subset_null[heldout],subset_threshold)
                selection_rows.append({"participant":participant,"k":k,"duration_seconds":duration,
                    "selected_indices":"_".join(map(str,subset)),"selected_hz":"+".join(f"{config.harmonics_hz[i]:g}" for i in subset),
                    **{f"inner_auc_{'_'.join(map(str,candidate))}":score for candidate,score in
                       [(candidate,subset_scores['_'.join(map(str,candidate))]) for candidate in config.harmonic_subsets]}})
                train_values,train_labels,train_groups=training_observations(positive[training],null[training],participant_numbers[training])
                if heldout in set(train_groups.tolist()): raise AssertionError("held-out participant leaked into monotone training")
                seed=int(np.random.SeedSequence([config.seed,duration_index,k,heldout,151]).generate_state(1)[0])
                best_c,cv_scores=select_monotone_regularization(train_values,train_labels,train_groups,config.c_grid,
                    config.inner_group_folds,seed,config.maximum_iterations,config.tolerance)
                model=fit_monotone_logistic(train_values,train_labels,best_c,config.maximum_iterations,config.tolerance)
                training_null_features=np.concatenate([ml_calibration[index] for index in participant_numbers[training]])
                threshold=empirical_threshold_fast(_decision_chunks(model,training_null_features),config.target_fpr)
                held_positive=_decision_chunks(model,positive[heldout]); held_null=_decision_chunks(model,null[heldout])
                evaluated["B5_MonotoneFusion"]=(held_positive,held_null,threshold)
                model_rows.append({"participant":participant,"k":k,"duration_seconds":duration,"selected_c":best_c,
                    "intercept":model.intercept,"converged":model.converged,"heldout_excluded":True,
                    **{f"coefficient_{int(f)}hz":float(value) for f,value in zip(config.harmonics_hz,model.coefficients,strict=True)},
                    **{f"inner_auc_c_{c:g}":score for c,score in cv_scores.items()}})
                for algorithm,(held_positive,held_null,threshold) in evaluated.items():
                    rows.extend(_stream_rows(participant,heldout,algorithm,k,duration,"population",held_positive,held_null,threshold))
                    false_positives=int(np.count_nonzero(held_null>threshold)); low,high=binomial_interval(false_positives,held_null.size)
                    calibration_rows.append({"participant":participant,"algorithm":algorithm,"k":k,"duration_seconds":duration,
                        "calibration_mode":"population","heldout_fpr":false_positives/held_null.size,"ci_low":low,"ci_high":high,
                        "n_null":held_null.size,"mean_threshold":threshold})
                    if k==10 and np.isclose(duration,4.5):
                        adaptive=_adaptive_thresholds(held_null,config.target_fpr)
                        rows.extend(_stream_rows(participant,heldout,algorithm,k,duration,"participant_adaptive",held_positive,held_null,adaptive))
                        tested=np.concatenate([held_null[3*c:3*c+3].reshape(-1) for c in range(15)])
                        decisions=np.concatenate([(held_null[3*c:3*c+3]>adaptive[c]).reshape(-1) for c in range(15)])
                        false_adaptive=int(decisions.sum()); low,high=binomial_interval(false_adaptive,tested.size)
                        calibration_rows.append({"participant":participant,"algorithm":algorithm,"k":k,"duration_seconds":duration,
                            "calibration_mode":"participant_adaptive","heldout_fpr":false_adaptive/tested.size,"ci_low":low,"ci_high":high,
                            "n_null":tested.size,"mean_threshold":float(adaptive.mean())})
            completed+=1; context.progress(stage,completed,total,f"T={duration:g}s K={k}")
    paths=[]
    for name,data in (("scores",rows),("calibration",calibration_rows),("monotone_models",model_rows),("subset_selections",selection_rows)):
        path=context.table_dir/f"{name}.csv"; pd.DataFrame(data).to_csv(path,index=False); paths.append(path)
    summary={"methods":list(ALGORITHMS),"population_cells":total,"participant_adaptive_endpoint_only":True,
        "null":"same_frequency_phase_randomization","monotone_coefficients":True,"runtime_seconds":time.perf_counter()-started}
    report=context.run_dir/"repaired_benchmark.json"; atomic_write_json(report,summary); paths.append(report)
    context.finish_stage(stage,paths,summary); return paths,summary
