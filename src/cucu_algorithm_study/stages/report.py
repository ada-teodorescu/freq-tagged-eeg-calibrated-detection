from __future__ import annotations

import json
import numpy as np
import pandas as pd

from cucu_detection.evaluation import crossed_bootstrap_gain,sign_flip_test
from ..runtime import RunContext,atomic_write_json,atomic_write_text


def _comparison(frame:pd.DataFrame,candidate:str,baseline:str,config,seed_offset:int)->dict:
    grouped=frame.groupby(["participant_index","condition","algorithm"],as_index=False).detection_rate.mean()
    pivot=grouped.pivot(index=["participant_index","condition"],columns="algorithm",values="detection_rate")
    gains=(pivot[candidate]-pivot[baseline]).to_numpy().reshape(25,15)
    low,high=crossed_bootstrap_gain(gains,config.bootstrap_iterations,config.seed+seed_offset)
    return {"candidate":candidate,"baseline":baseline,"mean_detection_gain":float(gains.mean()),
            "crossed_bootstrap_ci95":[low,high],"condition_sign_flip_p":sign_flip_test(gains.mean(axis=0),config.sign_flip_iterations,config.seed+100+seed_offset),
            "participants_positive":int((gains.mean(axis=1)>0).sum()),"conditions_positive":int((gains.mean(axis=0)>0).sum())}


def run(context:RunContext,force:bool=False)->tuple[list,dict]:
    del force; stage="03_report"; config=context.config; context.start_stage(stage,5)
    scores=pd.read_csv(context.table_dir/"scores.csv"); calibration=pd.read_csv(context.table_dir/"calibration.csv")
    aggregate=scores.groupby(["algorithm","k","duration_seconds","calibration_mode"],as_index=False).agg(
        detection_rate=("detection_rate","mean"),auc=("auc","mean"),heldout_fpr=("heldout_fpr","mean"))
    aggregate["fpr_accepted"]=aggregate.heldout_fpr.between(config.fpr_acceptance_low,config.fpr_acceptance_high)
    aggregate_path=context.table_dir/"algorithm_summary.csv"; aggregate.to_csv(aggregate_path,index=False)
    context.progress(stage,1,5,"endpoints aggregated")
    primary=scores[(scores.k==10)&np.isclose(scores.duration_seconds,4.5)]
    comparisons=[]
    for mode in ("population","participant_adaptive"):
        frame=primary[primary.calibration_mode==mode]
        for offset,candidate in enumerate(("B2_Hotelling","B3_EqualHarmonic","B4_CVSubsetFusion","B5_MonotoneFusion")):
            comparisons.append({"calibration_mode":mode,**_comparison(frame,candidate,"B1_ITPC",config,offset+(0 if mode=="population" else 20))})
        comparisons.append({"calibration_mode":mode,**_comparison(frame,"B4_CVSubsetFusion","B3_EqualHarmonic",config,40+(0 if mode=="population" else 20))})
        comparisons.append({"calibration_mode":mode,**_comparison(frame,"B5_MonotoneFusion","B3_EqualHarmonic",config,50+(0 if mode=="population" else 20))})
    comparison_table=pd.DataFrame(comparisons); comparison_path=context.table_dir/"primary_comparisons.csv"; comparison_table.to_csv(comparison_path,index=False)
    context.progress(stage,2,5,"paired comparisons")
    models=pd.read_csv(context.table_dir/"monotone_models.csv"); coefficients=[f"coefficient_{int(f)}hz" for f in config.harmonics_hz]
    if (models[coefficients]<-1e-12).any().any(): raise AssertionError("monotone model produced a negative coefficient")
    selections=pd.read_csv(context.table_dir/"subset_selections.csv")
    old_summary=pd.read_csv(config.baseline_article_run/"tables"/"algorithm_summary.csv")
    old_models=pd.read_csv(config.baseline_article_run/"tables"/"learned_models.csv")
    short_old=old_summary[(old_summary.algorithm=="A4_LearnedFusion")&(old_summary.k==10)&(old_summary.duration_seconds<=2)]
    short_new=aggregate[(aggregate.algorithm=="B5_MonotoneFusion")&(aggregate.k==10)&(aggregate.duration_seconds<=2)&(aggregate.calibration_mode=="population")]
    primary_cal=calibration[(calibration.k==10)&np.isclose(calibration.duration_seconds,4.5)]
    fpr_spread=primary_cal.groupby(["algorithm","calibration_mode"]).heldout_fpr.agg(["mean","std","min","max"]).reset_index()
    fpr_path=context.table_dir/"participant_fpr_spread.csv"; fpr_spread.to_csv(fpr_path,index=False)
    diagnostics={
        "negative_monotone_coefficients":0,"all_monotone_models_converged":bool(models.converged.all()),
        "old_short_window_detection":short_old[["duration_seconds","detection_rate","auc"]].to_dict(orient="records"),
        "repaired_short_window_detection":short_new[["duration_seconds","detection_rate","auc","heldout_fpr"]].to_dict(orient="records"),
        "primary_subset_selection_counts":selections[(selections.k==10)&np.isclose(selections.duration_seconds,4.5)].selected_hz.value_counts().to_dict(),
        "primary_fpr_spread":fpr_spread.to_dict(orient="records"),
    }
    diagnostics_path=context.run_dir/"repair_diagnostics.json"; atomic_write_json(diagnostics_path,diagnostics)
    context.progress(stage,3,5,"failure repairs checked")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":8,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":160,"savefig.bbox":"tight"})
    population=aggregate[(aggregate.calibration_mode=="population")&np.isclose(aggregate.duration_seconds,4.5)]
    fig,ax=plt.subplots(figsize=(5,3))
    for algorithm,group in population.groupby("algorithm"):
        group=group.sort_values("k"); ax.plot(group.k,group.detection_rate,marker="o",ms=3,label=algorithm.replace("_"," "))
    ax.axhline(.8,color=".5",ls="--"); ax.set(xlabel="Trials K",ylabel="Detection probability",ylim=(0,1)); ax.legend(fontsize=6,frameon=False,ncol=2)
    curve_path=context.figure_dir/"repaired_detection_by_trials.png"; fig.savefig(curve_path); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(6.5,2.6))
    for ax,algorithm in zip(axes,("B3_EqualHarmonic","B5_MonotoneFusion"),strict=True):
        data=aggregate[(aggregate.algorithm==algorithm)&(aggregate.calibration_mode=="population")].pivot(index="duration_seconds",columns="k",values="detection_rate")
        image=ax.imshow(data.to_numpy(),origin="lower",aspect="auto",vmin=0,vmax=1,cmap="viridis"); ax.set_title(algorithm.replace("_"," "))
        ax.set_xticks(range(len(data.columns)),data.columns); ax.set_yticks(range(len(data.index)),[f"{v:g}" for v in data.index]); ax.set(xlabel="Trials K",ylabel="Duration (s)")
    fig.colorbar(image,ax=axes,label="Detection probability"); duration_path=context.figure_dir/"repaired_duration_heatmaps.png"; fig.savefig(duration_path); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,2.8)); labels=[]; data=[]
    for algorithm in ("B1_ITPC","B3_EqualHarmonic","B5_MonotoneFusion"):
        for mode in ("population","participant_adaptive"):
            data.append(primary_cal[(primary_cal.algorithm==algorithm)&(primary_cal.calibration_mode==mode)].heldout_fpr.to_numpy()); labels.append(f"{algorithm.split('_')[0]}\n{mode.replace('_',' ')}")
    ax.boxplot(data,tick_labels=labels,showfliers=False); ax.axhline(.05,color=".5",ls="--"); ax.set_ylabel("Participant FPR")
    fpr_figure=context.figure_dir/"participant_calibration.png"; fig.savefig(fpr_figure); plt.close(fig)
    context.progress(stage,4,5,"diagnostic figures")
    pop=aggregate[(aggregate.k==10)&np.isclose(aggregate.duration_seconds,4.5)&(aggregate.calibration_mode=="population")].set_index("algorithm")
    b3_vs_b1=next(row for row in comparisons if row["calibration_mode"]=="population" and row["candidate"]=="B3_EqualHarmonic")
    b5_vs_b3=next(row for row in comparisons if row["calibration_mode"]=="population" and row["candidate"]=="B5_MonotoneFusion" and row["baseline"]=="B3_EqualHarmonic")
    conclusions={"primary":pop.reset_index().to_dict(orient="records"),"B3_vs_B1":b3_vs_b1,"B5_vs_B3":b5_vs_b3,
        "short_window_artifact_removed":bool(short_new.detection_rate.max()<short_old.detection_rate.max()),
        "participant_adaptive_calibration_is_sensitivity":True,"manuscript_claims_updated":False}
    conclusion_path=context.run_dir/"algorithm_conclusions.json"; atomic_write_json(conclusion_path,conclusions)
    report=f"""# Repaired algorithm study

This follow-up does not update the manuscript. It replaces neighboring-frequency nulls with same-frequency phase randomization, constrains learned harmonic coefficients to be non-negative, evaluates training-only harmonic subset selection, and tests participant-adaptive null calibration.

Primary population-calibrated B3 versus B1 gain: {100*b3_vs_b1['mean_detection_gain']:.1f} percentage points, 95% CI [{100*b3_vs_b1['crossed_bootstrap_ci95'][0]:.1f}, {100*b3_vs_b1['crossed_bootstrap_ci95'][1]:.1f}].

Primary monotone fusion versus equal fusion gain: {100*b5_vs_b3['mean_detection_gain']:.1f} percentage points, 95% CI [{100*b5_vs_b3['crossed_bootstrap_ci95'][0]:.1f}, {100*b5_vs_b3['crossed_bootstrap_ci95'][1]:.1f}].

See `repair_diagnostics.json` for short-window and participant-calibration repairs.
"""
    report_path=context.run_dir/"report.md"; atomic_write_text(report_path,report); context.progress(stage,5,5,"conclusions locked")
    outputs=[aggregate_path,comparison_path,fpr_path,diagnostics_path,curve_path,duration_path,fpr_figure,conclusion_path,report_path]
    context.finish_stage(stage,outputs,conclusions); return outputs,conclusions
