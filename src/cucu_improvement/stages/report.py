from __future__ import annotations

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from cucu_detection.evaluation import crossed_bootstrap_gain, sign_flip_test, trials_to_target
from ..runtime import RunContext, atomic_write_json, atomic_write_text


def _summary(table: pd.DataFrame, group: list[str], target: float) -> pd.DataFrame:
    aggregate = table.groupby(group + ["k"], as_index=False).agg(
        detection_rate=("detection_rate", "mean"), auc=("auc", "mean"),
        heldout_fpr=("heldout_fpr", "mean") if "heldout_fpr" in table else ("detection_rate", lambda _: np.nan),
    )
    rows = []
    for _, frame in aggregate.groupby(group):
        frame = frame.sort_values("k"); k80 = trials_to_target(frame.k.to_numpy(), frame.detection_rate.to_numpy(), target)
        for row in frame.to_dict(orient="records"): row["trials_to_80_percent"] = k80; rows.append(row)
    return pd.DataFrame(rows)


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "04_report"; context.start_stage(stage, 5)
    stored = pd.read_csv(context.table_dir / "stored_d8_scores.csv")
    calibration = pd.read_csv(context.table_dir / "stored_d8_calibration.csv")
    public = pd.read_csv(context.table_dir / "public_focused_scores.csv")
    stored = stored.merge(calibration[["heldout_participant", "method", "k", "heldout_fpr"]],
                          left_on=["participant", "method", "k"],
                          right_on=["heldout_participant", "method", "k"], how="left")
    stored_summary = _summary(stored, ["branch", "method"], context.config.target_detection)
    public_summary = _summary(public, ["branch", "method", "spatial"], context.config.target_detection)
    summary_path = context.table_dir / "focused_trials_auc_fpr.csv"
    pd.concat((stored_summary, public_summary), ignore_index=True).to_csv(summary_path, index=False)
    context.progress(stage, 1, 5, "method summaries")

    curves = stored.groupby(["participant", "condition", "method", "k"], as_index=False).detection_rate.mean()
    baseline = curves[(curves.method == "D1_4Hz") & (curves.k == 10)].pivot(
        index="participant", columns="condition", values="detection_rate"
    )
    comparisons = []
    for method in ("D8_all_equal", "D8_4_8_equal", "D8_train_weighted"):
        candidate = curves[(curves.method == method) & (curves.k == 10)].pivot(
            index="participant", columns="condition", values="detection_rate"
        )
        gain = candidate.to_numpy() - baseline.to_numpy()
        ci = crossed_bootstrap_gain(gain, 5000, context.config.seed)
        comparisons.append({"candidate": method, "comparison_k": 10, "mean_detection_gain": float(gain.mean()),
            "crossed_bootstrap_ci_low": ci[0], "crossed_bootstrap_ci_high": ci[1],
            "condition_sign_flip_p": sign_flip_test(gain.mean(axis=0), 10000, context.config.seed),
            "heldout_fpr": float(calibration[(calibration.method == method) & (calibration.k == 10)].heldout_fpr.mean())})
    comparison_path = context.table_dir / "stored_d8_followup_comparisons.csv"
    pd.DataFrame(comparisons).to_csv(comparison_path, index=False)
    context.progress(stage, 2, 5, "D8 comparisons")

    ranking_rows = []
    public_k10 = public[public.k == 10]
    reference = public_k10[(public_k10.method == "F1_D1_4Hz") & (public_k10.spatial == "S0")].groupby(
        "outer_condition").detection_rate.mean()
    for (method, spatial), frame in public_k10.groupby(["method", "spatial"]):
        condition = frame.groupby("outer_condition").detection_rate.mean().reindex(reference.index)
        ranking_rows.append({"method": method, "spatial": spatial,
            "condition_rank_spearman_vs_F1_S0": float(stats.spearmanr(reference, condition).statistic)})
    ranking_path = context.table_dir / "public_condition_ranking.csv"
    pd.DataFrame(ranking_rows).to_csv(ranking_path, index=False)
    context.progress(stage, 3, 5, "condition ranking")

    figure_path = context.figure_dir / "focused_detection_vs_trials.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for method, frame in stored_summary.groupby("method"):
        axes[0].plot(frame.k, frame.detection_rate, marker="o", label=method)
    axes[0].axhline(context.config.target_detection, color="black", ls="--", lw=1)
    axes[0].set(title="Stored MATLAB branch", xlabel="Trials (K)", ylabel="Held-out detection rate", ylim=(0, 1))
    axes[0].legend(fontsize=7)
    for (method, spatial), frame in public_summary.groupby(["method", "spatial"]):
        axes[1].plot(frame.k, frame.detection_rate, marker=".", label=f"{method}+{spatial}")
    axes[1].axhline(context.config.target_detection, color="black", ls="--", lw=1)
    axes[1].set(title="Public SET/FDT branch", xlabel="Trials (K)", ylabel="Held-out detection rate", ylim=(0, 1))
    axes[1].legend(fontsize=6, ncol=2)
    figure.savefig(figure_path, dpi=180); plt.close(figure)
    context.progress(stage, 4, 5, "figure")

    best_stored = json.loads(stored_summary.sort_values("detection_rate", ascending=False).iloc[0].to_json())
    best_public = json.loads(public_summary.sort_values("detection_rate", ascending=False).iloc[0].to_json())
    claims = {
        "baseline_run_immutable": True, "branches_numerically_merged": False,
        "hilbert_coefficient_corrected_by_demodulation": True,
        "thomson_line_and_f_implemented": True,
        "multitaper_calibrated_with_neighboring_frequencies": True,
        "s1_is_cross_fitted": True, "ress_dss_cca_retested": False,
        "stored_comparisons": comparisons, "best_stored_row": best_stored, "best_public_row": best_public,
    }
    claim_path = context.run_dir / "claim_status.json"; atomic_write_json(claim_path, claims)
    lines = ["# Focused Cucu Algorithm Follow-up", "",
        "This analysis is separate from the completed preregistered detector benchmark.", "",
        "## Changes tested", "",
        "- D8 with all harmonics, 4+8 Hz only, and weights fitted on training participants only.",
        "- Corrected Hilbert coefficients using complex demodulation before time averaging.",
        "- DPSS line estimates and Thomson F statistics calibrated against neighboring frequencies.",
        "- Only S0 and cross-fitted S1; RESS, DSS, and CCA were not rerun.", "",
        "## Interpretation rule", "",
        "The stored MATLAB and public SET/FDT branches remain separate because source equivalence failed.", "",
        f"Machine-readable results: `{claim_path.name}` and `tables/{summary_path.name}`.", ""]
    report_path = context.run_dir / "improvement_summary.md"; atomic_write_text(report_path, "\n".join(lines))
    context.progress(stage, 5, 5, "report")
    outputs = [summary_path, comparison_path, ranking_path, figure_path, claim_path, report_path]
    context.finish_stage(stage, outputs, claims); return outputs, claims
