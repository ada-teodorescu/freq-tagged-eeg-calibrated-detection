from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from ..runtime import RunContext, atomic_write_json, atomic_write_text
from .common import preprocess_paths


UNRESOLVED = (
    "The authors' exact EEGLAB filter calls and transition-band choices are unavailable.",
    "The authors' ICA command, seed, component IDs, and exact 12% formula are unavailable.",
    "The behavioral inclusion/exclusion decision log is unavailable.",
    "The released MATLAB participant axis has no explicit participant-ID table.",
    "The publication and repository disagree about the PCA observation matrix.",
)


def _participant_qc(context: RunContext, participant: str, epoch_rows: pd.DataFrame) -> Path:
    summary = json.loads(preprocess_paths(context, participant)["summary"].read_text(encoding="utf-8"))
    rows = epoch_rows[epoch_rows["participant"] == participant]
    lines = [
        f"# {participant} quality-control record", "",
        f"- Epochs: {summary['epochs']}",
        f"- Channels: {summary['channels']}",
        f"- Samples per epoch: {summary['samples_per_epoch']}",
        f"- Paper-rule rejected components: {summary['paper12_excluded']}",
        f"- ICLabel rejected components: {summary['iclabel_excluded']}", "",
        "## Same-ID SET/FDT agreement", "",
        "| Branch | Pearson r | normalized RMSE | amplitude ratio |",
        "|---|---:|---:|---:|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(f"| {row.branch} | {row.pearson_r:.6f} | {row.normalized_rmse:.6f} | {row.amplitude_ratio:.6f} |")
    figure = context.figure_dir / "participants" / f"{participant}_ica_scores.png"
    if figure.exists():
        lines.extend(("", f"![ICA frontal-power scores](../../figures/participants/{figure.name})"))
    destination = context.run_dir / "qc" / f"{participant}.md"
    atomic_write_text(destination, "\n".join(lines) + "\n")
    return destination


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "07_report"
    context.start_stage(stage, len(context.participants) + 4)
    inventory = json.loads((context.run_dir / "inventory.json").read_text(encoding="utf-8"))
    preprocessing = json.loads((context.run_dir / "preprocessing.json").read_text(encoding="utf-8"))
    products = json.loads((context.run_dir / "product_comparison.json").read_text(encoding="utf-8"))
    paper = json.loads((context.run_dir / "paper_results.json").read_text(encoding="utf-8"))
    epoch_rows = pd.read_csv(context.table_dir / "epoch_agreement.csv")
    context.progress(stage, 1, len(context.participants) + 4, "evidence inputs")
    outputs: list[Path] = []
    for index, participant in enumerate(context.participants, start=2):
        outputs.append(_participant_qc(context, participant, epoch_rows))
        context.progress(stage, index, len(context.participants) + 4, participant)

    primary = products["branches"].get("paper12")
    primary_paper = paper["branches"].get("paper12")
    claims = {
        "documented_method_adherence": {
            "status": "operationalized_with_disclosed_assumptions",
            "evidence": {
                "filter": preprocessing["participants"][0]["documented_filter"],
                "ica_rule": "Fp1+Fp2 squared scalp-map power > 12%",
                "verified_author_code": False,
            },
        },
        "raw_to_set_preprocessing_reproducibility": {
            "status": "reported_per_participant_and_branch",
            "source": "tables/epoch_agreement.csv",
        },
        "fft_itpc_product_reproducibility": {
            "status": "reported_without_post_hoc_tuning",
            "primary": primary,
        },
        "paper_result_reproducibility": {
            "status": "reported_by_effect_and_inferential_conclusion",
            "primary": primary_paper,
        },
        "unresolved_author_decisions": {"status": "cannot_be_determined", "items": list(UNRESOLVED)},
    }
    claims_path = context.run_dir / "claims.json"
    atomic_write_json(claims_path, claims)
    outputs.append(claims_path)
    context.progress(stage, len(context.participants) + 2, len(context.participants) + 4, "claim ledger")

    branch_lines = []
    for branch, summary in products["branches"].items():
        counts = summary["classification_counts"]
        branch_lines.append(
            f"| {branch} | {summary['same_id_mean_r']:.4f} | {summary['same_id_mean_nrmse']:.4f} | "
            f"{counts['matches']} | {counts['approximately_agrees']} | {counts['does_not_agree']} |"
        )
    unresolved_lines = "\n".join(f"- {item}" for item in UNRESOLVED)
    report = f"""# Modern Python/MNE raw replication

Analysis ID: `{context.analysis_id}`

This is an isolated Python/MNE operationalization of the published method. MATLAB R2018b and EEGLAB 13.6.5b are historical references only. Released products were fixed validation targets and were not used to tune preprocessing or ICA decisions.

## Cohort and structural checks

- Archive-defined comparison cohort: {len(context.participants)} participants.
- Every selected recording passed the configured 32-channel, 1,000-Hz, 450-event, 10-trials-per-stream gates.
- Behavioral exclusion decisions cannot be independently recreated from the released records.
- Checkpoint footprint: {preprocessing['checkpoint_gb']:.3f} GB (limit {context.config.maximum_checkpoint_gb:.1f} GB).

## Released-product agreement

The lexicographic same-ID mapping is the scientific comparison. The optimal assignment table is diagnostic only.

| Branch | mean r | mean nRMSE | matches | approximately agrees | does not agree |
|---|---:|---:|---:|---:|---:|
{chr(10).join(branch_lines)}

Thresholds were declared in `raw_replication_config.toml`. Disagreement is retained as a result rather than converted into a runtime failure.

## Primary paper-rule analysis

- PCA PC1 variance: {primary_paper['paper_pca_25x4']['explained_variance_ratio'][0]:.4%}
- PCA PC2 variance: {primary_paper['paper_pca_25x4']['explained_variance_ratio'][1]:.4%}
- Three-group Greenhouse-Geisser p: {primary_paper['three_group_pc1_anova']['p_gg']:.6g}
- Voiced versus unvoiced paired p: {primary_paper['voiced_vs_unvoiced_stops']['p']:.6g}

Full branch-specific estimates are in `paper_results.json` and the CSV tables. Participant QC records are in `qc/`.

## Interpretation boundaries

{unresolved_lines}

The 12% rule is an explicit reconstruction assumption, not a verified copy of the authors' unpublished preprocessing code.
"""
    report_path = context.run_dir / "report.md"
    atomic_write_text(report_path, report)
    outputs.append(report_path)
    context.progress(stage, len(context.participants) + 3, len(context.participants) + 4, "report")
    evidence_path = context.run_dir / "evidence.json"
    atomic_write_json(evidence_path, {
        "analysis_id": context.analysis_id, "inventory": inventory,
        "claims": claims, "product_comparison": products, "paper_results": paper,
    })
    outputs.append(evidence_path)
    context.progress(stage, len(context.participants) + 4, len(context.participants) + 4, "machine-readable evidence")
    context.finish_stage(stage, outputs, {"passed": True, "claims": list(claims)})
    return outputs, claims
