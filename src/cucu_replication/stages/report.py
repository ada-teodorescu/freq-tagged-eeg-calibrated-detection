from __future__ import annotations

import json

import pandas as pd

from ..runtime import RunContext, atomic_write_json, atomic_write_text


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run(context: RunContext, force: bool = False) -> tuple[list, dict]:
    stage = "07_report"
    context.start_stage(stage, 6)
    stored = _read_json(context.run_dir / "stored_fft_gate.json")
    context.progress(stage, 1, 6, "stored FFT gate")
    source = _read_json(context.run_dir / "source_provenance_gate.json")
    context.progress(stage, 2, 6, "source provenance")
    published = _read_json(context.run_dir / "published_results.json")
    context.progress(stage, 3, 6, "published outcomes")
    hilbert = _read_json(context.run_dir / "hilbert_robustness.json")
    context.progress(stage, 4, 6, "Hilbert robustness")
    filter_null = _read_json(context.run_dir / "filter_null.json")
    context.progress(stage, 5, 6, "filter null")
    frequency = _read_json(context.run_dir / "frequency_correction.json")
    context.progress(stage, 6, 6, "frequency correction")

    reconstructed = pd.read_csv(context.table_dir / "reconstructed_acoustic_correlations.csv")
    reconstructed_headline = reconstructed[
        (reconstructed["feature"] == "maximum_derivative_latency")
        & (reconstructed["outcome"] == "itpc_4hz")
    ]
    reconstructed_record = (
        None if reconstructed_headline.empty else reconstructed_headline.iloc[0].to_dict()
    )
    original_acoustic = published["original_acoustic"]
    exact_derivative = original_acoustic["maximum_derivative_latency_vs_itpc4"]
    paper_pca = published["paper_reported_pca"]
    repository_pca = published["repository_score_pca"]
    hilbert_table = pd.read_csv(context.table_dir / "hilbert_fft_comparison.csv")
    claims = {
        "stored_fft_replication": {
            "status": "supported" if stored["passed"] else "not_supported",
            "source": "stored_matlab",
            "evidence": stored,
        },
        "public_set_matches_stored_fft": {
            "status": "supported" if source["passed_exact_equivalence"] else "not_supported",
            "source": "public_set_fdt_vs_stored_matlab",
            "evidence": source,
        },
        "paper_reported_pca": {
            "status": "supported" if published["pca_gate_passed"] else "not_supported",
            "source": "stored_matlab_25_participant_means",
            "evidence": paper_pca,
        },
        "repository_score_pca": {
            "status": "supported"
            if max(repository_pca["archived_score_maximum_errors"]) < context.config.exact_tolerance
            else "not_supported",
            "source": "original_repository_1125_stream_participant_rows",
            "evidence": repository_pca,
        },
        "pca_provenance_discrepancy": {
            "status": "confirmed",
            "evidence": {
                "paper_reported_observations": paper_pca["observations"],
                "repository_score_observations": repository_pca["observations"],
                "paper_variance": paper_pca["explained_variance_ratio"][:2],
                "repository_variance": repository_pca["explained_variance_ratio"][:2],
            },
        },
        "principal_three_group_effect": {
            "status": "supported" if published["three_group_anova"]["p_gg"] < 0.05 else "not_supported",
            "source": "archived_r_input",
            "evidence": published["three_group_anova"],
        },
        "original_maximum_derivative_latency_relationship": {
            "status": "supported"
            if abs(exact_derivative["pearson_r"] - (-0.9129993)) < 1e-6
            else "not_supported",
            "source": "original_repository_stored_acoustic_matrix",
            "evidence": exact_derivative,
            "scope": "stored-product replication; raw 75-syllable marker extraction unavailable",
        },
        "continuous_stream_acoustic_reconstruction": {
            "status": "robustness_extension",
            "source": "independently_reconstructed_from_45_stream_wavs",
            "evidence": reconstructed_record,
        },
        "hilbert_group_level_robustness": {
            "status": "reported",
            "source": "public_set_fdt_vs_stored_matlab",
            "evidence": hilbert_table.to_dict(orient="records"),
            "interpretation_level": hilbert["interpretation_level"],
        },
        "filter_induced_false_positive_rate": {
            "status": "flagged" if filter_null["flagged_configurations"] else "no_inflation_detected",
            "source": "public_set_fdt_phase_randomized",
            "evidence": filter_null,
        },
        "frequency_mismatch": {
            "status": frequency["interpretation"],
            "source": "target_stream_wavs_and_public_set_fdt",
            "evidence": frequency,
        },
    }
    state = context.read_state()
    manifest = pd.read_csv(context.table_dir / "input_manifest.csv")
    commands = [
        "uv run cucu-replication list",
        "uv run cucu-replication run-all --workers 1",
        "uv run cucu-replication status",
        "uv run cucu-replication status --watch",
    ]
    lines = [
        "# Exact Cucu replication and robustness summary",
        "",
        f"Analysis ID: `{context.analysis_id}`",
        "",
        "## Provenance",
        "",
        f"- Stored FFT -> ITPC exact gate: **{stored['passed']}**; maximum error `{stored['maximum_absolute_error']:.3g}`.",
        f"- Public SET/FDT -> stored MATLAB equivalence: **{source['passed_exact_equivalence']}**.",
        f"- Optimal participant-assignment mean correlation: `{source['optimal_mean_correlation']:.4f}`.",
        "- Stored MATLAB and public time-domain results remain separately labelled whenever the equivalence gate fails.",
        "",
        "## Exact archived results",
        "",
        f"- Paper-reported 25 x 4 PCA: PC1 `{paper_pca['explained_variance_ratio'][0] * 100:.4f}%`, PC2 `{paper_pca['explained_variance_ratio'][1] * 100:.4f}%`.",
        f"- Repository-score 1,125 x 4 PCA: PC1 `{repository_pca['explained_variance_ratio'][0] * 100:.4f}%`, PC2 `{repository_pca['explained_variance_ratio'][1] * 100:.4f}%`.",
        "- These PCA definitions are reported separately because they answer different provenance questions.",
        f"- Three-group Greenhouse-Geisser p-value from archived R input: `{published['three_group_anova']['p_gg']:.6g}`.",
        f"- Stored acoustic maximum-derivative latency versus 4-Hz ITPC: `r = {exact_derivative['pearson_r']:.7f}`.",
        f"- Stored acoustic PCA: PC1 `{original_acoustic['pca_explained_variance_ratio'][0] * 100:.4f}%`, PC2 `{original_acoustic['pca_explained_variance_ratio'][1] * 100:.4f}%`; `|r| = {original_acoustic['acoustic_pc1_vs_archived_itpc1_abs_r']:.6f}` versus archived ITPC1.",
        "- Exact acoustic claims stop at the archived marker matrix because the original 75 individual syllable WAV files are not local.",
        "- Landmark extraction from the 45 continuous streams is reported separately as reconstruction robustness.",
        "",
        "## Robustness analyses",
        "",
        f"- Hilbert comparison interpretation: **{hilbert['interpretation_level']}**.",
        f"- Filter-null configurations flagged: **{filter_null['flagged_configurations']}**.",
        f"- Mean fitted stream frequency: `{frequency['mean_fitted_frequency_hz']:.6f} Hz`.",
        f"- Frequency result: **{frequency['interpretation']}**.",
        "",
        "## Reproduction commands",
        "",
        "```powershell",
        *commands,
        "```",
        "",
        "## Run record",
        "",
        f"- Configuration: `{context.config.config_path}`",
        f"- Run directory: `{context.run_dir}`",
        f"- Input files fingerprinted: `{len(manifest)}`",
        f"- Package versions: `{json.dumps(state['versions'], sort_keys=True)}`",
        "",
        "Detailed CSV tables and figures are stored beside this report.",
    ]
    summary_path = context.run_dir / "replication_summary.md"
    claims_path = context.run_dir / "claim_status.json"
    report_path = context.run_dir / "final_report.json"
    atomic_write_text(summary_path, "\n".join(lines) + "\n")
    atomic_write_json(claims_path, claims)
    report = {
        "analysis_id": context.analysis_id,
        "commands": commands,
        "claims_path": str(claims_path),
        "summary_path": str(summary_path),
        "input_files": len(manifest),
        "versions": state["versions"],
    }
    atomic_write_json(report_path, report)
    outputs = [summary_path, claims_path, report_path]
    context.finish_stage(stage, outputs, report)
    return outputs, report
