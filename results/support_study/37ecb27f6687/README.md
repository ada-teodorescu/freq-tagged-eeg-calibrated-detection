# Restricted-input Hilbert reporting inputs

Selected files copied byte-for-byte from `outputs/support_study/37ecb27f6687/`.
This is a reporting subset, not a full export or a new analysis.

- `tables/summary.csv`: static detection and held-out surrogate FPR for all variants.
- `tables/comparisons.csv`: primary exploratory and padding-sensitivity contrasts.
- `tables/source_coefficient_audit.csv`: agreement with recomputed source coefficients.
- `design.json`, `config_snapshot.toml`, `input_fingerprints.json`: provenance and design.
- `report.md`: source analysis summary and interpretation limits.

The source confirmatory baseline is `4d6a54f6f902`. The new comparisons are exploratory,
with pointwise crossed-bootstrap intervals conditional on fitted models. They do not
replace the original locked Hann--Hilbert comparison or its multiplicity correction.
Participant/fold checkpoints and detailed score/membership tables remain in the source run.
