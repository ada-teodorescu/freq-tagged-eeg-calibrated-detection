# Extension export: rectangular and filter-matched comparators

Full export of `outputs/validation_extension/e884f9e7398e/`, superseding
`results/validation_extension/b6e133add4eb/`. This run adds one exploratory
contrast (`mechanism_exploratory`, `H2_Hilbert4s - M2_FilteredLS4s`) to
`tables/inferential_comparisons.csv` and `validation_extension_conclusions.json`
relative to `b6e133add4eb`; the analysis identifier changed because it is a
content hash over `src/cucu_validation_extension/**/*.py` plus configuration
and inputs, and the new contrast required a code change in `stages/statistics.py`.
`static_summary.csv` and every other pre-existing number are byte-identical to
`b6e133add4eb` (stages 00-05 read from unchanged checkpoints; only the
statistics stage changed).

- `tables/static_summary.csv`: estimator detection and held-out surrogate FPR,
  including `M2_FilteredLS4s` (the filter-matched exploratory control: the same
  Hilbert Butterworth bandpass, then sinusoidal regression without the Hilbert
  transform).
- `tables/inferential_comparisons.csv`: mechanism and sequential contrasts.
  The `mechanism` family (`H2_Hilbert4s - H1_Hann4s`, `H2_Hilbert4s - M1_RectLS4s`)
  is the original, Holm-corrected, locked two-contrast family
  (`design_lock.json`'s `primary_inference.mechanism_contrasts`). The single
  `mechanism_exploratory` row (`H2_Hilbert4s - M2_FilteredLS4s`) is reported
  pointwise and was never pooled into that Holm correction.
- `validation_extension_conclusions.json`: adds `mechanism_exploratory_filter_matched`,
  including the participant-condition-level McNemar exact test (0 of 375
  discordant decisions between Hilbert and the filter-matched control).
- `design_lock.json`, `config_snapshot.toml`, `input_fingerprints.json`: source
  provenance, unchanged in substance from `b6e133add4eb`.
- `methods.md`, `results.md`, `limitations.md`, `report.md`: full manuscript-draft
  text and figures from the `export` CLI command (a fuller export than the
  minimal hand-picked copy used for `b6e133add4eb`).

The source configuration identifies confirmatory baseline `4d6a54f6f902`. The
Cucu rectangular and filter-matched comparisons are exploratory and remain
outside the original confirmatory three-hypothesis family. The manuscript's
original Hann--Hilbert interval and adjusted p-value continue to come from the
confirmatory export. The independent ASSR secondary comparison comes from ASSR
export `a94671b9a071`.
