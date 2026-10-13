# Curated results

These snapshots contain compact result records, tables, figures, and provenance for the analysis. Run identifiers refer to the original analysis environment; rerunning from this source copy may produce new content hashes. Raw EEG, intermediate coefficients, checkpoints, and full output runs are not included.

| Analysis | Result directory | Role |
| --- | --- | --- |
| Source replication | `replication/` | Archived product and estimator audit |
| Raw reconstruction | `raw_replication/42ae2efcc15c/` | Public epoch and group-effect provenance audit |
| Exploratory development | `detection/`, `improvement/`, `algorithm_study/`, `accuracy_study/`, `advanced_study/` | Detector selection and robustness context |
| Frozen syllable comparison | `confirmatory_study/4d6a54f6f902/` | Hann–Hilbert and TRCA comparisons |
| External negative controls | `external_validation/2341d3cd4a7a/` | Frozen-detector false-positive evaluation |
| Independent ASSR comparison | `assr_validation/a94671b9a071/` | 40-Hz primary and 27-Hz secondary analyses |
| Exploratory estimator controls | `validation_extension/e884f9e7398e/` | Rectangular and filter-matched comparisons |
| Restricted-input control | `support_study/37ecb27f6687/` | Temporal-context sensitivity |

Local file paths in text result records were converted to paths relative to the repository root. Numerical results and recorded input hashes were preserved. These are analysis reports, not manuscript drafts.

The frozen records retain their original `cucu_*` module-path references. Current code and tests use `eeg_*`; the former `cucu_article` helper package is now `eeg_core`. The recorded run identifiers and numerical results were not changed by this source-tree rename.
