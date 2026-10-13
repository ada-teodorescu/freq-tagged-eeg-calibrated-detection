# Advanced Detector Study

Analysis ID: `8bf2fe56f4b9`

## Locked primary verdict

The S2 sequential detector **PASSED** the locked success rule. Its detection gain over S1 was
11.1 percentage points (crossed-bootstrap 95% CI
[5.2, 17.5]),
and its held-out surrogate trajectory-wise FWER was 4.82%.
The gain-CI criterion was `True` and the locked 4.5–5.5% calibration criterion
was `True`.

## Stream-level results

- S1_ITPC4: sensitivity 24.4%; held-out surrogate FWER 5.21% (participant-cluster 95% CI 4.16–6.34%).
- S2_Equal4plus8: sensitivity 35.5%; held-out surrogate FWER 4.82% (participant-cluster 95% CI 4.21–5.50%).
- S3_RidgeComplex4plus8: sensitivity 31.9%; held-out surrogate FWER 5.03% (participant-cluster 95% CI 4.69–5.39%).
- S4_RegionalMonotone4plus8: sensitivity 44.4%; held-out surrogate FWER 5.08% (participant-cluster 95% CI 4.61–5.59%).

Sequential detection used the natural recorded repetition order. Undetected streams were censored at 11
for restricted mean stopping-trial summaries. The 100 deterministic order permutations are exported only
as robustness evidence.

## Secondary stream comparisons

- S3_RidgeComplex4plus8 vs S2_Equal4plus8: gain -3.6 pp, crossed-bootstrap 95% CI [-7.0, 0.1], Holm p=0.0119.
- S4_RegionalMonotone4plus8 vs S2_Equal4plus8: gain 8.9 pp, crossed-bootstrap 95% CI [4.7, 12.9], Holm p=0.0002.

## Population versus shrinkage calibration

- S1_ITPC4: population FPR RMSE 0.0274; shrinkage RMSE 0.0091; sensitivity change -1.3 pp; success=True.
- S2_Equal4plus8: population FPR RMSE 0.0164; shrinkage RMSE 0.0068; sensitivity change 0.0 pp; success=True.
- S3_RidgeComplex4plus8: population FPR RMSE 0.0088; shrinkage RMSE 0.0022; sensitivity change 0.4 pp; success=True.
- S4_RegionalMonotone4plus8: population FPR RMSE 0.0124; shrinkage RMSE 0.0045; sensitivity change 0.2 pp; success=True.

Participant adaptation used only null trajectories from the held participant's other 14 conditions. The
evaluated condition was excluded both from local boundary estimation and null-only pseudo-LOPO weight selection.

## Separate condition-level endpoint

- C1_ConditionITPC4: 58.4% sensitivity.
- C2_ConditionEqual4plus8: 74.7% sensitivity.
- C3_RegionalMonotone4plus8: 79.5% sensitivity.
- C4_PreviousRegionalMonotone4H: 82.1% sensitivity.

The condition analysis aggregates three independently scored streams. It is not directly comparable with
the stream-level primary endpoint.

## Negative-control scope

External negative status: **external negatives unavailable**.

The six frequency families are off-target controls, not genuine sham or response-absent recordings. Therefore
operational FPR is not claimed.

## Figures

- `figures/sequential_detection.png`
- `figures/calibration_comparison.png`
- `figures/condition_benchmark.png`
- `figures/off_target_controls.png`

All thresholds, fold records, model choices, scores, calibration comparisons, and control results are in
the exported CSV tables. Existing study and manuscript outputs were not modified.
