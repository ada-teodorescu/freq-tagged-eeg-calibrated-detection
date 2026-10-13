# Confirmatory Cucu Detector Study

Analysis ID: `4d6a54f6f902`

## Locked three-hypothesis confirmatory family

- H2_Hilbert4s_Regional4H vs H1_Hann4s_Regional4H: gain 10.1 pp, crossed-bootstrap 95% CI [4.3, 17.3], Holm p=0.0006999, FPR=5.04%, pass=True.
- TRCA_S2_Grouped vs S2_Equal4plus8: gain -18.2 pp, crossed-bootstrap 95% CI [-28.1, -9.2], Holm p=0.0003, FPR=5.01%, pass=False.
- TRCA_S2_Grouped vs S4_RegionalMonotone4plus8: gain -26.8 pp, crossed-bootstrap 95% CI [-36.2, -18.0], Holm p=0.0003, FPR=5.01%, pass=False.

Each pass criterion requires the ordinary crossed-bootstrap 95% CI entirely above zero, the
Holm-adjusted sign-flip p-value below .05, and held-out surrogate FPR inside the locked 4.5--5.5%
interval. Multiplicity is adjusted across exactly these three hypotheses.

## Secondary comparisons (excluded from the Holm family)

- H2_Hilbert4s_Regional4H vs H3_Hann4p5s_Regional4H: gain 5.3 pp, crossed-bootstrap 95% CI [1.1, 10.7] (secondary; excluded from the confirmatory Holm family).
- TRCA_S2_Grouped vs S4_RegionalMonotone4plus8_shrinkage: gain -27.0 pp, crossed-bootstrap 95% CI [-35.7, -18.5] (secondary; excluded from the confirmatory Holm family).

The H2-vs-H3 comparison is duration-confounded (H3 has 0.5s more data than H2). The sequential
condition-level trajectories reported separately in `hilbert_sequential_*` tables are Pocock
familywise-corrected and are **not** comparable to the static single-look K=10 numbers above, or to
the historical C4 figure (82.1%) from `cucu_advanced_study`. Repeated-look correction typically lowers
the K=10 boundary, while cumulative early crossings make it a distinct endpoint rather than a strict
numerical transformation of the single-look result.

## TRCA classification

TRCA+S2 beats population-calibrated S2: **False**. TRCA+S2 beats
population-calibrated S4: **False**. Classification: **no_confirmed_benefit**
(useful_spatial_improvement / new_stream_leader / no_confirmed_benefit -- a genuine improvement over
the fixed detector is acknowledged even without also beating S4).

TRCA is a subject-calibrated detector: its spatial filter was fit per participant via
leave-one-condition-out (42 training streams from the held participant's other 14 conditions), unlike
S2/S4's population calibration. The TRCA-vs-shrinkage-S4 secondary comparison above probes calibration
adaptation, but is not fully information-matched because the fitted spatial models differ.

## Stop-and-report gate

Gate triggered (neither H2 nor TRCA_S2 passed its locked confirmatory comparison): **False**.
At least one candidate passed its locked confirmatory comparison; see the classification above for which.

## Baselines this study compares against

Locked `cucu_advanced_study` baseline (`outputs\advanced_study\8bf2fe56f4b9`): S2_Equal4plus8 35.5%,
S4_RegionalMonotone4plus8 44.4% (stream-level sequential detection by K=10);
C4_PreviousRegionalMonotone4H 82.1% (condition-level, single-look K=10). This study recomputes S2 and
S4 fresh in its own harness rather than reusing those exported numbers directly, so the population
comparisons above are apples-to-apples within this study's own folds and nulls.

## Figures

- `figures/confirmatory_family.png`
- `figures/trca_sequential.png`

## Guardrail

Surrogate temporal-shift nulls calibrate familywise FPR; they are not genuine sham recordings, and operational FPR is not claimed beyond this study's own surrogate-based calibration.

All thresholds, fold records, model coefficients, scores, calibration comparisons, and control results
are in the exported CSV tables. Existing study packages (`cucu_detection`, `cucu_accuracy_study`,
`cucu_algorithm_study`, `cucu_advanced_study`) and their exported results were not modified.
