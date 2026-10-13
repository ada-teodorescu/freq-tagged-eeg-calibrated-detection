# Independent External-Negative Validation

Analysis ID: `2341d3cd4a7a`

## Locked primary and secondary results

- ldaep_jittered_auditory, static: FPR 6.79% (cluster 95% CI 3.70--10.49%; one-sided upper 9.88%).
- ldaep_jittered_auditory, sequential: FPR 3.09% (cluster 95% CI 0.62--5.56%; one-sided upper 5.56%).
- rest_eyes_open, static: FPR 2.90% (cluster 95% CI 0.00--7.25%; one-sided upper 5.80%).
- rest_eyes_open, sequential: FPR 1.45% (cluster 95% CI 0.00--4.35%; one-sided upper 4.35%).
- rest_eyes_closed, static: FPR 5.80% (cluster 95% CI 1.45--11.59%; one-sided upper 10.14%).
- rest_eyes_closed, sequential: FPR 5.80% (cluster 95% CI 1.45--11.59%; one-sided upper 11.59%).

Primary jittered-auditory static pass: **True**.
Primary jittered-auditory sequential pass: **True**.
Minimum primary cohort satisfied: **True**.
Mandatory resting cohorts satisfied: **True**.

## Submission gate

Both primary upper confidence bounds must be below 10%, and the locked minimum cohorts must be present.
Submission gate passed: **True**. Manuscript replacement allowed: **True**.

## Montage limitation

The BioSemi-64 source lacks TP9, TP10, PO9, and PO10. These four Cucu target sites were reconstructed by
locked spherical interpolation from the full source montage. Dataset-marked bad targets and every interpolation
or exclusion are exported. This is an independent-cohort/device transfer test, not a within-Cucu sham.

## Guardrail

Passing supports transfer to this independent jittered-auditory cohort, not universal or clinical operational FPR.

If the gate fails, the result is preserved and no preprocessing, model, threshold, boundary, or margin is retuned.
