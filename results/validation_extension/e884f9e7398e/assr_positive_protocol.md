# Optional independent positive ASSR protocol

Dataset: OpenNeuro ds006780 (https://openneuro.org/datasets/ds006780). This phase is estimator-level validation, not validation of the complete Cucu detector.

- Primary cohort: typically developing participants; require at least 35 eligible recordings.
- Primary endpoint: standard 40-Hz trials; 27 Hz is secondary and oddballs are excluded.
- Filter continuous EEG before epoching; use non-overlapping chronological blocks of ten trials.
- Compare matched Hann, rectangular least squares, and Hilbert fundamental coefficients.
- Hilbert lock for 0.5-s stimuli: order 4, +/-4 Hz, retained [0.05, 0.45] s.
- Use participant-held-out null calibration and participant-cluster inference.
- Generalization requires a positive 95% CI for Hilbert minus Hann at 40 Hz and held-out FPR <=5.5%.
- Execute only after stages 01-06 pass and matched-support results do not reveal an implementation artifact.
