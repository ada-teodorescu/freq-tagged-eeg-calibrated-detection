# Focused Cucu Algorithm Follow-up

This analysis is separate from the completed preregistered detector benchmark.

## Changes tested

- D8 with all harmonics, 4+8 Hz only, and weights fitted on training participants only.
- Corrected Hilbert coefficients using complex demodulation before time averaging.
- DPSS line estimates and Thomson F statistics calibrated against neighboring frequencies.
- Only S0 and cross-fitted S1; RESS, DSS, and CCA were not rerun.

## Interpretation rule

The stored MATLAB and public SET/FDT branches remain separate because source equivalence failed.

Machine-readable results: `claim_status.json` and `tables/focused_trials_auc_fpr.csv`.
