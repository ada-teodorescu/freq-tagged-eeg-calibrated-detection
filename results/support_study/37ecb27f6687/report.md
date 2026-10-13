# Restricted-input Hilbert comparison

Analysis: `37ecb27f6687` (exploratory)

Static K=10 per stream; three streams per participant-condition. Each estimator has its own participant-held-out model, standardizer, and training-null threshold.

## Detection and calibration

- hann: detection 76.8%; held-out surrogate FPR 5.01%.
- rectangular: detection 88.3%; held-out surrogate FPR 5.01%.
- existing_hilbert: detection 86.9%; held-out surrogate FPR 5.04%.
- restricted_odd: detection 85.3%; held-out surrogate FPR 5.01%.
- restricted_even: detection 85.1%; held-out surrogate FPR 5.01%.

## Paired contrasts

- primary_exploratory: restricted_odd minus hann: 8.53 pp (pointwise 95% CI 2.93 to 15.47).
- primary_exploratory: restricted_odd minus rectangular: -2.93 pp (pointwise 95% CI -9.07 to 1.87).
- primary_exploratory: existing_hilbert minus restricted_odd: 1.60 pp (pointwise 95% CI -1.07 to 4.80).
- padding_sensitivity: restricted_even minus hann: 8.27 pp (pointwise 95% CI 3.20 to 14.40).
- padding_sensitivity: restricted_even minus rectangular: -3.20 pp (pointwise 95% CI -8.53 to 1.33).
- padding_sensitivity: existing_hilbert minus restricted_even: 1.87 pp (pointwise 95% CI -0.80 to 5.33).
- padding_sensitivity: restricted_even minus restricted_odd: -0.27 pp (pointwise 95% CI -4.80 to 4.00).

## Interpretation and limits

Odd-reflection restricted Hilbert is the primary control. Even reflection is a fixed sensitivity analysis, not a candidate to select by performance. Both crop to samples [500,4500) before any new filtering or analytic transformation. Synthetic margins contain no additional recorded EEG. See design.json for both filter and FFT boundaries.

An advantage over Hann with restricted input shows that additional estimator-input samples are unnecessary for that advantage under the specified boundary handling. The rectangular contrast addresses an untapered comparator. Existing minus restricted Hilbert combines the effects of recorded context and replacing that context with synthetic boundary assumptions; it does not isolate the Hilbert transform itself.

The public EEG was already preprocessed. Restriction applies to the inputs of the additional target-frequency estimator, not to the temporal support of all upstream preprocessing.

Review surrogate FPRs before interpreting detection gains. Pointwise crossed-bootstrap intervals resample participants and conditions with pairing preserved, hold fitted models fixed, and have no multiplicity adjustment. No new confirmatory gate or equivalence claim is made. No behavioral learning or perception endpoint is measured.

Source coefficients are checked against recomputation from every stream. Large EEG input identity uses size/mtime metadata; configuration, source checkpoints, and code use SHA-256. Outputs are separate from the original study and manuscript.
