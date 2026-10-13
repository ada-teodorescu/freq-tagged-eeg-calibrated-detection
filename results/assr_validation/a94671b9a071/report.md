# Independent positive ASSR validation

The prospectively locked external-positive analysis used 37 typically developing participants from OpenNeuro ds006780 version 1.0.0. The primary 40-Hz analysis used the first 170 standard trials in 17 chronological blocks of ten; the secondary 27-Hz analysis used the first 160 standards in 16 blocks. Oddballs, responses, block markers, and surplus standards were excluded.

Continuous 64-channel EEG was zero-phase filtered from 1 to 100 Hz, notch filtered at 60 Hz, and average referenced. Hann weighted least squares, rectangular least squares, and fourth-order Hilbert coefficients with a +/-4-Hz band were evaluated on the identical [0.05, 0.45)-s support. Scores were 2K times equal-channel ITPC-squared.

Thresholds and sequential boundaries were calibrated using only other participants' phase-rotation nulls. The primary interval used 10000 paired participant-bootstrap draws. Generalization required at least 35 participants, a strictly positive lower 95% confidence bound for Hilbert minus Hann at K=10, and Hilbert held-out FPR no greater than 5.5%.

# Results

The primary 40-Hz Hilbert-minus-Hann gain was 7.9% (95% CI 4.6% to 11.8%). Hilbert held-out FPR was 5.0%. The predefined external-positive gate **passed**.

The Hann-minus-Hilbert restricted mean stopping difference was 0.44 trials (95% CI 0.28 to 0.60). Full estimator, frequency, sequential, and artifact-sensitivity results are supplied in the exported tables.

# Limitations

This dataset tests estimator behavior on short auditory steady-state responses in children. It does not reproduce the syllable paradigm, its four-harmonic feature space, the learned regional detector, or its clinical context. The artifact-rejection analysis is explicitly secondary and cannot change the primary decision. Equal channel weighting avoids learned spatial leakage but may dilute focal responses.
