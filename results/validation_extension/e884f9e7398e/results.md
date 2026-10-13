# Results: robustness and generalization extension

The matched estimator contrasts were: H2_Hilbert4s - H1_Hann4s = 10.1% (95% CI 4.3% to 17.3%, Holm p=0.0001); H2_Hilbert4s - M1_RectLS4s = -1.3% (95% CI -6.9% to 3.5%, Holm p=0.4361).

At K=10, the sequential Hilbert-Hann gain was 9.9% (95% CI 4.3% to 16.3%). The reduction in restricted mean stopping K was 0.99 repetitions per stream (95% CI 0.65 to 1.35).

The learned-minus-simple detection gap was 3.7%; its one-sided 95% upper bound was 8.0% against the predeclared 3.0% margin. Non-inferiority was False.

Under participant-and-condition double holdout, the Hilbert-Hann gain was 10.1% (95% CI 4.5% to 17.1%); the predefined generalization rule was True.

Post hoc, exploratory, outside the locked mechanism family: a filter-matched control (the same Hilbert Butterworth bandpass, then sinusoidal regression without the Hilbert transform) reached the same detection decision as full Hilbert on every participant-condition (McNemar discordant=0, p=1; paired gain 0.0%, 95% CI 0.0% to 0.0%).
