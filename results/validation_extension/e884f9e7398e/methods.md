# Methods: robustness and generalization extension

This post-hoc extension preserved the frozen confirmatory inputs and compared matched-support coefficient estimators over 0.5-4.5 s. All learned comparisons used the same five-region, four-harmonic features, temporal-shift nulls, participant-held-out fitting, and empirical 5% null calibration. Mechanism contrasts used participant-by-condition crossed bootstrap intervals and participant-level sign-flip tests with Holm adjustment.

Sequential analyses used K=2 through 10, Pocock-style familywise boundaries, simultaneous crossed-bootstrap bands, and restricted mean stopping K with non-detections assigned K=11. Double-holdout folds excluded both the tested participant and tested condition from every fitting and calibration step. The transparent detector equally averaged all 20 regional-by-harmonic 2K R-squared features and used no fitted weights.

External robustness preserved the original ds007615 gate. It added complete-case artifact summaries, a detector trained on the 28 genuinely shared channels, and 100 deterministic non-overlapping raw-window selections per recording.
