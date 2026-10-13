# Condition-level accuracy study

The locked target of at least 90% sensitivity at a held-out 5% false-positive rate was **not reached**.

Best primary method: **C5_RegionalMonotone**, sensitivity 81.6%, FPR 5.02%, balanced accuracy 88.3%, AUC 0.957.

All methods operate on the paper-aligned participant-condition unit formed from three streams. Learned models use matched condition-level training and evaluation observations, grouped inner validation, nonnegative coefficients, and training-only thresholds.

The result is an algorithm-development experiment, not a manuscript update. Failure to reach 90% is retained rather than hidden through threshold or endpoint changes.
