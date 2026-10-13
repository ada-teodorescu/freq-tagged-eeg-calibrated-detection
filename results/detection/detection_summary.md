# Full ITPC algorithm benchmark

Analysis ID: `fd67653dead6`

## Prespecified comparison

- Baseline trials to 80%: `None`.
- D8 candidate trials to 80%: `None`.
- Candidate held-out FPR: `0.0501`.
- Crossed-bootstrap gain CI: `[0.03553333333333333, 0.16977777777777778]`.
- Condition sign-flip p-value: `9.999e-05`.
- All four success criteria passed: **False**.

D2 and D3 are reported as monotonic recalibrations of D1 at fixed K, not as new sensitivity methods.

## Provenance boundary

The archived FFT detector screen (`stored_matlab`) and time-domain estimator/spatial screen (`public_set_fdt`) are reported separately. Their scores are never pooled.

## Commands

```bash
uv run cucu-detection list
uv run cucu-detection run 03_detector_screen --workers 1
uv run cucu-detection run-all --workers 1
uv run cucu-detection status --watch
```
