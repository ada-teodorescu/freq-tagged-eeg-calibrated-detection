# Limitations of the extension

- These analyses were designed after observing the original estimator result and must be labelled post-hoc.
- Repeated participant-condition units are not independent; crossed resampling and participant sign flips mitigate but do not eliminate this limitation.
- Alternative external windows use continuous-recording filtering and cumulative complex demodulation; this avoids per-window filter transients but is not numerically identical to filtering every five-second window separately.
- The 28-channel analysis changes the trained detector and is an exploratory transfer sensitivity, not a replacement for the locked external gate.
- OpenNeuro ds006780 tests ASSR estimator generalization at different frequencies and trial durations; it cannot validate the full four-harmonic Cucu syllable detector.
