# Exact Cucu replication and robustness summary

Analysis ID: `67167da6a1d8`

## Provenance

- Stored FFT -> ITPC exact gate: **True**; maximum error `9.99e-16`.
- Public SET/FDT -> stored MATLAB equivalence: **False**.
- Optimal participant-assignment mean correlation: `0.7416`.
- Stored MATLAB and public time-domain results remain separately labelled whenever the equivalence gate fails.

## Exact archived results

- Paper-reported 25 x 4 PCA: PC1 `76.2027%`, PC2 `20.0788%`.
- Repository-score 1,125 x 4 PCA: PC1 `48.4796%`, PC2 `30.5780%`.
- These PCA definitions are reported separately because they answer different provenance questions.
- Three-group Greenhouse-Geisser p-value from archived R input: `0.000135708`.
- Stored acoustic maximum-derivative latency versus 4-Hz ITPC: `r = -0.9129993`.
- Stored acoustic PCA: PC1 `53.0437%`, PC2 `29.7658%`; `|r| = 0.839659` versus archived ITPC1.
- Exact acoustic claims stop at the archived marker matrix because the original 75 individual syllable WAV files are not local.
- Landmark extraction from the 45 continuous streams is reported separately as reconstruction robustness.

## Robustness analyses

- Hilbert comparison interpretation: **group_level_only**.
- Filter-null configurations flagged: **21**.
- Mean fitted stream frequency: `4.003667 Hz`.
- Frequency result: **published_condition_pattern_robust_to_frequency_offset_test**.

## Reproduction commands

```powershell
uv run cucu-replication list
uv run cucu-replication run-all --workers 1
uv run cucu-replication status
uv run cucu-replication status --watch
```

## Run record

- Configuration: `replication_config.toml`
- Local run: `outputs/replication/runs/67167da6a1d8`
- Input files fingerprinted: `198`
- Package versions: `{"matplotlib": "3.11.1", "numpy": "2.5.2", "pandas": "3.0.5", "platform": "Windows-11-10.0.26200-SP0", "python": "3.12.13", "rich": "15.0.0", "scipy": "1.18.0"}`

Detailed CSV tables and figures are stored beside this report.
