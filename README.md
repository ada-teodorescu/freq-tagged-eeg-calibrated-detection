# Calibrated detection of frequency-tagged auditory EEG

Analysis source and curated results for **“Calibrated Detection of Frequency-Tagged Auditory EEG: Comparing Processing Pipelines Across Datasets.”** The code evaluates Hann-weighted Fourier, rectangular least-squares, and Hilbert processing with surrogate-calibrated detection. The reported syllable comparison was frozen after exploratory development on the same dataset; its intervals do not account for that earlier selection. The external negative-control and ASSR analyses use separate public datasets.

## Contents

- `src/`: analysis pipelines, estimators, calibration, statistics, and export code. `cucu_article` contains shared numerical helpers used by these pipelines.
- `*_config.toml`: analysis settings and input/output paths. Run identifiers are content hashes; a fresh run may have a different identifier from a published snapshot.
- `scripts/`: dataset download and analysis helper scripts.
- `tests/`: unit and integration tests for analysis code.
- `results/`: selected result records, tables, figures, model artifact, and provenance. See [results/README.md](results/README.md) for the reported run identifiers.

Manuscript source, drafts, build scripts, and raw EEG data are outside this repository. Generated checkpoints and full intermediate runs belong under the ignored `outputs/` directory.

## Environment and data

Python 3.11 or newer is required. With [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --extra raw --extra dev
uv run pytest
```

Place the public Cucu SET/FDT release under `data/Pre-processed EEGLAB datasets/` and the original study repository, when needed for replication, under `cv_syllables-master/`. The independent datasets are [OpenNeuro ds007615](https://openneuro.org/datasets/ds007615) and [OpenNeuro ds006780](https://openneuro.org/datasets/ds006780). The download scripts place these under `data/external/`; those recordings are large and are not redistributed here.

The configuration files specify the full analysis path. For example:

```powershell
uv run cucu-confirmatory-study doctor
uv run cucu-confirmatory-study list
uv run cucu-assr-validation doctor
uv run cucu-external-validation doctor
```

Earlier stages produce checkpoints required by later stages. The exported `results/` snapshots support inspection of the published numbers but do not replace the `outputs/` checkpoints for a full rerun. See each CLI's `--help`, `list`, and `doctor` commands before a rerun. The download scripts are `scripts/download_ds007615.ps1` and `scripts/download_ds006780_assr.ps1`.

## Analysis sequence

1. `cucu-replication` and `cucu-raw-replication` audit the source products and raw reconstruction.
2. `cucu-detection`, `cucu-improvement`, `cucu-algorithm-study`, `cucu-accuracy-study`, and `cucu-advanced-study` document exploratory method development.
3. `cucu-confirmatory-study` performs the frozen-after-exploration syllable comparison.
4. `cucu-external-validation` evaluates the frozen detector on presumed response-negative EEG.
5. `cucu-assr-validation` performs the independent ASSR comparison.
6. `cucu-validation-extension` and `cucu-support-study` provide exploratory estimator and temporal-context controls.

The word *detection rate* here means calibrated threshold exceedance in presumed response-positive recordings. It is not diagnostic sensitivity against independently verified labels.
