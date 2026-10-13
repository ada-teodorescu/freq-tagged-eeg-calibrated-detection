# Calibrated detection of frequency-tagged auditory EEG

Analysis source and curated results for **“Calibrated Detection of Frequency-Tagged Auditory EEG: Comparing Processing Pipelines Across Datasets.”** The code evaluates Hann-weighted Fourier, rectangular least-squares, and Hilbert processing with surrogate-calibrated detection. The reported syllable comparison was frozen after exploratory development on the same dataset; its intervals do not account for that earlier selection. The external negative-control and ASSR analyses use separate public datasets.

## Contents

- `src/`: analysis pipelines, estimators, calibration, statistics, and export code. `eeg_core` contains shared numerical helpers; `eeg_exposure_study` supplies fitting functions used by the restricted-input support study.
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

Obtain the Cucu raw and processed EEG from the [original study's OSF project](https://osf.io/3c6tv/) and place the SET/FDT files under `data/Pre-processed EEGLAB datasets/`. For source-product replication, place the [original study's code repository](https://github.com/phonemes-and-speech-entrainment/cv_syllables) under `cv_syllables-master/`. Neither is fetched by the scripts below. The independent datasets are [OpenNeuro ds007615](https://openneuro.org/datasets/ds007615) and [OpenNeuro ds006780](https://openneuro.org/datasets/ds006780). Place their dataset trees under `data/external/`; the recordings are large and are not redistributed here.

## Helper scripts

Run these from the repository root when needed. They do not run during installation or when the repository is cloned.

| Script | Purpose and prerequisite |
| --- | --- |
| `download_ds007615.ps1` | Download the external negative-control EEG into `data/external/ds007615/`. An existing dataset inventory is used when present. |
| `download_ds006780_assr.ps1` | Fill in the ASSR BDF recordings for the selected cohort in an existing ds006780 dataset clone. It checks file sizes and SHA-256 hashes. |
| `repair_ds007615_annex_prefixes.ps1` | Optional repair for ds007615 files whose first bytes contain a Git annex pointer. It verifies each repaired file against the annex hash. |
| `region_vs_global_ablation.py` | Compare five-region features with whole-scalp averaging at the matched static look. Requires the confirmatory coefficient checkpoints under `outputs/` and writes details to `.tmp/`. |
| `run_assr_validation.ps1` | PowerShell shortcut for ASSR CLI commands such as `doctor`, `list`, `run-all`, and `export`. |
| `run_validation_extension.ps1` | PowerShell shortcut for the validation-extension CLI. Requires the project environment and its installed command. |

By default, the download and repair scripts modify local files under the ignored `data/` tree only when explicitly run; their path arguments can change that location. The `run_*.ps1` scripts are optional; the same analysis CLIs are listed in `pyproject.toml`.

The configuration files specify the full analysis path. For example:

```powershell
uv run eeg-confirmatory-study doctor
uv run eeg-confirmatory-study list
uv run eeg-assr-validation doctor
uv run eeg-external-validation doctor
```

Earlier stages produce checkpoints required by later stages. The exported `results/` snapshots support inspection of the published numbers but do not replace the `outputs/` checkpoints for a full rerun. See each CLI's `--help`, `list`, and `doctor` commands before a rerun.

## Analysis sequence

1. `eeg-replication` and `eeg-raw-replication` audit the source products and raw reconstruction.
2. `eeg-detection`, `eeg-improvement`, `eeg-algorithm-study`, `eeg-accuracy-study`, and `eeg-advanced-study` document exploratory method development.
3. `eeg-confirmatory-study` performs the frozen-after-exploration syllable comparison.
4. `eeg-external-validation` evaluates the frozen detector on presumed response-negative EEG.
5. `eeg-assr-validation` performs the independent ASSR comparison.
6. `eeg-validation-extension` and `eeg-support-study` provide exploratory estimator and temporal-context controls.

The word *detection rate* here means calibrated threshold exceedance in presumed response-positive recordings. It is not diagnostic sensitivity against independently verified labels.
