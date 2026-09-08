[中文](README_zh.md) | **English**

# IMU EEG MobileBCI Denoising Benchmark

This project benchmarks **ASR** (Artifact Subspace Reconstruction) and an **iCanClean-style CCA** (Canonical Correlation Analysis) regression cleaner on the MobileBCI BrainVision EEG/IMU dataset.

## Method Notes

### ASR

ASR is implemented as an artifact-subspace reconstruction cleaner:

1. **Calibration**: ERP training data (`ses-01`) is band-pass filtered (0.5–45 Hz), windowed, and the cleanest windows (lowest RMS, bottom 25% quantile) are selected to compute a reference covariance per subject.
2. **Transform**: Each movement recording is windowed and whitened against the reference covariance. Eigenvalues of the whitened covariance exceeding the cutoff threshold identify high-variance (artifact) subspaces, which are reconstructed from the remaining good subspace.

### iCanClean

iCanClean is implemented as a CCA-based noise-reference cleaner:

1. **Decomposition**: EEG and IMU motion channels are band-pass filtered to 0.5–15 Hz and standardized. A CCA decomposition is computed via SVD of the whitened cross-covariance matrix, yielding canonical correlation coefficients and spatial filters.
2. **Regression removal**: Components whose canonical correlation exceeds the threshold are selected. Their regression contribution to the broadband EEG is estimated via least-squares projection and subtracted from the original signal.

## Session Mapping

| Session | Meaning | Speed |
| --- | --- | --- |
| `ses-01` | ERP training session | training |
| `ses-02` | standing | 0.0 m/s |
| `ses-03` | slow walking | 0.8 m/s |
| `ses-04` | fast walking | 1.6 m/s |
| `ses-05` | slight running | 2.0 m/s |

## What It Produces

Running the main pipeline creates:

- Raw EEG vs IMU coherence by speed
- ASR and iCanClean gait-frequency power reduction curves
- ERP preservation at Pz
- Artifact reduction vs signal preservation trade-off
- ASR cutoff sensitivity
- ERP LDA AUC and SSVEP CCA accuracy by speed

Running the supplementary experiments creates:

- Default robust metrics (ASR vs iCanClean) with paired Wilcoxon tests
- ASR parameter sweep across cutoffs [3, 5, 8, 10, 15, 20, 30]
- iCanClean IMU reference ablation (acc, gyro, acc+gyro, magnitude, enhanced, enhanced+lagged)
- iCanClean parameter sweep (correlation threshold × max components)
- Preservation-constrained best-config selection

## Install

```bash
python -m pip install -r requirements.txt
```

Optional editable install:

```bash
python -m pip install -e .
```

## Run

### Main pipeline

Full run (all subjects found in the data directory):

```bash
python run_pipeline.py --data-dir Motion_eeg_data --out-dir outputs
```

Smoke test on `sub-01` ERP only:

```bash
python run_pipeline.py --quick --tasks ERP --out-dir outputs_quick
```

Run selected subjects:

```bash
python run_pipeline.py --subjects sub-01 sub-02 --out-dir outputs_sub01_02
```

Change ASR settings:

```bash
python run_pipeline.py --asr-cutoff 8 --asr-sensitivity-cutoffs 3 5 8 10 15 20
```

### Supplementary experiments

```bash
python run_supplementary_experiments.py --data-dir Motion_eeg_data --out-dir outputs_supplement
```

Smoke test on `sub-01`:

```bash
python run_supplementary_experiments.py --quick --data-dir Motion_eeg_data --out-dir outputs_supplement_quick
```

## Data Sources

- Dataset repository: https://github.com/ChenfeiJiangSICAU/MobileBCI_Data
- Dataset paper: Lee et al., *Scientific Data* 2021, DOI `10.1038/s41597-021-01094-4`

## Project Structure

```
run_pipeline.py                     # Main pipeline entry point
run_supplementary_experiments.py     # Supplementary experiments entry point
requirements.txt
pyproject.toml
src/grf_eeg/
    __init__.py
    config.py        # Session-speed mapping, channel lists, pipeline configuration
    dataio.py         # BrainVision reader, recording discovery, channel selection
    sigproc.py        # Butterworth filter, covariance, PSD, gait frequency estimation
    denoise.py        # ASR model and iCanClean CCA decomposition
    metrics.py        # ERP preservation, artifact reduction, coherence, power reduction
    bci.py            # ERP LDA classifier and SSVEP CCA accuracy
    plots.py          # Visualization functions for the main pipeline
    pipeline.py       # Main benchmark pipeline
    experiments.py    # Supplementary parameter sweep experiments
```

## License

MIT
