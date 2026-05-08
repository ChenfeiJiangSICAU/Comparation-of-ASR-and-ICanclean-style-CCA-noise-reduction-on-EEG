# IMU EEG MobileBCI Denoising Benchmark

This project benchmarks ASR and an iCanClean-style CCA regression cleaner on the
MobileBCI BrainVision EEG/IMU files from https://github.com/ChenfeiJiangSICAU/MobileBCI_Data.

The default run uses the three local subjects and the dataset speed mapping:
## Method Notes

ASR is implemented as an artifact-subspace reconstruction style cleaner:
ERP calibrates a clean covariance per subject, then each moving-speed
recording is windowed, whitened against that covariance, and high-variance
subspaces above the cutoff are reconstructed.

iCanClean is implemented as a CCA-based noise-reference cleaner:
EEG and IMU motion channels are band-limited to the gait-artifact range,
canonical EEG components correlated with IMU references are identified, and
their regression contribution is removed from the broadband EEG.

These implementations are intended for a reproducible Python benchmark and
parameter comparison. 
| Session | Meaning | Speed |
| --- | --- | --- |
| `ses-01` | ERP training session | training |
| `ses-02` | standing | 0.0 m/s |
| `ses-03` | slow walking | 0.8 m/s |
| `ses-04` | fast walking | 1.6 m/s |
| `ses-05` | slight running | 2.0 m/s |


## What It Produces

Running the pipeline creates:

- raw EEG vs IMU coherence by speed
- ASR and iCanClean gait-frequency power reduction curves
- ERP preservation at Pz
- artifact reduction vs signal preservation trade-off
- ASR cutoff sensitivity
- ERP LDA AUC and SSVEP CCA accuracy by speed


## Install

The current Anaconda environment already has the required packages except MNE,
which is intentionally not used here. If needed:

```powershell
python -m pip install -r requirements.txt
```

Optional editable install:

```powershell
python -m pip install -e .
```

## Run

Full three-subject run:

```powershell
python run_pipeline.py --data-dir Motion_eeg_data --out-dir outputs
```

Smoke test on `sub-01` ERP only:

```powershell
python run_pipeline.py --quick --tasks ERP --out-dir outputs_quick
```

Run selected subjects:

```powershell
python run_pipeline.py --subjects sub-01 sub-02 --out-dir outputs_sub01_02
```

Change ASR settings:

```powershell
python run_pipeline.py --asr-cutoff 8 --asr-sensitivity-cutoffs 3 5 8 10 15 20
```



## Data Sources

- GitHub code reference: https://github.com/ChenfeiJiangSICAU/MobileBCI_Data
- Dataset paper: Lee et al., Scientific Data 2021, DOI `10.1038/s41597-021-01094-4`
