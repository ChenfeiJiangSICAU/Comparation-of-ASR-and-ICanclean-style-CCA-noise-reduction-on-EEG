from __future__ import annotations

from dataclasses import dataclass


SESSION_SPEEDS = {
    "ses-01": ("train", None),
    "ses-02": ("0.0 m/s", 0.0),
    "ses-03": ("0.8 m/s", 0.8),
    "ses-04": ("1.6 m/s", 1.6),
    "ses-05": ("2.0 m/s", 2.0),
}

SPEED_ORDER = ["0.0 m/s", "0.8 m/s", "1.6 m/s", "2.0 m/s"]

ERP_LABELS = {1: "non-target", 2: "target"}
SSVEP_FREQS = {11: 60.0 / 11.0, 12: 60.0 / 7.0, 13: 12.0}

SCALP_CHANNELS = [
    "Fp1",
    "Fp2",
    "AFz",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "C3",
    "Cz",
    "C4",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "O1",
    "Oz",
    "O2",
]

EAR_CHANNELS = [
    "L1",
    "L2",
    "L4",
    "L5",
    "L6",
    "L7",
    "L9",
    "L10",
    "R1",
    "R2",
    "R4",
    "R5",
    "R7",
    "R8",
]

ERP_CHANNELS = ["Pz", "CP1", "CP2", "P3", "P4", "POz"]
SSVEP_CHANNELS = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: str = "Motion_eeg_data"
    out_dir: str = "outputs"
    asr_cutoff: float = 10.0
    asr_window_s: float = 1.0
    asr_step_s: float = 0.5
    ican_corr_threshold: float = 0.22
    ican_max_components: int = 6
    nperseg_s: float = 8.0
