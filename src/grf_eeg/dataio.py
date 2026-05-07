from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import SESSION_SPEEDS


@dataclass(frozen=True)
class BrainVisionHeader:
    data_file: str
    marker_file: str
    n_channels: int
    n_samples: int
    sfreq: float
    channel_names: list[str]
    resolutions: np.ndarray


@dataclass(frozen=True)
class Recording:
    subject: str
    session: str
    task: str
    speed_label: str
    speed_mps: float | None
    vhdr_path: Path
    eeg_path: Path
    vmrk_path: Path
    channels_path: Path
    events_path: Path
    header: BrainVisionHeader
    channels: pd.DataFrame


def _read_text(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace")


def read_vhdr(path: str | Path) -> BrainVisionHeader:
    path = Path(path)
    text = _read_text(path)
    data_file = ""
    marker_file = ""
    n_channels = 0
    n_samples = 0
    sampling_interval_us = 0.0
    names: dict[int, str] = {}
    resolutions: dict[int, float] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "DataFile":
            data_file = value
        elif key == "MarkerFile":
            marker_file = value
        elif key == "NumberOfChannels":
            n_channels = int(value)
        elif key == "DataPoints":
            n_samples = int(value)
        elif key == "SamplingInterval":
            sampling_interval_us = float(value)
        elif key.startswith("Ch"):
            match = re.match(r"Ch(\d+)", key)
            if not match:
                continue
            idx = int(match.group(1))
            parts = value.split(",")
            names[idx] = parts[0].strip()
            try:
                resolutions[idx] = float(parts[2])
            except (IndexError, ValueError):
                resolutions[idx] = 1.0

    if not data_file or not marker_file or n_channels <= 0:
        raise ValueError(f"Invalid BrainVision header: {path}")
    sfreq = 1_000_000.0 / sampling_interval_us
    ordered_names = [names[i] for i in range(1, n_channels + 1)]
    ordered_res = np.array([resolutions[i] for i in range(1, n_channels + 1)], dtype=np.float32)
    return BrainVisionHeader(
        data_file=data_file,
        marker_file=marker_file,
        n_channels=n_channels,
        n_samples=n_samples,
        sfreq=sfreq,
        channel_names=ordered_names,
        resolutions=ordered_res,
    )


def read_brainvision_data(recording: Recording) -> np.ndarray:
    raw = np.fromfile(recording.eeg_path, dtype="<i2")
    n_channels = recording.header.n_channels
    expected = n_channels * recording.header.n_samples
    if raw.size < expected:
        raise ValueError(f"{recording.eeg_path} is shorter than expected")
    if raw.size != expected:
        raw = raw[: raw.size - (raw.size % n_channels)]
    data = raw.reshape(-1, n_channels).T.astype(np.float32, copy=False)
    data *= recording.header.resolutions[:, None]
    return data


def read_events(recording: Recording, n_samples: int | None = None) -> pd.DataFrame:
    if recording.events_path.exists():
        events = pd.read_csv(recording.events_path, sep="\t")
        events = events.rename(columns={c: c.strip().lower() for c in events.columns})
    else:
        events = read_vmrk(recording.vmrk_path)
    events["value"] = events["value"].astype(int)
    onset = events["onset"].astype(float).to_numpy()
    sfreq = recording.header.sfreq
    limit_seconds = (n_samples or recording.header.n_samples) / sfreq
    if len(onset) and np.nanmax(onset) > limit_seconds + 1:
        onset_samples = np.rint(onset).astype(int)
    else:
        onset_samples = np.rint(onset * sfreq).astype(int)
    events["onset_sample"] = onset_samples
    return events[["onset", "onset_sample", "duration", "value"]]


def read_vmrk(path: str | Path) -> pd.DataFrame:
    rows = []
    for line in _read_text(Path(path)).splitlines():
        line = line.strip()
        if not line.startswith("Mk") or "=" not in line:
            continue
        _, value = line.split("=", 1)
        parts = value.split(",")
        if len(parts) < 4 or parts[0] != "Stimulus":
            continue
        label = parts[1].strip().replace("S", "").strip()
        if not label:
            continue
        rows.append(
            {
                "onset": int(parts[2]),
                "duration": int(parts[3]),
                "value": int(label),
            }
        )
    return pd.DataFrame(rows)


def discover_recordings(
    data_dir: str | Path,
    subjects: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
) -> list[Recording]:
    data_dir = Path(data_dir)
    subject_filter = {normalise_subject(s) for s in subjects} if subjects else None
    task_filter = {t.upper() for t in tasks} if tasks else None
    out: list[Recording] = []
    pattern = re.compile(r"(sub-\d+)_ses-(\d+)_task-(ERP|SSVEP)_eeg\.vhdr$", re.IGNORECASE)

    for vhdr in sorted(data_dir.rglob("*_task-*_eeg.vhdr")):
        match = pattern.match(vhdr.name)
        if not match:
            continue
        subject = match.group(1)
        session = f"ses-{int(match.group(2)):02d}"
        task = match.group(3).upper()
        if subject_filter and subject not in subject_filter:
            continue
        if task_filter and task not in task_filter:
            continue
        header = read_vhdr(vhdr)
        stem = vhdr.name.removesuffix("_eeg.vhdr")
        channels_path = vhdr.with_name(f"{stem}_channels.tsv")
        events_path = vhdr.with_name(f"{stem}_events.tsv")
        channels = pd.read_csv(channels_path, sep="\t")
        label, speed = SESSION_SPEEDS.get(session, (session, None))
        out.append(
            Recording(
                subject=subject,
                session=session,
                task=task,
                speed_label=label,
                speed_mps=speed,
                vhdr_path=vhdr,
                eeg_path=vhdr.with_name(header.data_file),
                vmrk_path=vhdr.with_name(header.marker_file),
                channels_path=channels_path,
                events_path=events_path,
                header=header,
                channels=channels,
            )
        )
    return out


def normalise_subject(subject: str) -> str:
    match = re.search(r"(\d+)", subject)
    if not match:
        raise ValueError(f"Cannot parse subject id from {subject!r}")
    return f"sub-{int(match.group(1)):02d}"


def channel_indices(recording: Recording, group: str, good_only: bool = False) -> list[int]:
    names = recording.channels["name"].astype(str).tolist()
    types = recording.channels["type"].astype(str).str.upper().tolist()
    status = recording.channels.get("status", pd.Series(["good"] * len(names))).astype(str).str.lower()
    group = group.lower()
    if group == "eeg":
        selected = [i for i, t in enumerate(types) if t == "EEG"]
    elif group == "imu":
        selected = [i for i, t in enumerate(types) if t == "IMU"]
    elif group == "imu_motion":
        selected = [
            i
            for i, (name, t) in enumerate(zip(names, types))
            if t == "IMU" and ("acc" in name.lower() or "gyro" in name.lower())
        ]
    elif group == "scalp":
        selected = list(range(min(32, len(names))))
    elif group == "ear":
        selected = list(range(32, min(46, len(names))))
    else:
        wanted = {item.strip() for item in group.split(",")}
        selected = [i for i, name in enumerate(names) if name in wanted]
    if good_only:
        selected = [i for i in selected if status.iloc[i] == "good"]
    return selected


def pick_named(recording: Recording, names: Iterable[str], good_only: bool = True) -> list[int]:
    wanted = set(names)
    all_names = recording.channels["name"].astype(str).tolist()
    status = recording.channels.get("status", pd.Series(["good"] * len(all_names))).astype(str).str.lower()
    out = [i for i, name in enumerate(all_names) if name in wanted]
    if good_only:
        out = [i for i in out if status.iloc[i] == "good"]
    return out
