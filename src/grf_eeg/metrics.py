from __future__ import annotations

import numpy as np
from scipy import signal

from .sigproc import estimate_gait_frequency, gait_signal, welch_mean_psd


def epoch_data(
    data: np.ndarray,
    events,
    sfreq: float,
    tmin: float,
    tmax: float,
    valid_labels: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_offset = int(round(tmin * sfreq))
    stop_offset = int(round(tmax * sfreq))
    times = np.arange(start_offset, stop_offset) / sfreq
    epochs = []
    labels = []
    for row in events.itertuples(index=False):
        label = int(row.value)
        if valid_labels is not None and label not in valid_labels:
            continue
        start = int(row.onset_sample) + start_offset
        stop = int(row.onset_sample) + stop_offset
        if start < 0 or stop > data.shape[1] or stop <= start:
            continue
        epochs.append(data[:, start:stop])
        labels.append(label)
    if not epochs:
        return np.empty((0, data.shape[0], len(times))), np.array([], dtype=int), times
    return np.stack(epochs, axis=0), np.array(labels, dtype=int), times


def baseline_correct(epochs: np.ndarray, times: np.ndarray, baseline: tuple[float, float]) -> np.ndarray:
    mask = (times >= baseline[0]) & (times <= baseline[1])
    if not np.any(mask):
        return epochs
    base = np.mean(epochs[:, :, mask], axis=2, keepdims=True)
    return epochs - base


def coherence_curve(
    eeg: np.ndarray,
    imu: np.ndarray,
    imu_names: list[str],
    sfreq: float,
    nperseg_s: float = 8.0,
    fmax: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    gait = gait_signal(imu, imu_names)
    nperseg = min(eeg.shape[1], max(64, int(round(nperseg_s * sfreq))))
    curves = []
    freqs = None
    for channel in eeg:
        freqs, coh = signal.coherence(channel, gait, fs=sfreq, nperseg=nperseg)
        curves.append(coh)
    assert freqs is not None
    curves = np.vstack(curves)
    mask = freqs <= fmax
    return freqs[mask], np.nanmean(curves[:, mask], axis=0)


def gait_power_reduction_curve(
    raw_eeg: np.ndarray,
    clean_eeg: np.ndarray,
    sfreq: float,
    nperseg_s: float = 8.0,
    fmax: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    freqs, raw_psd = welch_mean_psd(raw_eeg, sfreq, nperseg_s=nperseg_s, fmax=fmax)
    _, clean_psd = welch_mean_psd(clean_eeg, sfreq, nperseg_s=nperseg_s, fmax=fmax)
    reduction_db = 10.0 * np.log10((raw_psd + 1e-20) / (clean_psd + 1e-20))
    return freqs, reduction_db


def artifact_reduction_at_gait(
    raw_eeg: np.ndarray,
    clean_eeg: np.ndarray,
    imu: np.ndarray,
    imu_names: list[str],
    sfreq: float,
    half_width: float = 0.25,
) -> tuple[float, float]:
    gait_freq = estimate_gait_frequency(imu, imu_names, sfreq)
    freqs, reduction = gait_power_reduction_curve(raw_eeg, clean_eeg, sfreq)
    if not np.isfinite(gait_freq):
        return float(np.nanmean(reduction)), np.nan
    mask = (freqs >= gait_freq - half_width) & (freqs <= gait_freq + half_width)
    if not np.any(mask):
        return float(np.nanmean(reduction)), gait_freq
    return float(np.nanmean(reduction[mask])), gait_freq


def erp_waveforms(
    eeg: np.ndarray,
    events,
    sfreq: float,
    channel_index: int,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    epochs, labels, times = epoch_data(eeg, events, sfreq, -0.2, 0.8, valid_labels={1, 2})
    if epochs.size == 0:
        return times, {}
    epochs = baseline_correct(epochs, times, (-0.2, 0.0))
    waves = {}
    for label in (1, 2):
        mask = labels == label
        if np.any(mask):
            waves[label] = np.mean(epochs[mask, channel_index, :], axis=0)
    return times, waves


def erp_preservation_score(
    raw_eeg: np.ndarray,
    clean_eeg: np.ndarray,
    events,
    sfreq: float,
    channel_index: int,
) -> float:
    times, raw_waves = erp_waveforms(raw_eeg, events, sfreq, channel_index)
    _, clean_waves = erp_waveforms(clean_eeg, events, sfreq, channel_index)
    if 2 not in raw_waves or 2 not in clean_waves:
        return np.nan
    mask = (times >= 0.25) & (times <= 0.55)
    raw_amp = np.max(raw_waves[2][mask])
    clean_amp = np.max(clean_waves[2][mask])
    return float(clean_amp / (raw_amp + 1e-12))


def _safe_ratio(numerator: float, denominator: float, min_abs_denominator: float = 0.5) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan
    if abs(denominator) < min_abs_denominator:
        return np.nan
    return float(numerator / denominator)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return np.nan
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def erp_preservation_metrics(
    raw_eeg: np.ndarray,
    clean_eeg: np.ndarray,
    events,
    sfreq: float,
    channel_index: int,
    p300_window: tuple[float, float] = (0.25, 0.55),
    min_denominator_uv: float = 0.5,
) -> dict[str, float]:
    times, raw_waves = erp_waveforms(raw_eeg, events, sfreq, channel_index)
    _, clean_waves = erp_waveforms(clean_eeg, events, sfreq, channel_index)
    if 1 not in raw_waves or 2 not in raw_waves or 1 not in clean_waves or 2 not in clean_waves:
        return {
            "target_peak_ratio": np.nan,
            "diff_peak_ratio": np.nan,
            "diff_area_ratio": np.nan,
            "diff_ptp_ratio": np.nan,
            "diff_wave_corr": np.nan,
            "raw_target_peak_uv": np.nan,
            "clean_target_peak_uv": np.nan,
            "raw_diff_peak_uv": np.nan,
            "clean_diff_peak_uv": np.nan,
        }

    mask = (times >= p300_window[0]) & (times <= p300_window[1])
    raw_target = raw_waves[2]
    clean_target = clean_waves[2]
    raw_diff = raw_waves[2] - raw_waves[1]
    clean_diff = clean_waves[2] - clean_waves[1]

    raw_target_peak = float(np.max(raw_target[mask]))
    clean_target_peak = float(np.max(clean_target[mask]))
    raw_diff_peak = float(np.max(raw_diff[mask]))
    clean_diff_peak = float(np.max(clean_diff[mask]))
    raw_diff_area = float(np.trapezoid(raw_diff[mask], times[mask]))
    clean_diff_area = float(np.trapezoid(clean_diff[mask], times[mask]))
    raw_diff_ptp = float(np.ptp(raw_diff[mask]))
    clean_diff_ptp = float(np.ptp(clean_diff[mask]))

    return {
        "target_peak_ratio": _safe_ratio(clean_target_peak, raw_target_peak, min_denominator_uv),
        "diff_peak_ratio": _safe_ratio(clean_diff_peak, raw_diff_peak, min_denominator_uv),
        "diff_area_ratio": _safe_ratio(clean_diff_area, raw_diff_area, min_denominator_uv * 0.05),
        "diff_ptp_ratio": _safe_ratio(clean_diff_ptp, raw_diff_ptp, min_denominator_uv),
        "diff_wave_corr": _safe_corr(raw_diff[mask], clean_diff[mask]),
        "raw_target_peak_uv": raw_target_peak,
        "clean_target_peak_uv": clean_target_peak,
        "raw_diff_peak_uv": raw_diff_peak,
        "clean_diff_peak_uv": clean_diff_peak,
    }
