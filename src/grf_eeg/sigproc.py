from __future__ import annotations

import numpy as np
from scipy import signal


def demean(data: np.ndarray, axis: int = -1) -> np.ndarray:
    return data - np.mean(data, axis=axis, keepdims=True)


def standardize(data: np.ndarray, axis: int = -1, eps: float = 1e-12, copy: bool = True) -> np.ndarray:
    out = np.array(data, dtype=np.result_type(data.dtype, np.float32), copy=copy)
    if axis == -1 and out.ndim >= 2 and out.size > 1_000_000:
        flat = out.reshape(-1, out.shape[-1])
        for channel in flat:
            channel -= np.mean(channel)
            scale = np.sqrt(np.mean(channel * channel))
            channel /= max(scale, eps)
        return out
    out -= np.mean(out, axis=axis, keepdims=True)
    scale = np.std(out, axis=axis, keepdims=True)
    out /= np.maximum(scale, eps)
    return out


def butter_filter(
    data: np.ndarray,
    sfreq: float,
    l_freq: float | None = None,
    h_freq: float | None = None,
    order: int = 4,
) -> np.ndarray:
    nyq = sfreq / 2.0
    if h_freq is not None:
        h_freq = min(h_freq, nyq - 1e-3)
    if l_freq is None and h_freq is None:
        return data
    if l_freq is not None and h_freq is not None:
        sos = signal.butter(order, [l_freq / nyq, h_freq / nyq], btype="bandpass", output="sos")
    elif l_freq is not None:
        sos = signal.butter(order, l_freq / nyq, btype="highpass", output="sos")
    else:
        sos = signal.butter(order, h_freq / nyq, btype="lowpass", output="sos")
    if data.ndim >= 2 and data.size > 1_000_000:
        filtered = np.empty_like(data, dtype=np.result_type(data.dtype, np.float32))
        flat_in = data.reshape(-1, data.shape[-1])
        flat_out = filtered.reshape(-1, data.shape[-1])
        for idx, channel in enumerate(flat_in):
            flat_out[idx] = signal.sosfiltfilt(sos, channel)
        return filtered
    return signal.sosfiltfilt(sos, data, axis=-1)


def invsqrtm(cov: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, eps)
    invsqrt = (vecs / np.sqrt(vals)) @ vecs.T
    sqrt = (vecs * np.sqrt(vals)) @ vecs.T
    return invsqrt, sqrt


def regularized_cov(data: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    cov = np.cov(data)
    trace = np.trace(cov) / max(cov.shape[0], 1)
    return cov + np.eye(cov.shape[0]) * trace * reg


def window_slices(n_samples: int, window: int, step: int) -> list[slice]:
    if n_samples <= window:
        return [slice(0, n_samples)]
    starts = list(range(0, n_samples - window + 1, step))
    if starts[-1] + window < n_samples:
        starts.append(n_samples - window)
    return [slice(start, start + window) for start in starts]


def welch_mean_psd(
    data: np.ndarray,
    sfreq: float,
    nperseg_s: float = 8.0,
    fmax: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    nperseg = max(64, int(round(nperseg_s * sfreq)))
    nperseg = min(nperseg, data.shape[-1])
    if data.ndim >= 2:
        psds = []
        freqs = None
        for channel in data.reshape(-1, data.shape[-1]):
            freqs, channel_psd = signal.welch(channel, fs=sfreq, nperseg=nperseg)
            psds.append(channel_psd)
        assert freqs is not None
        psd_mean = np.mean(np.vstack(psds), axis=0)
    else:
        freqs, psd_mean = signal.welch(data, fs=sfreq, nperseg=nperseg)
    if fmax is not None:
        mask = freqs <= fmax
        freqs = freqs[mask]
        psd_mean = psd_mean[mask]
    return freqs, psd_mean


def gait_signal(imu_data: np.ndarray, imu_names: list[str]) -> np.ndarray:
    groups: dict[str, list[int]] = {}
    for idx, name in enumerate(imu_names):
        lower = name.lower()
        if "acc" not in lower:
            continue
        groups.setdefault(name[0].upper(), []).append(idx)
    mags = []
    for idxs in groups.values():
        if len(idxs) >= 3:
            mags.append(np.linalg.norm(imu_data[idxs, :], axis=0))
    if not mags:
        mags = [np.mean(imu_data, axis=0)]
    stacked = standardize(np.vstack(mags), axis=-1)
    return np.mean(stacked, axis=0)


def estimate_gait_frequency(
    imu_data: np.ndarray,
    imu_names: list[str],
    sfreq: float,
    fmin: float = 0.5,
    fmax: float = 3.5,
) -> float:
    gait = gait_signal(imu_data, imu_names)
    freqs, psd = signal.welch(gait, fs=sfreq, nperseg=min(len(gait), int(12 * sfreq)))
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan
    return float(freqs[mask][np.argmax(psd[mask])])
