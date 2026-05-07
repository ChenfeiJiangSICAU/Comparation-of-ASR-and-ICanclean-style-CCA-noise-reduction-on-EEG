from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sigproc import butter_filter, invsqrtm, regularized_cov, standardize, window_slices


@dataclass
class ASRModel:
    cov_ref: np.ndarray
    sfreq: float
    window_s: float = 1.0
    step_s: float = 0.5

    @classmethod
    def calibrate(
        cls,
        eeg: np.ndarray,
        sfreq: float,
        window_s: float = 1.0,
        step_s: float = 0.5,
        clean_quantile: float = 0.25,
    ) -> "ASRModel":
        fit = butter_filter(eeg, sfreq, l_freq=0.5, h_freq=45.0)
        fit = fit - np.median(fit, axis=1, keepdims=True)
        window = max(16, int(round(window_s * sfreq)))
        step = max(8, int(round(step_s * sfreq)))
        slices = window_slices(fit.shape[1], window, step)
        rms = np.array([np.sqrt(np.mean(fit[:, sl] ** 2)) for sl in slices])
        threshold = np.quantile(rms, clean_quantile)
        selected = [sl for sl, score in zip(slices, rms) if score <= threshold]
        if not selected:
            selected = slices[: max(1, len(slices) // 4)]
        covs = [regularized_cov(fit[:, sl], reg=1e-5) for sl in selected]
        cov_ref = np.mean(covs, axis=0)
        return cls(cov_ref=cov_ref, sfreq=sfreq, window_s=window_s, step_s=step_s)

    def transform(self, eeg: np.ndarray, cutoff: float = 10.0) -> np.ndarray:
        invsqrt_ref, sqrt_ref = invsqrtm(self.cov_ref)
        window = max(16, int(round(self.window_s * self.sfreq)))
        step = max(8, int(round(self.step_s * self.sfreq)))
        slices = window_slices(eeg.shape[1], window, step)
        out_dtype = np.result_type(eeg.dtype, np.float32)
        cleaned = np.zeros_like(eeg, dtype=out_dtype)
        weights = np.zeros(eeg.shape[1], dtype=out_dtype)
        taper = np.hanning(window)
        if not np.any(taper):
            taper = np.ones(window)

        fit_all = butter_filter(eeg, self.sfreq, l_freq=0.5, h_freq=45.0)
        for sl in slices:
            x = eeg[:, sl]
            fit = fit_all[:, sl]
            mean = np.mean(x, axis=1, keepdims=True)
            fit_centered = fit - np.mean(fit, axis=1, keepdims=True)
            cov = regularized_cov(fit_centered, reg=1e-5)
            whitened_cov = invsqrt_ref @ cov @ invsqrt_ref.T
            vals, vecs = np.linalg.eigh(whitened_cov)
            bad = vals > cutoff
            if np.any(bad):
                centered = x - mean
                z = invsqrt_ref @ centered
                good_vecs = vecs[:, ~bad]
                if good_vecs.size:
                    z_clean = good_vecs @ (good_vecs.T @ z)
                else:
                    z_clean = np.zeros_like(z)
                x_clean = sqrt_ref @ z_clean + mean
            else:
                x_clean = x
            local_taper = taper if len(taper) == x.shape[1] else np.hanning(x.shape[1])
            cleaned[:, sl] += x_clean * local_taper[None, :]
            weights[sl] += local_taper
        weights = np.maximum(weights, 1e-12)
        cleaned /= weights[None, :]
        return cleaned


@dataclass
class ICANDecomposition:
    filters: np.ndarray
    corrs: np.ndarray

    def transform(self, eeg: np.ndarray, corr_threshold: float = 0.22, max_components: int = 6) -> np.ndarray:
        selected = np.flatnonzero(self.corrs >= corr_threshold)[:max_components]
        if selected.size == 0 or self.filters.size == 0:
            return eeg.copy()
        dtype = np.result_type(eeg.dtype, np.float32)
        a = self.filters[:, selected].astype(dtype, copy=False)
        x_centered = np.array(eeg, dtype=dtype, copy=True)
        mean = np.mean(x_centered, axis=1, keepdims=True)
        x_centered -= mean
        components = a.T @ x_centered
        beta = x_centered @ components.T @ np.linalg.pinv(components @ components.T)
        chunk = max(1024, int(2_000_000 / max(beta.shape[0], 1)))
        for start in range(0, x_centered.shape[1], chunk):
            stop = min(start + chunk, x_centered.shape[1])
            x_centered[:, start:stop] -= beta @ components[:, start:stop]
        x_centered += mean
        return x_centered


def _channel_kind(name: str) -> str | None:
    lower = name.lower()
    if "acc" in lower:
        return "acc"
    if "gyro" in lower:
        return "gyro"
    if "mag" in lower:
        return "mag"
    return None


def _select_imu_channels(imu: np.ndarray, imu_names: list[str], kinds: set[str]) -> np.ndarray:
    idx = [i for i, name in enumerate(imu_names) if _channel_kind(name) in kinds]
    if not idx:
        return np.empty((0, imu.shape[1]))
    return imu[idx, :]


def _imu_magnitudes(imu: np.ndarray, imu_names: list[str], kinds: set[str]) -> np.ndarray:
    groups: dict[tuple[str, str], list[int]] = {}
    for idx, name in enumerate(imu_names):
        kind = _channel_kind(name)
        if kind not in kinds:
            continue
        sensor = name[0].upper() if name else "?"
        groups.setdefault((sensor, kind), []).append(idx)
    mags = []
    for idxs in groups.values():
        if len(idxs) >= 3:
            mags.append(np.linalg.norm(imu[idxs, :], axis=0))
    if not mags:
        return np.empty((0, imu.shape[1]))
    return np.vstack(mags)


def _lag_reference(ref: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return ref
    out = np.empty_like(ref)
    if lag > 0:
        out[:, :lag] = ref[:, :1]
        out[:, lag:] = ref[:, :-lag]
    else:
        step = -lag
        out[:, -step:] = ref[:, -1:]
        out[:, :-step] = ref[:, step:]
    return out


def _stack_references(parts: list[np.ndarray], n_samples: int) -> np.ndarray:
    kept = [part for part in parts if part.size]
    if not kept:
        return np.empty((0, n_samples))
    return np.vstack(kept)


def build_imu_reference(
    imu: np.ndarray,
    imu_names: list[str],
    sfreq: float,
    mode: str = "acc_gyro",
) -> np.ndarray:
    if imu.size == 0:
        return imu
    mode = mode.lower()
    acc = _select_imu_channels(imu, imu_names, {"acc"})
    gyro = _select_imu_channels(imu, imu_names, {"gyro"})

    if mode == "acc":
        ref = acc
    elif mode == "gyro":
        ref = gyro
    elif mode in {"acc_gyro", "motion"}:
        ref = _stack_references([acc, gyro], imu.shape[1])
    elif mode == "acc_gyro_mag":
        mags = _imu_magnitudes(imu, imu_names, {"acc", "gyro"})
        ref = _stack_references([acc, gyro, mags], imu.shape[1])
    elif mode in {"enhanced", "enhanced_lagged"}:
        mags = _imu_magnitudes(imu, imu_names, {"acc", "gyro"})
        jerk = np.gradient(acc, axis=1) * sfreq if acc.size else np.empty((0, imu.shape[1]))
        jerk_mags = _imu_magnitudes(jerk, [name for name in imu_names if _channel_kind(name) == "acc"], {"acc"})
        ref = _stack_references([acc, gyro, mags, jerk, jerk_mags], imu.shape[1])
        if mode == "enhanced_lagged" and ref.size:
            lags = [int(round(seconds * sfreq)) for seconds in (-0.25, -0.125, 0.0, 0.125, 0.25)]
            ref = np.vstack([_lag_reference(ref, lag) for lag in lags])
    else:
        raise ValueError(f"Unknown IMU reference mode: {mode}")

    return ref if ref.size else imu


def ican_clean(
    eeg: np.ndarray,
    imu: np.ndarray,
    sfreq: float,
    corr_threshold: float = 0.22,
    max_components: int = 6,
    fit_l_freq: float = 0.5,
    fit_h_freq: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    model = fit_ican_decomposition(
        eeg,
        imu,
        sfreq,
        fit_l_freq=fit_l_freq,
        fit_h_freq=fit_h_freq,
    )
    return model.transform(eeg, corr_threshold=corr_threshold, max_components=max_components), model.corrs


def fit_ican_decomposition(
    eeg: np.ndarray,
    imu: np.ndarray,
    sfreq: float,
    fit_l_freq: float = 0.5,
    fit_h_freq: float = 15.0,
) -> ICANDecomposition:
    if imu.size == 0 or eeg.size == 0:
        return ICANDecomposition(filters=np.empty((eeg.shape[0], 0)), corrs=np.array([]))
    x_fit = butter_filter(eeg, sfreq, l_freq=fit_l_freq, h_freq=fit_h_freq)
    r_fit = butter_filter(imu, sfreq, l_freq=fit_l_freq, h_freq=fit_h_freq)
    xz = standardize(x_fit, axis=-1, copy=False)
    rz = standardize(r_fit, axis=-1, copy=False)
    n = xz.shape[1]
    cxx = regularized_cov(xz, reg=1e-4)
    crr = regularized_cov(rz, reg=1e-4)
    cxr = (xz @ rz.T) / max(n - 1, 1)
    invsqrt_x, _ = invsqrtm(cxx)
    invsqrt_r, _ = invsqrtm(crr)
    u, s, _ = np.linalg.svd(invsqrt_x @ cxr @ invsqrt_r, full_matrices=False)
    corrs = np.clip(s, 0.0, 1.0)
    filters = invsqrt_x @ u
    return ICANDecomposition(filters=filters, corrs=corrs)
