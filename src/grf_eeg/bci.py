from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SSVEP_FREQS
from .metrics import baseline_correct, epoch_data


def erp_features(
    eeg: np.ndarray,
    events,
    sfreq: float,
    channel_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    epochs, labels, times = epoch_data(eeg[channel_indices, :], events, sfreq, -0.2, 0.8, {1, 2})
    if epochs.size == 0:
        return np.empty((0, 0)), labels
    epochs = baseline_correct(epochs, times, (-0.2, 0.0))
    windows = [(0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 0.45)]
    chunks = []
    for start, stop in windows:
        mask = (times >= start) & (times < stop)
        chunks.append(np.mean(epochs[:, :, mask], axis=2))
    features = np.concatenate(chunks, axis=1)
    return features, labels


def train_erp_classifier(features: np.ndarray, labels: np.ndarray):
    model = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
    model.fit(features, labels)
    return model


def score_erp_classifier(model, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if features.size == 0 or len(np.unique(labels)) < 2:
        return {"auc": np.nan, "balanced_accuracy": np.nan}
    scores = model.decision_function(features)
    predictions = model.predict(features)
    binary = (labels == 2).astype(int)
    return {
        "auc": float(roc_auc_score(binary, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def _canonical_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x - np.mean(x, axis=0, keepdims=True)
    y = y - np.mean(y, axis=0, keepdims=True)
    x_std = np.std(x, axis=0, keepdims=True)
    y_std = np.std(y, axis=0, keepdims=True)
    x = x / np.maximum(x_std, 1e-12)
    y = y / np.maximum(y_std, 1e-12)
    cxx = x.T @ x / max(x.shape[0] - 1, 1) + np.eye(x.shape[1]) * 1e-6
    cyy = y.T @ y / max(y.shape[0] - 1, 1) + np.eye(y.shape[1]) * 1e-6
    cxy = x.T @ y / max(x.shape[0] - 1, 1)
    vals_x, vecs_x = linalg.eigh(cxx)
    vals_y, vecs_y = linalg.eigh(cyy)
    wx = (vecs_x / np.sqrt(np.maximum(vals_x, 1e-9))) @ vecs_x.T
    wy = (vecs_y / np.sqrt(np.maximum(vals_y, 1e-9))) @ vecs_y.T
    return float(np.linalg.svd(wx @ cxy @ wy, compute_uv=False)[0])


def ssvep_accuracy(
    eeg: np.ndarray,
    events,
    sfreq: float,
    channel_indices: list[int],
) -> dict[str, float]:
    epochs, labels, _ = epoch_data(eeg[channel_indices, :], events, sfreq, 0.0, 5.0, set(SSVEP_FREQS))
    if epochs.size == 0:
        return {"accuracy": np.nan}
    predictions = []
    n_times = epochs.shape[2]
    t = np.arange(n_times) / sfreq
    references = {}
    for label, freq in SSVEP_FREQS.items():
        references[label] = np.column_stack(
            [
                np.sin(2 * np.pi * freq * t),
                np.cos(2 * np.pi * freq * t),
                np.sin(2 * np.pi * 2 * freq * t),
                np.cos(2 * np.pi * 2 * freq * t),
            ]
        )
    for epoch in epochs:
        x = epoch.T
        corrs = {label: _canonical_corr(x, ref) for label, ref in references.items()}
        predictions.append(max(corrs, key=corrs.get))
    predictions = np.array(predictions, dtype=int)
    return {"accuracy": float(np.mean(predictions == labels))}
