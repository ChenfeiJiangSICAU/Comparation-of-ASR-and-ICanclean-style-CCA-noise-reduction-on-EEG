from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import SPEED_ORDER


COLORS = {"raw": "#4c4c4c", "asr": "#2c7fb8", "ican": "#d95f02"}
LABELS = {"raw": "Raw", "asr": "ASR", "ican": "iCanClean"}
PRESERVATION_YLIM = (-0.2, 2.2)


def _ensure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _mark_preservation_outliers(ax, df: pd.DataFrame) -> None:
    if df.empty:
        return
    lower, upper = PRESERVATION_YLIM
    values = df["signal_preservation"].dropna()
    n_out = int(((values < lower) | (values > upper)).sum())
    ax.set_ylim(lower, upper)
    if n_out:
        ax.text(
            0.98,
            0.04,
            f"{n_out} outlier{'s' if n_out > 1 else ''} outside axis",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#666666",
        )


def plot_coherence(df: pd.DataFrame, path: Path) -> None:
    _ensure(path)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for speed in SPEED_ORDER:
        sub = df[df["speed"] == speed]
        if sub.empty:
            continue
        grouped = sub.groupby("frequency_hz")["coherence"].mean()
        ax.plot(grouped.index, grouped.values, label=speed, linewidth=2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Raw EEG-IMU coherence")
    ax.set_title("Raw EEG and IMU coherence by speed")
    ax.set_xlim(0, 15)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_gait_reduction(df: pd.DataFrame, path: Path) -> None:
    _ensure(path)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, speed in zip(axes.ravel(), SPEED_ORDER):
        for method in ("asr", "ican"):
            sub = df[(df["speed"] == speed) & (df["method"] == method)]
            if sub.empty:
                continue
            grouped = sub.groupby("frequency_hz")["reduction_db"].mean()
            ax.plot(grouped.index, grouped.values, label=LABELS[method], color=COLORS[method], linewidth=2)
        gait = df[df["speed"] == speed]["gait_frequency_hz"].dropna()
        if not gait.empty:
            ax.axvline(gait.mean(), color="#666666", linestyle="--", linewidth=1)
        ax.set_title(speed)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("Power reduction (dB)")
    axes[1, 0].set_ylabel("Power reduction (dB)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.suptitle("Gait-frequency EEG power reduction", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_erp_preservation(df: pd.DataFrame, path: Path) -> None:
    _ensure(path)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, speed in zip(axes.ravel(), SPEED_ORDER):
        for method in ("raw", "asr", "ican"):
            sub = df[(df["speed"] == speed) & (df["method"] == method) & (df["label"] == 2)]
            if sub.empty:
                continue
            grouped = sub.groupby("time_s")["amplitude"].mean()
            ax.plot(grouped.index, grouped.values, color=COLORS[method], label=LABELS[method], linewidth=2)
        ax.axvspan(0.25, 0.55, color="#f0f0f0", zorder=0)
        ax.axvline(0, color="#777777", linewidth=1)
        ax.set_title(speed)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Time from stimulus (s)")
    axes[1, 1].set_xlabel("Time from stimulus (s)")
    axes[0, 0].set_ylabel("Pz target ERP (microV)")
    axes[1, 0].set_ylabel("Pz target ERP (microV)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    fig.suptitle("ERP preservation after denoising", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_tradeoff(df: pd.DataFrame, path: Path) -> None:
    _ensure(path)
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = []
    for method in ("asr", "ican"):
        sub = df[df["method"] == method].dropna(subset=["artifact_reduction_db", "signal_preservation"])
        if sub.empty:
            continue
        plotted.append(sub)
        ax.scatter(
            sub["artifact_reduction_db"],
            sub["signal_preservation"],
            s=60,
            alpha=0.75,
            color=COLORS[method],
            label=LABELS[method],
        )
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Artifact reduction at gait frequency (dB)")
    ax.set_ylabel("ERP P300 preservation ratio")
    ax.set_title("Artifact reduction vs signal preservation (pooled speeds)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    if plotted:
        _mark_preservation_outliers(ax, pd.concat(plotted, ignore_index=True))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_tradeoff_by_speed(df: pd.DataFrame, path: Path) -> None:
    _ensure(path)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, speed in zip(axes.ravel(), SPEED_ORDER):
        plotted = []
        for method in ("asr", "ican"):
            sub = df[(df["speed"] == speed) & (df["method"] == method)].dropna(
                subset=["artifact_reduction_db", "signal_preservation"]
            )
            if sub.empty:
                continue
            plotted.append(sub)
            ax.scatter(
                sub["artifact_reduction_db"],
                sub["signal_preservation"],
                s=60,
                alpha=0.75,
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(speed)
        ax.grid(alpha=0.25)
        if plotted:
            _mark_preservation_outliers(ax, pd.concat(plotted, ignore_index=True))
    axes[1, 0].set_xlabel("Artifact reduction at gait frequency (dB)")
    axes[1, 1].set_xlabel("Artifact reduction at gait frequency (dB)")
    axes[0, 0].set_ylabel("ERP P300 preservation ratio")
    axes[1, 0].set_ylabel("ERP P300 preservation ratio")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.suptitle("Artifact reduction vs signal preservation by speed", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_asr_sensitivity(df: pd.DataFrame, path: Path) -> None:
    _ensure(path)
    grouped = df.groupby("cutoff").agg(
        artifact_reduction_db=("artifact_reduction_db", "mean"),
        signal_preservation=("signal_preservation", "mean"),
        auc=("auc", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()
    line1 = ax.plot(
        grouped.index,
        grouped["artifact_reduction_db"],
        marker="o",
        color=COLORS["asr"],
        label="Artifact reduction (dB)",
    )
    line2 = ax2.plot(
        grouped.index,
        grouped["signal_preservation"],
        marker="o",
        color=COLORS["ican"],
        label="ERP preservation ratio",
    )
    lines = line1 + line2
    if not grouped["auc"].isna().all():
        lines += ax2.plot(grouped.index, grouped["auc"], marker="o", color="#2ca02c", label="ERP AUC")
    ax.set_xlabel("ASR cutoff")
    ax.set_ylabel("Artifact reduction (dB)")
    ax2.set_ylabel("Preservation / AUC")
    ax2.set_ylim(0, 1.1)
    ax.set_title("ASR cutoff sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_asr_sensitivity_by_speed(df: pd.DataFrame, path: Path) -> None:
    _ensure(path)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    legend_lines = []
    right_axes = []
    for ax, speed in zip(axes.ravel(), SPEED_ORDER):
        sub = df[df["speed"] == speed]
        ax2 = ax.twinx()
        right_axes.append(ax2)
        if sub.empty:
            ax.set_title(speed)
            ax.grid(alpha=0.25)
            ax2.set_ylim(0, 1.1)
            continue
        grouped = sub.groupby("cutoff").agg(
            artifact_reduction_db=("artifact_reduction_db", "mean"),
            signal_preservation=("signal_preservation", "mean"),
            auc=("auc", "mean"),
        )
        line1 = ax.plot(
            grouped.index,
            grouped["artifact_reduction_db"],
            marker="o",
            color=COLORS["asr"],
            label="Artifact reduction (dB)",
        )
        line2 = ax2.plot(
            grouped.index,
            grouped["signal_preservation"],
            marker="o",
            color=COLORS["ican"],
            label="ERP preservation ratio",
        )
        lines = line1 + line2
        if not grouped["auc"].isna().all():
            lines += ax2.plot(grouped.index, grouped["auc"], marker="o", color="#2ca02c", label="ERP AUC")
        if not legend_lines:
            legend_lines = lines
        ax2.set_ylim(0, 1.1)
        ax.set_title(speed)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("ASR cutoff")
    axes[1, 1].set_xlabel("ASR cutoff")
    axes[0, 0].set_ylabel("Artifact reduction (dB)")
    axes[1, 0].set_ylabel("Artifact reduction (dB)")
    for idx in (0, 2):
        right_axes[idx].tick_params(right=False, labelright=False)
    right_axes[1].set_ylabel("Preservation / AUC")
    right_axes[3].set_ylabel("Preservation / AUC")
    if legend_lines:
        fig.legend(
            legend_lines,
            [line.get_label() for line in legend_lines],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=3,
            frameon=False,
        )
    fig.suptitle("ASR cutoff sensitivity by speed", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_bci(df: pd.DataFrame, path: Path) -> None:
    _ensure(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=False)
    erp = df[df["task"] == "ERP"]
    ssvep = df[df["task"] == "SSVEP"]
    x = np.arange(len(SPEED_ORDER))
    width = 0.25
    for ax, sub, metric, title in [
        (axes[0], erp, "auc", "ERP LDA AUC"),
        (axes[1], ssvep, "accuracy", "SSVEP CCA accuracy"),
    ]:
        for offset, method in zip((-width, 0, width), ("raw", "asr", "ican")):
            vals = []
            for speed in SPEED_ORDER:
                part = sub[(sub["speed"] == speed) & (sub["method"] == method)][metric].dropna()
                vals.append(part.mean() if not part.empty else np.nan)
            ax.bar(x + offset, vals, width=width, color=COLORS[method], label=LABELS[method])
        ax.set_xticks(x)
        ax.set_xticklabels(SPEED_ORDER, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    fig.suptitle("BCI classification performance by speed", y=0.98)
    fig.tight_layout(rect=(0, 0.10, 1, 0.90))
    fig.savefig(path, dpi=180)
    plt.close(fig)
