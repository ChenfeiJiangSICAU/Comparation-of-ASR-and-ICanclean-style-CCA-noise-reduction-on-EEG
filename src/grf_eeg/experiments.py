from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import SPEED_ORDER, PipelineConfig
from .dataio import channel_indices, discover_recordings, read_brainvision_data, read_events
from .denoise import ASRModel, build_imu_reference, fit_ican_decomposition
from .metrics import artifact_reduction_at_gait, erp_preservation_metrics, erp_preservation_score


ASR_CUTOFFS = [3, 5, 8, 10, 15, 20, 30]
ICAN_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]
ICAN_MAX_COMPONENTS = [2, 4, 6, 8]
IMU_MODES = ["acc", "gyro", "acc_gyro", "acc_gyro_mag", "enhanced", "enhanced_lagged"]
COLORS = {
    "ASR": "#2c7fb8",
    "iCanClean": "#d95f02",
    "iCanClean reference": "#9467bd",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supplementary denoising experiments.")
    parser.add_argument("--data-dir", default="Motion_eeg_data")
    parser.add_argument("--out-dir", default="outputs_supplement")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true", help="Run sub-01 only for a smoke test.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    subjects = args.subjects
    if args.quick and not subjects:
        subjects = ["sub-01"]
    config = PipelineConfig(data_dir=args.data_dir, out_dir=args.out_dir)
    run_supplementary_experiments(config, subjects=subjects)


def run_supplementary_experiments(config: PipelineConfig, subjects: list[str] | None = None) -> None:
    out_dir = Path(config.out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    recordings = discover_recordings(config.data_dir, subjects=subjects, tasks=["ERP"])
    train_recordings = [rec for rec in recordings if rec.session == "ses-01"]
    test_recordings = [rec for rec in recordings if rec.speed_mps is not None]
    if not train_recordings or not test_recordings:
        raise SystemExit("Supplementary experiments require ERP training and movement recordings.")
    print(f"Found {len(train_recordings)} ERP training recordings and {len(test_recordings)} ERP test recordings")

    asr_models = calibrate_asr(train_recordings, config)
    default_rows = run_default_robust(test_recordings, asr_models, config)
    asr_rows = run_asr_parameter_sweep(test_recordings, asr_models, config)
    ican_reference_rows, ican_grid_rows = run_ican_experiments(test_recordings, config)

    all_rows = default_rows + asr_rows + ican_reference_rows + ican_grid_rows
    summary = summarize_configs(all_rows)
    stats_rows = paired_default_stats(default_rows)
    selected = preservation_constrained_selection(summary)

    write_csv(tables_dir / "default_robust_metrics.csv", default_rows)
    write_csv(tables_dir / "asr_parameter_sweep.csv", asr_rows)
    write_csv(tables_dir / "ican_reference_ablation.csv", ican_reference_rows)
    write_csv(tables_dir / "ican_parameter_sweep.csv", ican_grid_rows)
    write_csv(tables_dir / "supplementary_all_results.csv", all_rows)
    summary.to_csv(tables_dir / "supplementary_config_summary.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(tables_dir / "paired_default_stats.csv", index=False)
    selected.to_csv(tables_dir / "preservation_constrained_selection.csv", index=False)

    plot_robust_tradeoff(pd.DataFrame(default_rows), figures_dir / "robust_tradeoff_by_speed.png")
    plot_reference_ablation(pd.DataFrame(ican_reference_rows), figures_dir / "imu_reference_ablation.png")
    plot_parameter_pareto(summary, figures_dir / "parameter_pareto.png")
    plot_parameter_pareto_by_speed(summary, figures_dir / "parameter_pareto_by_speed.png")
    plot_selection(selected, figures_dir / "preservation_constrained_selection.png")

    manifest = {
        "config": config.__dict__,
        "tables": [str(path) for path in sorted(tables_dir.glob("*.csv"))],
        "figures": [str(path) for path in sorted(figures_dir.glob("*.png"))],
        "notes": {
            "primary_preservation_metric": "diff_peak_ratio",
            "preservation_constraint": "diff_peak_ratio >= 0.8",
            "asr_cutoffs": ASR_CUTOFFS,
            "ican_thresholds": ICAN_THRESHOLDS,
            "ican_max_components": ICAN_MAX_COMPONENTS,
            "imu_reference_modes": IMU_MODES,
        },
    }
    (tables_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Supplementary results written to {out_dir.resolve()}")


def write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def calibrate_asr(train_recordings, config: PipelineConfig) -> dict[str, ASRModel]:
    models = {}
    for rec in train_recordings:
        print(f"Calibrating ASR supplementary model on {rec.subject} {rec.session}")
        data = read_brainvision_data(rec)
        eeg_idx = channel_indices(rec, "eeg", good_only=False)
        models[rec.subject] = ASRModel.calibrate(
            data[eeg_idx, :],
            rec.header.sfreq,
            window_s=config.asr_window_s,
            step_s=config.asr_step_s,
        )
    return models


def recording_context(rec):
    data = read_brainvision_data(rec)
    events = read_events(rec, n_samples=data.shape[1])
    eeg_idx = channel_indices(rec, "eeg", good_only=False)
    scalp_idx = channel_indices(rec, "scalp", good_only=True) or channel_indices(rec, "scalp", good_only=False)
    imu_idx = channel_indices(rec, "imu_motion", good_only=True)
    imu_names = rec.channels.iloc[imu_idx]["name"].astype(str).tolist()
    names = rec.channels["name"].astype(str).tolist()
    pz = names.index("Pz") if "Pz" in names else scalp_idx[0]
    return data, events, eeg_idx, scalp_idx, imu_idx, imu_names, pz


def evaluate_cleaning(
    rec,
    raw: np.ndarray,
    clean: np.ndarray,
    events,
    scalp_idx: list[int],
    imu_idx: list[int],
    imu_names: list[str],
    pz: int,
    metadata: dict,
) -> dict:
    artifact, gait_freq = artifact_reduction_at_gait(
        raw[scalp_idx, :],
        clean[scalp_idx, :],
        raw[imu_idx, :],
        imu_names,
        rec.header.sfreq,
    )
    metrics = erp_preservation_metrics(raw, clean, events, rec.header.sfreq, pz)
    row = {
        "subject": rec.subject,
        "session": rec.session,
        "speed": rec.speed_label,
        "speed_mps": rec.speed_mps,
        "artifact_reduction_db": artifact,
        "gait_frequency_hz": gait_freq,
        "legacy_target_peak_ratio": erp_preservation_score(raw, clean, events, rec.header.sfreq, pz),
    }
    row.update(metrics)
    row.update(metadata)
    return row


def run_default_robust(test_recordings, asr_models: dict[str, ASRModel], config: PipelineConfig) -> list[dict]:
    rows = []
    for rec in test_recordings:
        print(f"Default robust metrics {rec.subject} {rec.session}")
        data, events, eeg_idx, scalp_idx, imu_idx, imu_names, pz = recording_context(rec)
        if rec.subject in asr_models:
            clean = data.copy()
            clean[eeg_idx, :] = asr_models[rec.subject].transform(data[eeg_idx, :], cutoff=config.asr_cutoff)
            rows.append(
                evaluate_cleaning(
                    rec,
                    data,
                    clean,
                    events,
                    scalp_idx,
                    imu_idx,
                    imu_names,
                    pz,
                    {
                        "experiment": "default",
                        "family": "ASR",
                        "method": "ASR",
                        "config_label": f"ASR cutoff={config.asr_cutoff:g}",
                        "asr_cutoff": config.asr_cutoff,
                        "ican_threshold": np.nan,
                        "ican_max_components": np.nan,
                        "imu_mode": "",
                    },
                )
            )
        if imu_idx:
            ref = build_imu_reference(data[imu_idx, :], imu_names, rec.header.sfreq, mode="acc_gyro")
            decomposition = fit_ican_decomposition(data[eeg_idx, :], ref, rec.header.sfreq)
            clean = data.copy()
            clean[eeg_idx, :] = decomposition.transform(
                data[eeg_idx, :],
                corr_threshold=config.ican_corr_threshold,
                max_components=config.ican_max_components,
            )
            rows.append(
                evaluate_cleaning(
                    rec,
                    data,
                    clean,
                    events,
                    scalp_idx,
                    imu_idx,
                    imu_names,
                    pz,
                    {
                        "experiment": "default",
                        "family": "iCanClean",
                        "method": "iCanClean",
                        "config_label": "iCanClean acc+gyro threshold=0.22 max=6",
                        "asr_cutoff": np.nan,
                        "ican_threshold": config.ican_corr_threshold,
                        "ican_max_components": config.ican_max_components,
                        "imu_mode": "acc_gyro",
                    },
                )
            )
        del data, events
        gc.collect()
    return rows


def run_asr_parameter_sweep(test_recordings, asr_models: dict[str, ASRModel], config: PipelineConfig) -> list[dict]:
    rows = []
    for rec in test_recordings:
        if rec.subject not in asr_models:
            continue
        print(f"ASR parameter sweep {rec.subject} {rec.session}")
        data, events, eeg_idx, scalp_idx, imu_idx, imu_names, pz = recording_context(rec)
        for cutoff in ASR_CUTOFFS:
            clean = data.copy()
            clean[eeg_idx, :] = asr_models[rec.subject].transform(data[eeg_idx, :], cutoff=cutoff)
            rows.append(
                evaluate_cleaning(
                    rec,
                    data,
                    clean,
                    events,
                    scalp_idx,
                    imu_idx,
                    imu_names,
                    pz,
                    {
                        "experiment": "asr_parameter_sweep",
                        "family": "ASR",
                        "method": "ASR",
                        "config_label": f"ASR cutoff={cutoff:g}",
                        "asr_cutoff": cutoff,
                        "ican_threshold": np.nan,
                        "ican_max_components": np.nan,
                        "imu_mode": "",
                    },
                )
            )
            del clean
        del data, events
        gc.collect()
    return rows


def run_ican_experiments(test_recordings, config: PipelineConfig) -> tuple[list[dict], list[dict]]:
    reference_rows = []
    grid_rows = []
    for rec in test_recordings:
        print(f"iCanClean supplementary experiments {rec.subject} {rec.session}")
        data, events, eeg_idx, scalp_idx, imu_idx, imu_names, pz = recording_context(rec)
        if not imu_idx:
            continue
        decompositions = {}
        for mode in IMU_MODES:
            ref = build_imu_reference(data[imu_idx, :], imu_names, rec.header.sfreq, mode=mode)
            decomposition = fit_ican_decomposition(data[eeg_idx, :], ref, rec.header.sfreq)
            decompositions[mode] = decomposition
            clean = data.copy()
            clean[eeg_idx, :] = decomposition.transform(
                data[eeg_idx, :],
                corr_threshold=config.ican_corr_threshold,
                max_components=config.ican_max_components,
            )
            reference_rows.append(
                evaluate_cleaning(
                    rec,
                    data,
                    clean,
                    events,
                    scalp_idx,
                    imu_idx,
                    imu_names,
                    pz,
                    {
                        "experiment": "ican_reference_ablation",
                        "family": "iCanClean reference",
                        "method": f"iCanClean {mode}",
                        "config_label": f"{mode}, threshold=0.22, max=6",
                        "asr_cutoff": np.nan,
                        "ican_threshold": config.ican_corr_threshold,
                        "ican_max_components": config.ican_max_components,
                        "imu_mode": mode,
                    },
                )
            )
            del clean

        decomposition = decompositions["enhanced_lagged"]
        for threshold in ICAN_THRESHOLDS:
            for max_components in ICAN_MAX_COMPONENTS:
                clean = data.copy()
                clean[eeg_idx, :] = decomposition.transform(
                    data[eeg_idx, :],
                    corr_threshold=threshold,
                    max_components=max_components,
                )
                grid_rows.append(
                    evaluate_cleaning(
                        rec,
                        data,
                        clean,
                        events,
                        scalp_idx,
                        imu_idx,
                        imu_names,
                        pz,
                        {
                            "experiment": "ican_parameter_sweep",
                            "family": "iCanClean",
                            "method": "iCanClean enhanced_lagged",
                            "config_label": f"enhanced_lagged threshold={threshold:.2f} max={max_components}",
                            "asr_cutoff": np.nan,
                            "ican_threshold": threshold,
                            "ican_max_components": max_components,
                            "imu_mode": "enhanced_lagged",
                        },
                    )
                )
                del clean
        del data, events, decompositions
        gc.collect()
    return reference_rows, grid_rows


def summarize_configs(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    grouped = df.groupby(["experiment", "family", "method", "config_label", "speed"], dropna=False)
    summary = grouped.agg(
        n=("subject", "nunique"),
        artifact_mean_db=("artifact_reduction_db", "mean"),
        artifact_median_db=("artifact_reduction_db", "median"),
        diff_peak_ratio_mean=("diff_peak_ratio", "mean"),
        diff_peak_ratio_median=("diff_peak_ratio", "median"),
        diff_area_ratio_median=("diff_area_ratio", "median"),
        diff_ptp_ratio_median=("diff_ptp_ratio", "median"),
        diff_wave_corr_median=("diff_wave_corr", "median"),
        legacy_target_peak_ratio_median=("legacy_target_peak_ratio", "median"),
        asr_cutoff=("asr_cutoff", "first"),
        ican_threshold=("ican_threshold", "first"),
        ican_max_components=("ican_max_components", "first"),
        imu_mode=("imu_mode", "first"),
    )
    return summary.reset_index()


def paired_default_stats(rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(rows)
    metrics = ["artifact_reduction_db", "diff_peak_ratio", "diff_area_ratio", "diff_ptp_ratio", "diff_wave_corr"]
    out = []
    for speed in SPEED_ORDER:
        sub = df[(df["experiment"] == "default") & (df["speed"] == speed)]
        for metric in metrics:
            pivot = sub.pivot_table(index="subject", columns="method", values=metric, aggfunc="mean").dropna()
            if {"ASR", "iCanClean"} - set(pivot.columns) or len(pivot) < 5:
                continue
            diff = pivot["ASR"] - pivot["iCanClean"]
            try:
                stat, pvalue = stats.wilcoxon(pivot["ASR"], pivot["iCanClean"], zero_method="wilcox")
            except ValueError:
                stat, pvalue = np.nan, np.nan
            out.append(
                {
                    "speed": speed,
                    "metric": metric,
                    "comparison": "ASR - iCanClean",
                    "n": len(pivot),
                    "mean_difference": float(diff.mean()),
                    "median_difference": float(diff.median()),
                    "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
                    "p_value": float(pvalue) if np.isfinite(pvalue) else np.nan,
                }
            )
    return out


def preservation_constrained_selection(summary: pd.DataFrame, min_preservation: float = 0.8) -> pd.DataFrame:
    candidates = summary[summary["diff_peak_ratio_median"] >= min_preservation].copy()
    if candidates.empty:
        return candidates
    idx = candidates.groupby(["experiment", "family", "speed"])["artifact_mean_db"].idxmax()
    selected = candidates.loc[idx].sort_values(["speed", "family", "experiment"])
    selected["constraint"] = f"median diff_peak_ratio >= {min_preservation:g}"
    return selected


def _metric_frame(df: pd.DataFrame, y_metric: str = "diff_peak_ratio") -> pd.DataFrame:
    return df.dropna(subset=["artifact_reduction_db", y_metric]).copy()


def plot_robust_tradeoff(df: pd.DataFrame, path: Path, y_metric: str = "diff_peak_ratio") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, speed in zip(axes.ravel(), SPEED_ORDER):
        for family in ("ASR", "iCanClean"):
            sub = _metric_frame(df[(df["speed"] == speed) & (df["family"] == family)], y_metric)
            if sub.empty:
                continue
            ax.scatter(
                sub["artifact_reduction_db"],
                sub[y_metric],
                s=48,
                alpha=0.75,
                color=COLORS[family],
                label=family,
            )
        ax.axhline(0.8, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(speed)
        ax.grid(alpha=0.25)
        ax.set_ylim(-0.1, 2.2)
    axes[1, 0].set_xlabel("Artifact reduction at gait frequency (dB)")
    axes[1, 1].set_xlabel("Artifact reduction at gait frequency (dB)")
    axes[0, 0].set_ylabel("P300 diff-wave peak preservation")
    axes[1, 0].set_ylabel("P300 diff-wave peak preservation")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.suptitle("Robust artifact-signal tradeoff by speed", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_reference_ablation(df: pd.DataFrame, path: Path) -> None:
    summary = (
        df.groupby(["imu_mode", "speed"])
        .agg(
            artifact_mean_db=("artifact_reduction_db", "mean"),
            preservation_median=("diff_peak_ratio", "median"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    x = np.arange(len(IMU_MODES))
    width = 0.18
    speed_colors = dict(zip(SPEED_ORDER, plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(SPEED_ORDER)]))
    for offset, speed in zip(np.linspace(-1.5 * width, 1.5 * width, len(SPEED_ORDER)), SPEED_ORDER):
        sub = summary[summary["speed"] == speed].set_index("imu_mode").reindex(IMU_MODES)
        axes[0].bar(x + offset, sub["artifact_mean_db"], width=width, label=speed, color=speed_colors[speed])
        axes[1].bar(x + offset, sub["preservation_median"], width=width, color=speed_colors[speed])
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(IMU_MODES, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean artifact reduction (dB)")
    axes[1].set_ylabel("Median P300 diff peak preservation")
    axes[1].axhline(0.8, color="#777777", linestyle="--", linewidth=1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False)
    fig.suptitle("iCanClean IMU reference ablation", y=0.98)
    fig.tight_layout(rect=(0, 0.14, 1, 0.90))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_parameter_pareto(summary: pd.DataFrame, path: Path) -> None:
    aggregate = (
        summary.groupby(["experiment", "family", "method", "config_label"])
        .agg(
            artifact_mean_db=("artifact_mean_db", "mean"),
            preservation_median=("diff_peak_ratio_median", "median"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for family, sub in aggregate.groupby("family"):
        ax.scatter(
            sub["artifact_mean_db"],
            sub["preservation_median"],
            s=55,
            alpha=0.8,
            label=family,
            color=COLORS.get(family, "#666666"),
        )
    ax.axhline(0.8, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean artifact reduction at gait frequency (dB)")
    ax.set_ylabel("Median P300 diff peak preservation")
    ax.set_title("Parameter-level Pareto map")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_parameter_pareto_by_speed(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, speed in zip(axes.ravel(), SPEED_ORDER):
        sub_speed = summary[summary["speed"] == speed]
        for family, sub in sub_speed.groupby("family"):
            ax.scatter(
                sub["artifact_mean_db"],
                sub["diff_peak_ratio_median"],
                s=45,
                alpha=0.78,
                label=family,
                color=COLORS.get(family, "#666666"),
            )
        ax.axhline(0.8, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(speed)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Mean artifact reduction (dB)")
    axes[1, 1].set_xlabel("Mean artifact reduction (dB)")
    axes[0, 0].set_ylabel("Median P300 diff peak preservation")
    axes[1, 0].set_ylabel("Median P300 diff peak preservation")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    fig.suptitle("Parameter-level Pareto map by speed", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_selection(selected: pd.DataFrame, path: Path) -> None:
    if selected.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = selected["speed"] + "\n" + selected["family"]
    x = np.arange(len(selected))
    ax.bar(x, selected["artifact_mean_db"], color=[COLORS.get(f, "#666666") for f in selected["family"]])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean artifact reduction (dB)")
    ax.set_title("Best configs under preservation constraint")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
