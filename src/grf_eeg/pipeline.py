from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .bci import erp_features, score_erp_classifier, ssvep_accuracy, train_erp_classifier
from .config import ERP_CHANNELS, SSVEP_CHANNELS, PipelineConfig
from .dataio import (
    Recording,
    channel_indices,
    discover_recordings,
    pick_named,
    read_brainvision_data,
    read_events,
)
from .denoise import ASRModel, ican_clean
from .metrics import (
    artifact_reduction_at_gait,
    coherence_curve,
    erp_preservation_score,
    erp_waveforms,
    gait_power_reduction_curve,
)
from .plots import (
    plot_asr_sensitivity,
    plot_asr_sensitivity_by_speed,
    plot_bci,
    plot_coherence,
    plot_erp_preservation,
    plot_gait_reduction,
    plot_tradeoff,
    plot_tradeoff_by_speed,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ASR and iCanClean on MobileBCI EEG/IMU data.")
    parser.add_argument("--data-dir", default="Motion_eeg_data", help="Path to the BIDS-like data folder.")
    parser.add_argument("--out-dir", default="outputs", help="Directory for tables and figures.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Subset such as sub-01 sub-02.")
    parser.add_argument("--tasks", nargs="*", choices=["ERP", "SSVEP"], default=["ERP", "SSVEP"])
    parser.add_argument("--asr-cutoff", type=float, default=10.0)
    parser.add_argument("--asr-sensitivity-cutoffs", nargs="*", type=float, default=[3, 5, 10, 15, 20])
    parser.add_argument("--quick", action="store_true", help="Run only sub-01 for a smoke test.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = PipelineConfig(data_dir=args.data_dir, out_dir=args.out_dir, asr_cutoff=args.asr_cutoff)
    run_pipeline(config, args.subjects, args.tasks, args.asr_sensitivity_cutoffs, args.quick)


def run_pipeline(
    config: PipelineConfig,
    subjects: list[str] | None,
    tasks: list[str],
    sensitivity_cutoffs: list[float],
    quick: bool = False,
) -> None:
    out_dir = Path(config.out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if quick and not subjects:
        subjects = ["sub-01"]
    recordings = discover_recordings(config.data_dir, subjects=subjects, tasks=tasks)
    if not recordings:
        raise SystemExit(f"No recordings found under {config.data_dir!r}")
    print(f"Found {len(recordings)} recordings")

    asr_models = calibrate_asr(recordings, config)

    coherence_rows = []
    gait_rows = []
    erp_wave_rows = []
    tradeoff_rows = []
    bci_rows = []
    erp_train: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    erp_tests: list[tuple[str, str, str, np.ndarray, np.ndarray]] = []

    for rec in recordings:
        print(f"Processing {rec.subject} {rec.session} {rec.task}")
        data = read_brainvision_data(rec)
        events = read_events(rec, n_samples=data.shape[1])
        eeg_idx = channel_indices(rec, "eeg", good_only=False)
        eeg_good = channel_indices(rec, "scalp", good_only=True)
        imu_motion_idx = channel_indices(rec, "imu_motion", good_only=True)
        imu_names = rec.channels.iloc[imu_motion_idx]["name"].astype(str).tolist()
        methods = clean_recording(data, rec, asr_models, config, eeg_idx, imu_motion_idx)

        if rec.speed_mps is not None:
            coherence_rows += collect_coherence(rec, methods["raw"], eeg_good, imu_motion_idx, imu_names, config)
            gait_rows += collect_gait_reduction(rec, methods, eeg_good, imu_motion_idx, imu_names, config)
            tradeoff_rows += collect_tradeoff(rec, methods, events, eeg_good, imu_motion_idx, imu_names)
            if rec.task == "ERP":
                erp_wave_rows += collect_erp_waves(rec, methods, events)

        if rec.task == "ERP":
            clf_channels = pick_named(rec, ERP_CHANNELS, good_only=False) or eeg_good
            for method, clean_data in methods.items():
                feats, labels = erp_features(clean_data, events, rec.header.sfreq, clf_channels)
                key = (rec.subject, method)
                if rec.session == "ses-01":
                    erp_train[key] = (feats, labels)
                elif rec.speed_mps is not None:
                    erp_tests.append((rec.subject, method, rec.speed_label, feats, labels))
        elif rec.task == "SSVEP" and rec.speed_mps is not None:
            clf_channels = pick_named(rec, SSVEP_CHANNELS, good_only=False) or eeg_good
            for method, clean_data in methods.items():
                score = ssvep_accuracy(clean_data, events, rec.header.sfreq, clf_channels)
                bci_rows.append(
                    {
                        "task": "SSVEP",
                        "subject": rec.subject,
                        "speed": rec.speed_label,
                        "method": method,
                        "accuracy": score["accuracy"],
                        "auc": np.nan,
                        "balanced_accuracy": np.nan,
                    }
                )

    bci_rows += score_erp_tests(erp_train, erp_tests)
    asr_sensitivity = run_asr_sensitivity(recordings, config, sensitivity_cutoffs)

    write_outputs(
        tables_dir,
        figures_dir,
        coherence_rows,
        gait_rows,
        erp_wave_rows,
        tradeoff_rows,
        bci_rows,
        asr_sensitivity,
        config,
    )
    print(f"Done. Results written to {out_dir.resolve()}")


def calibrate_asr(recordings: list[Recording], config: PipelineConfig) -> dict[str, ASRModel]:
    models = {}
    for rec in recordings:
        if rec.task != "ERP" or rec.session != "ses-01":
            continue
        print(f"Calibrating ASR on {rec.subject} ses-01 ERP")
        data = read_brainvision_data(rec)
        eeg_idx = channel_indices(rec, "eeg", good_only=False)
        models[rec.subject] = ASRModel.calibrate(
            data[eeg_idx, :],
            rec.header.sfreq,
            window_s=config.asr_window_s,
            step_s=config.asr_step_s,
        )
    return models


def clean_recording(
    data: np.ndarray,
    rec: Recording,
    asr_models: dict[str, ASRModel],
    config: PipelineConfig,
    eeg_idx: list[int],
    imu_idx: list[int],
) -> dict[str, np.ndarray]:
    raw = data.copy()
    methods = {"raw": raw}

    if rec.subject in asr_models:
        asr_data = data.copy()
        asr_data[eeg_idx, :] = asr_models[rec.subject].transform(data[eeg_idx, :], cutoff=config.asr_cutoff)
        methods["asr"] = asr_data

    if imu_idx:
        ican_data = data.copy()
        ican_data[eeg_idx, :], _ = ican_clean(
            data[eeg_idx, :],
            data[imu_idx, :],
            rec.header.sfreq,
            corr_threshold=config.ican_corr_threshold,
            max_components=config.ican_max_components,
        )
        methods["ican"] = ican_data
    return methods


def collect_coherence(
    rec: Recording,
    data: np.ndarray,
    eeg_idx: list[int],
    imu_idx: list[int],
    imu_names: list[str],
    config: PipelineConfig,
) -> list[dict]:
    freqs, coh = coherence_curve(
        data[eeg_idx, :],
        data[imu_idx, :],
        imu_names,
        rec.header.sfreq,
        nperseg_s=config.nperseg_s,
    )
    return [
        {
            "subject": rec.subject,
            "task": rec.task,
            "speed": rec.speed_label,
            "frequency_hz": f,
            "coherence": c,
        }
        for f, c in zip(freqs, coh)
    ]


def collect_gait_reduction(
    rec: Recording,
    methods: dict[str, np.ndarray],
    eeg_idx: list[int],
    imu_idx: list[int],
    imu_names: list[str],
    config: PipelineConfig,
) -> list[dict]:
    rows = []
    raw = methods["raw"]
    _, gait_freq = artifact_reduction_at_gait(
        raw[eeg_idx, :], raw[eeg_idx, :], raw[imu_idx, :], imu_names, rec.header.sfreq
    )
    for method in ("asr", "ican"):
        if method not in methods:
            continue
        freqs, reduction = gait_power_reduction_curve(
            raw[eeg_idx, :],
            methods[method][eeg_idx, :],
            rec.header.sfreq,
            nperseg_s=config.nperseg_s,
        )
        for f, value in zip(freqs, reduction):
            rows.append(
                {
                    "subject": rec.subject,
                    "task": rec.task,
                    "speed": rec.speed_label,
                    "method": method,
                    "frequency_hz": f,
                    "reduction_db": value,
                    "gait_frequency_hz": gait_freq,
                }
            )
    return rows


def collect_tradeoff(
    rec: Recording,
    methods: dict[str, np.ndarray],
    events,
    eeg_idx: list[int],
    imu_idx: list[int],
    imu_names: list[str],
) -> list[dict]:
    rows = []
    raw = methods["raw"]
    pz = channel_index_by_name(rec, "Pz")
    if pz is None:
        pz = eeg_idx[0]
    for method in ("asr", "ican"):
        if method not in methods:
            continue
        artifact_reduction, gait_freq = artifact_reduction_at_gait(
            raw[eeg_idx, :],
            methods[method][eeg_idx, :],
            raw[imu_idx, :],
            imu_names,
            rec.header.sfreq,
        )
        if rec.task == "ERP":
            signal_pres = erp_preservation_score(raw, methods[method], events, rec.header.sfreq, pz)
        else:
            signal_pres = np.nan
        rows.append(
            {
                "subject": rec.subject,
                "task": rec.task,
                "speed": rec.speed_label,
                "method": method,
                "artifact_reduction_db": artifact_reduction,
                "gait_frequency_hz": gait_freq,
                "signal_preservation": signal_pres,
            }
        )
    return rows


def collect_erp_waves(rec: Recording, methods: dict[str, np.ndarray], events) -> list[dict]:
    pz = channel_index_by_name(rec, "Pz")
    if pz is None:
        return []
    rows = []
    for method, data in methods.items():
        times, waves = erp_waveforms(data, events, rec.header.sfreq, pz)
        for label, wave in waves.items():
            for t, amp in zip(times, wave):
                rows.append(
                    {
                        "subject": rec.subject,
                        "speed": rec.speed_label,
                        "method": method,
                        "label": label,
                        "time_s": t,
                        "amplitude": amp,
                    }
                )
    return rows


def score_erp_tests(
    erp_train: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    erp_tests: list[tuple[str, str, str, np.ndarray, np.ndarray]],
) -> list[dict]:
    models = {}
    for key, (features, labels) in erp_train.items():
        if features.size == 0 or len(np.unique(labels)) < 2:
            continue
        models[key] = train_erp_classifier(features, labels)
    rows = []
    for subject, method, speed, features, labels in erp_tests:
        model = models.get((subject, method))
        if model is None:
            continue
        scores = score_erp_classifier(model, features, labels)
        rows.append(
            {
                "task": "ERP",
                "subject": subject,
                "speed": speed,
                "method": method,
                "auc": scores["auc"],
                "balanced_accuracy": scores["balanced_accuracy"],
                "accuracy": np.nan,
            }
        )
    return rows


def run_asr_sensitivity(
    recordings: list[Recording],
    config: PipelineConfig,
    cutoffs: list[float],
) -> list[dict]:
    if not cutoffs:
        return []
    rows = []
    by_subject: dict[str, list[Recording]] = defaultdict(list)
    for rec in recordings:
        if rec.task == "ERP":
            by_subject[rec.subject].append(rec)
    for subject, recs in by_subject.items():
        train = next((r for r in recs if r.session == "ses-01"), None)
        tests = [r for r in recs if r.speed_mps is not None]
        if train is None or not tests:
            continue
        train_data = read_brainvision_data(train)
        train_events = read_events(train, n_samples=train_data.shape[1])
        eeg_idx_train = channel_indices(train, "eeg", good_only=False)
        model = ASRModel.calibrate(train_data[eeg_idx_train, :], train.header.sfreq)
        clf_channels = pick_named(train, ERP_CHANNELS, good_only=False) or channel_indices(
            train, "scalp", good_only=True
        )
        for cutoff in cutoffs:
            clean_train = train_data.copy()
            all_eeg_train = channel_indices(train, "eeg", good_only=False)
            clean_train[all_eeg_train, :] = model.transform(train_data[all_eeg_train, :], cutoff=cutoff)
            train_features, train_labels = erp_features(clean_train, train_events, train.header.sfreq, clf_channels)
            if train_features.size == 0 or len(np.unique(train_labels)) < 2:
                continue
            clf = train_erp_classifier(train_features, train_labels)
            for test in tests:
                data = read_brainvision_data(test)
                events = read_events(test, n_samples=data.shape[1])
                all_eeg = channel_indices(test, "eeg", good_only=False)
                scalp = channel_indices(test, "scalp", good_only=True)
                imu_idx = channel_indices(test, "imu_motion", good_only=True)
                imu_names = test.channels.iloc[imu_idx]["name"].astype(str).tolist()
                clean = data.copy()
                clean[all_eeg, :] = model.transform(data[all_eeg, :], cutoff=cutoff)
                test_channels = pick_named(test, ERP_CHANNELS, good_only=False) or scalp
                features, labels = erp_features(clean, events, test.header.sfreq, test_channels)
                scores = score_erp_classifier(clf, features, labels)
                pz = channel_index_by_name(test, "Pz")
                preservation = np.nan if pz is None else erp_preservation_score(data, clean, events, test.header.sfreq, pz)
                artifact, gait_freq = artifact_reduction_at_gait(
                    data[scalp, :], clean[scalp, :], data[imu_idx, :], imu_names, test.header.sfreq
                )
                rows.append(
                    {
                        "subject": subject,
                        "speed": test.speed_label,
                        "cutoff": cutoff,
                        "auc": scores["auc"],
                        "balanced_accuracy": scores["balanced_accuracy"],
                        "artifact_reduction_db": artifact,
                        "signal_preservation": preservation,
                        "gait_frequency_hz": gait_freq,
                    }
                )
    return rows


def channel_index_by_name(rec: Recording, name: str) -> int | None:
    names = rec.channels["name"].astype(str).tolist()
    try:
        return names.index(name)
    except ValueError:
        return None


def write_outputs(
    tables_dir: Path,
    figures_dir: Path,
    coherence_rows: list[dict],
    gait_rows: list[dict],
    erp_wave_rows: list[dict],
    tradeoff_rows: list[dict],
    bci_rows: list[dict],
    asr_sensitivity_rows: list[dict],
    config: PipelineConfig,
) -> None:
    manifest = {
        "config": config.__dict__,
        "tables": [],
        "figures": [],
    }
    outputs = [
        ("coherence_curves.csv", coherence_rows, plot_coherence, "raw_eeg_imu_coherence_by_speed.png"),
        ("gait_power_reduction_curves.csv", gait_rows, plot_gait_reduction, "gait_power_reduction_by_speed.png"),
        ("erp_preservation_curves.csv", erp_wave_rows, plot_erp_preservation, "erp_preservation_pz.png"),
        ("tradeoff_metrics.csv", tradeoff_rows, plot_tradeoff, "artifact_signal_tradeoff.png"),
        ("bci_performance.csv", bci_rows, plot_bci, "bci_performance_by_speed.png"),
        ("asr_cutoff_sensitivity.csv", asr_sensitivity_rows, plot_asr_sensitivity, "asr_cutoff_sensitivity.png"),
    ]
    for csv_name, rows, plotter, fig_name in outputs:
        df = pd.DataFrame(rows)
        csv_path = tables_dir / csv_name
        df.to_csv(csv_path, index=False)
        manifest["tables"].append(str(csv_path))
        if not df.empty:
            fig_path = figures_dir / fig_name
            plotter(df, fig_path)
            manifest["figures"].append(str(fig_path))

    extra_figures = [
        (tradeoff_rows, plot_tradeoff_by_speed, "artifact_signal_tradeoff_by_speed.png"),
        (asr_sensitivity_rows, plot_asr_sensitivity_by_speed, "asr_cutoff_sensitivity_by_speed.png"),
    ]
    for rows, plotter, fig_name in extra_figures:
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        fig_path = figures_dir / fig_name
        plotter(df, fig_path)
        manifest["figures"].append(str(fig_path))
    (tables_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
