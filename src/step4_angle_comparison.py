"""④ 비교 관찰: 같은 동작의 30°와 60° 신호 및 RMS를 비교한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .week1_common import (
    CHANNEL_MUSCLES,
    CHANNEL_NAMES,
    FS,
    GESTURE_NAMES,
    MOTION_SLICE,
    REST_SLICES,
    SIGNAL_COLUMNS,
    TRIAL_SAMPLES,
    extract_trial,
    load_signal,
    parse_label,
    trial_starts,
)


def calculate_trial_rms(data_dir: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for path in sorted(data_dir.glob("*.csv")):
        label = parse_label(path)
        frame = load_signal(path)
        values = frame[SIGNAL_COLUMNS].to_numpy()
        for trial_index, start in enumerate(trial_starts(frame), start=1):
            trial = values[start : start + TRIAL_SAMPLES]
            motion = trial[MOTION_SLICE]
            rest = np.concatenate([trial[REST_SLICES[0]], trial[REST_SLICES[1]]])
            motion_rms = np.sqrt(np.mean(np.square(motion), axis=0)) * 1_000
            rest_rms = np.sqrt(np.mean(np.square(rest), axis=0)) * 1_000
            for channel in range(5):
                records.append(
                    {
                        "participant": label.participant,
                        "angle_deg": label.angle,
                        "gesture_code": label.gesture,
                        "gesture": GESTURE_NAMES[label.gesture],
                        "trial": trial_index,
                        "channel": channel + 1,
                        "channel_name": CHANNEL_NAMES[channel],
                        "muscle": CHANNEL_MUSCLES[channel],
                        "motion_rms_x1e3": motion_rms[channel],
                        "rest_rms_x1e3": rest_rms[channel],
                        "motion_to_rest_ratio": motion_rms[channel]
                        / max(rest_rms[channel], np.finfo(float).eps),
                    }
                )
    return pd.DataFrame(records)


def summarize_rms(trial_rms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        trial_rms.groupby(
            ["angle_deg", "gesture_code", "gesture", "channel", "channel_name"],
            as_index=False,
        )
        .agg(
            trials=("motion_rms_x1e3", "count"),
            motion_rms_mean_x1e3=("motion_rms_x1e3", "mean"),
            motion_rms_std_x1e3=("motion_rms_x1e3", "std"),
            rest_rms_mean_x1e3=("rest_rms_x1e3", "mean"),
            motion_to_rest_mean=("motion_to_rest_ratio", "mean"),
        )
        .sort_values(["gesture", "channel", "angle_deg"])
        .reset_index(drop=True)
    )
    comparison = summary.pivot(
        index=["gesture_code", "gesture", "channel", "channel_name"],
        columns="angle_deg",
        values="motion_rms_mean_x1e3",
    ).reset_index()
    comparison.columns.name = None
    comparison = comparison.rename(columns={30: "rms_30_x1e3", 60: "rms_60_x1e3"})
    comparison["difference_60_minus_30_x1e3"] = (
        comparison["rms_60_x1e3"] - comparison["rms_30_x1e3"]
    )
    comparison["percent_change_60_vs_30"] = (
        comparison["difference_60_minus_30_x1e3"]
        / comparison["rms_30_x1e3"]
        * 100
    )
    return summary, comparison.sort_values(["gesture", "channel"]).reset_index(drop=True)


def plot_signal_comparison(
    signal_30: np.ndarray, signal_60: np.ndarray, output_path: Path
) -> None:
    time_s = np.arange(TRIAL_SAMPLES) / FS
    signals = [signal_30 * 1_000, signal_60 * 1_000]
    fig, axes = plt.subplots(5, 2, figsize=(15, 10), sharex=True)
    for channel in range(5):
        row_values = np.concatenate([signals[0][:, channel], signals[1][:, channel]])
        limit = max(float(np.quantile(np.abs(row_values), 0.995)) * 1.1, 0.01)
        for column, angle in enumerate((30, 60)):
            axis = axes[channel, column]
            axis.plot(
                time_s,
                signals[column][:, channel],
                color=("#1d4ed8" if angle == 30 else "#b91c1c"),
                linewidth=0.5,
            )
            axis.axvspan(2.0, 3.0, color="#f59e0b", alpha=0.16)
            axis.set_ylim(-limit, limit)
            axis.grid(alpha=0.2, linewidth=0.5)
            if column == 0:
                axis.set_ylabel(f"{CHANNEL_NAMES[channel]} (raw x1e3)")
            if channel == 0:
                axis.set_title(f"Elbow angle: {angle} degrees")
    axes[-1, 0].set_xlabel("Time from trial marker (s)")
    axes[-1, 1].set_xlabel("Time from trial marker (s)")
    fig.suptitle("Same participant and gesture: 30 vs 60 degrees", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rms_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    gestures = ["Fist", "Open hand", "Pronation", "Supination"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    x = np.arange(5)
    width = 0.36
    for axis, gesture in zip(axes.flat, gestures):
        subset = comparison[comparison["gesture"] == gesture]
        axis.bar(x - width / 2, subset["rms_30_x1e3"], width, label="30 degrees")
        axis.bar(x + width / 2, subset["rms_60_x1e3"], width, label="60 degrees")
        axis.set_title(gesture)
        axis.set_xticks(x, CHANNEL_NAMES)
        axis.set_ylabel("Mean motion RMS (raw units x1e3)")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Angle comparison across all 1,920 trials", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(
    data_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    participant: str = "a",
    gesture: str = "fist",
    trial: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    trial_rms = calculate_trial_rms(data_dir)
    rms_summary, comparison = summarize_rms(trial_rms)
    rms_summary.to_csv(tables_dir / "rms_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(
        tables_dir / "angle_comparison.csv", index=False, encoding="utf-8-sig"
    )

    signal_30 = extract_trial(data_dir / f"{participant}_30 {gesture}.csv", trial)
    signal_60 = extract_trial(data_dir / f"{participant}_60 {gesture}.csv", trial)
    plot_signal_comparison(
        signal_30,
        signal_60,
        figures_dir / f"comparison_{participant}_{gesture}_trial{trial:02d}_30_vs_60.png",
    )
    plot_rms_comparison(comparison, figures_dir / "rms_angle_comparison_all_trials.png")
    print(comparison[comparison["gesture_code"] == gesture].round(3).to_string(index=False))
    return rms_summary, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/data"))
    parser.add_argument(
        "--tables-dir", type=Path, default=Path("outputs/week1/tables")
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=Path("outputs/week1/figures")
    )
    parser.add_argument("--participant", default="a")
    parser.add_argument(
        "--gesture", default="fist", choices=["fist", "open", "pro", "extor"]
    )
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()
    run(
        args.data_dir,
        args.tables_dir,
        args.figures_dir,
        args.participant,
        args.gesture,
        args.trial,
    )


if __name__ == "__main__":
    main()
