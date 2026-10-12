"""③ 신호 그림: 한 시행의 5채널 sEMG와 동작 구간을 그린다."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .week1_common import CHANNEL_NAMES, FS, extract_trial


def plot_five_channels(signal: np.ndarray, output_path: Path, title: str) -> None:
    time_s = np.arange(len(signal)) / FS
    signal_x1e3 = signal * 1_000
    fig, axes = plt.subplots(5, 1, figsize=(12, 9), sharex=True)
    for channel, axis in enumerate(axes):
        axis.plot(time_s, signal_x1e3[:, channel], color="#155e75", linewidth=0.55)
        axis.axvspan(2.0, 3.0, color="#f59e0b", alpha=0.18)
        axis.set_ylabel(f"{CHANNEL_NAMES[channel]}\n(raw x1e3)")
        axis.grid(alpha=0.2, linewidth=0.5)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Time from trial marker (s)")
    fig.text(0.995, 0.99, "Shaded interval: motion (2-3 s)", ha="right", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(
    data_dir: Path,
    output_dir: Path,
    participant: str = "a",
    angle: int = 30,
    gesture: str = "fist",
    trial: int = 1,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / f"{participant}_{angle} {gesture}.csv"
    signal = extract_trial(source, trial)
    output_path = output_dir / f"signal_5ch_{participant}_{angle}_{gesture}_trial{trial:02d}.png"
    plot_five_channels(
        signal,
        output_path,
        f"Participant {participant}, {gesture} at {angle} degrees, trial {trial}",
    )
    print(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/week1/figures")
    )
    parser.add_argument("--participant", default="a")
    parser.add_argument("--angle", type=int, default=30, choices=[30, 60])
    parser.add_argument(
        "--gesture", default="fist", choices=["fist", "open", "pro", "extor"]
    )
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()
    run(
        args.data_dir,
        args.output_dir,
        args.participant,
        args.angle,
        args.gesture,
        args.trial,
    )


if __name__ == "__main__":
    main()
