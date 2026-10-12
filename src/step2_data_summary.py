"""② 데이터 요약: 파일, 클래스, 시행, 배열 및 채널 구성을 확인한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .week1_common import GESTURE_NAMES, SIGNAL_COLUMNS, parse_label, trial_starts


def summarize_dataset(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for path in sorted(data_dir.glob("*.csv")):
        label = parse_label(path)
        frame = pd.read_csv(path, usecols=["time", "onset", "mark"])
        starts = trial_starts(frame)
        rows.append(
            {
                "file": path.name,
                "participant": label.participant,
                "angle_deg": label.angle,
                "gesture_code": label.gesture,
                "gesture": GESTURE_NAMES[label.gesture],
                "class_name": label.class_name,
                "rows": len(frame),
                "complete_trials": len(starts),
                "signal_channels": len(SIGNAL_COLUMNS),
                "total_columns": 8,
                "time_start_s": frame["time"].iloc[0],
                "time_end_s": frame["time"].iloc[-1],
            }
        )

    file_summary = pd.DataFrame(rows)
    class_summary = (
        file_summary.groupby(
            ["angle_deg", "gesture_code", "gesture", "class_name"],
            as_index=False,
        )
        .agg(
            participants=("participant", "nunique"),
            files=("file", "count"),
            trials=("complete_trials", "sum"),
            min_rows=("rows", "min"),
            max_rows=("rows", "max"),
        )
        .sort_values(["angle_deg", "gesture"])
        .reset_index(drop=True)
    )
    return file_summary, class_summary


def run(data_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_summary, class_summary = summarize_dataset(data_dir)
    file_summary.to_csv(
        output_dir / "file_summary.csv", index=False, encoding="utf-8-sig"
    )
    class_summary.to_csv(
        output_dir / "class_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(class_summary.to_string(index=False))
    print(f"전체 시행 수: {int(class_summary['trials'].sum()):,}")
    return file_summary, class_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/week1/tables")
    )
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
