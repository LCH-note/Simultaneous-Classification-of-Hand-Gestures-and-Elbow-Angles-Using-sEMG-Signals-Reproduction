"""Run Week 1 assignment items ② through ⑤ in order."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import step2_data_summary, step3_signal_plot, step4_angle_comparison
from . import step5_class_design


def run(data_dir: Path, output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    print("[② 데이터 요약]")
    step2_data_summary.run(data_dir, tables_dir)
    print("\n[③ 신호 그림]")
    step3_signal_plot.run(data_dir, figures_dir)
    print("\n[④ 비교 관찰]")
    step4_angle_comparison.run(data_dir, tables_dir, figures_dir)
    print("\n[⑤ 클래스 설계안]")
    step5_class_design.run(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/week1")
    )
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
