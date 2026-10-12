"""⑤ 클래스 설계안: 지정된 다섯 항목을 한 페이지로 작성한다."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .week1_common import CHANNEL_MUSCLES, CHANNEL_NAMES, GESTURE_NAMES, markdown_table


GESTURE_KO = {
    "fist": "주먹 쥐기",
    "open": "손 펴기",
    "pro": "회내",
    "extor": "회외",
}


def class_design_table() -> pd.DataFrame:
    rows = []
    class_id = 0
    for angle in (30, 60):
        for gesture_code in ("fist", "open", "pro", "extor"):
            gesture = GESTURE_NAMES[gesture_code]
            class_name = f"{angle}_{gesture.lower().replace(' ', '_')}"
            rows.append(
                {
                    "ID": class_id,
                    "클래스명": class_name,
                    "원자료 표기": f"{angle} {gesture_code}",
                    "정의": f"팔꿈치 {angle}°에서 {GESTURE_KO[gesture_code]}",
                }
            )
            class_id += 1
    return pd.DataFrame(rows)


def channel_design_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "채널": range(1, 6),
            "약어": CHANNEL_NAMES,
            "측정 근육": CHANNEL_MUSCLES,
            "부착 위치": ["전완 후면", "전완 후면", "전완 전면", "상완 전면", "상완 후면"],
            "주요 정보": [
                "손목 신전",
                "손가락 신전",
                "손가락 굴곡",
                "팔꿈치 굴곡·회외",
                "팔꿈치 신전",
            ],
        }
    )


def build_one_page_design() -> str:
    """Return a concise design document containing exactly five sections."""
    return f"""# ⑤ 클래스 설계안

**클래스 목록**

{markdown_table(class_design_table(), decimals=0)}

**채널 대응**

{markdown_table(channel_design_table(), decimals=0)}

**설계 근거**

손동작만 분류하면 같은 동작을 서로 다른 팔 자세에서 수행한 상황을 구분할 수 없다. 따라서 손동작 4종과 팔꿈치 각도 2종을 조합한 8개 클래스를 정의하여 한 번의 예측으로 동작과 각도 조건을 함께 출력한다.

**대안 검토**

| 방식 | 장점 | 단점 |
| --- | --- | --- |
| 8클래스 단일 분류 | 구현과 평가가 단순하고 조합별 오류를 혼동행렬에서 바로 확인할 수 있음 | 축이 늘면 클래스 수와 데이터 요구량이 곱셈으로 증가 |
| 다중 출력: 동작 4종 + 각도 2종 | 두 과제를 따로 최적화하면서 특징을 공유할 수 있음 | 손실 가중치와 결합 오류 평가가 필요 |
| 계층형: 동작 분류 후 각도 분류 | 동작별 각도 판별기를 구성할 수 있음 | 첫 단계 오류가 다음 단계로 전파됨 |

현재 데이터는 8개 조합이 클래스별 240회로 균형을 이루며 논문도 단일 분류 방식을 사용하므로, 8클래스 단일 분류를 기본안으로 선택한다.

**확장 시 문제**

각도를 3종으로 늘리면 12클래스, 4종으로 늘리면 16클래스가 된다. 현재와 같은 클래스당 240회를 유지하려면 각각 2,880회와 3,840회가 필요하다. 인접 각도의 신호 차이가 작을수록 각도 혼동이 증가하고 피험자별 근활성 차이의 영향도 커질 수 있다. 따라서 각도별 균형 수집과 피험자 독립 검증을 적용하고, 연속 각도 회귀 또는 다중 출력 모델도 비교해야 한다.
"""


def run(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    class_design_table().to_csv(
        tables_dir / "class_design.csv", index=False, encoding="utf-8-sig"
    )
    channel_design_table().to_csv(
        tables_dir / "channel_design.csv", index=False, encoding="utf-8-sig"
    )
    output_path = output_dir / "step5_class_design.md"
    output_path.write_text(build_one_page_design(), encoding="utf-8")
    print(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/week1"))
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
