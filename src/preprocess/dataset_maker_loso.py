"""피험자 단위로 분리한 LOSO 특징 데이터셋을 생성한다."""

from collections import Counter
from pathlib import Path

import numpy as np

# 모듈 실행과 IDE의 파일 직접 실행을 모두 지원한다.
if __package__:
    from .data_preprocess import (
        TRIAL_SAMPLES, cut, make_windows, preprocess, td_features,
    )
else:
    from data_preprocess import (
        TRIAL_SAMPLES, cut, make_windows, preprocess, td_features,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NO_TIME_AXIS_INPUT_DIR = PROJECT_ROOT / "data" / "data"
TIME_AXIS_INPUT_DIR = PROJECT_ROOT / "data" / "data_pre"
OUTPUT_DIR = PROJECT_ROOT / "data" / "loso"
NO_TIME_AXIS_OUTPUT_PATH = OUTPUT_DIR / "dataset_loso.npz"
TIME_AXIS_OUTPUT_PATH = OUTPUT_DIR / "dataset_time_axis_loso.npz"

DATASET_CONFIGS = (
    (NO_TIME_AXIS_INPUT_DIR, NO_TIME_AXIS_OUTPUT_PATH),
    (TIME_AXIS_INPUT_DIR, TIME_AXIS_OUTPUT_PATH),
)


def metadata_of(path):
    """파일명에서 피험자와 각도·동작 라벨을 반환한다."""
    stem = Path(path).stem.removesuffix("_pre")
    subject, label = stem.split("_", 1)
    return subject, label


def build(files):
    """모든 피험자에게 같은 시행 단위 전처리 과정을 적용한다."""
    features = []
    labels = []
    subjects = []
    trial_ids = []

    for path in files:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        signals = data[:, :5]
        starts = np.flatnonzero(data[:, 7] == 1)
        subject, label = metadata_of(path)
        file_id = path.stem.removesuffix("_pre")

        for trial_number, start in enumerate(starts, start=1):
            # 필터링 전에 시행을 분리하여 시행 경계의 신호가 섞이지 않게 한다.
            trial = signals[start:start + TRIAL_SAMPLES].copy()
            segment = cut(preprocess(trial))
            windows = make_windows(segment)
            trial_features = np.stack(
                [td_features(window) for window in windows]
            )

            features.append(trial_features)
            labels.append(label)
            subjects.append(subject)
            trial_ids.append(f"{file_id}_{trial_number:02d}")

    return (
        np.stack(features),
        np.asarray(labels),
        np.asarray(subjects),
        np.asarray(trial_ids),
    )


def print_distribution(name, labels):
    """시행 단위 클래스 분포를 출력한다."""
    print(f"{name} 클래스 분포:")
    for label, count in sorted(Counter(labels).items()):
        print(f"  {label}: {count}")


def create_dataset(input_dir, output_path):
    """한 데이터 조건의 전체 피험자 특징을 통합 파일로 저장한다."""
    files = sorted(input_dir.glob("*.csv"))
    features, labels, subjects, trial_ids = build(files)

    np.savez(
        output_path,
        X=features,
        y=labels,
        subjects=subjects,
        trial_ids=trial_ids,
    )

    print(f"입력 경로: {input_dir}")
    print(f"입력 CSV: {len(files)}")
    print(f"전체 특징: {features.shape}")
    print(f"피험자: {sorted(set(subjects))}")
    print_distribution("전체", labels)
    print(f"저장 파일: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for input_dir, output_path in DATASET_CONFIGS:
        create_dataset(input_dir, output_path)


if __name__ == "__main__":
    main()
