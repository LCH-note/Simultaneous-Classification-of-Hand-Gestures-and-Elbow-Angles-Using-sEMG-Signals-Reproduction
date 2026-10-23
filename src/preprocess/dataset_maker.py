"""파일별 시행을 먼저 분할한 뒤 시행 단위 특징 데이터셋을 만든다."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    from .data_preprocess import (
        TRIAL_SAMPLES,
        cut,
        make_windows,
        preprocess,
        td_features,
    )
except ImportError:
    from data_preprocess import (
        TRIAL_SAMPLES,
        cut,
        make_windows,
        preprocess,
        td_features,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "data_pre"
OUTPUT_PATH = PROJECT_ROOT / "data" / "dataset_improve_time_axis.npz"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def metadata_of(path):
    """파일명에서 피험자와 각도·동작 라벨을 반환한다."""
    stem = Path(path).stem.removesuffix("_pre")
    subject, label = stem.split("_", 1)
    return subject, label


def split_trials_by_file(files):
    """각 파일의 시행 30개를 먼저 8:2로 분할한다."""
    rng = np.random.default_rng(RANDOM_STATE)
    train_trials = []
    test_trials = []

    for path in files:
        marks = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            usecols=7,
        )
        starts = np.flatnonzero(marks == 1)
        trial_indices = np.arange(len(starts))
        rng.shuffle(trial_indices)

        test_count = round(len(starts) * TEST_SIZE)
        test_indices = set(trial_indices[:test_count])
        subject, label = metadata_of(path)

        for trial_index, start in enumerate(starts):
            trial = (
                str(path),
                int(start),
                trial_index + 1,
                subject,
                label,
            )
            if trial_index in test_indices:
                test_trials.append(trial)
            else:
                train_trials.append(trial)

    return train_trials, test_trials


def build(trials):
    """각 시행을 19개 윈도우의 25차원 특징으로 변환한다."""
    trials_by_file = defaultdict(list)
    for trial in trials:
        trials_by_file[trial[0]].append(trial[1:])

    features = []
    labels = []
    subjects = []
    trial_ids = []

    for path, file_trials in trials_by_file.items():
        signals = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            usecols=range(5),
        )
        file_id = Path(path).stem.removesuffix("_pre")

        for start, trial_number, subject, label in file_trials:
            trial = signals[start:start + TRIAL_SAMPLES]
            segment = cut(preprocess(trial))
            trial_features = [
                td_features(window)
                for window in make_windows(segment)
            ]

            features.append(np.stack(trial_features))
            labels.append(label)
            subjects.append(subject)
            trial_ids.append(f"{file_id}_{trial_number:02d}")

    return (
        np.stack(features),
        np.asarray(labels),
        np.asarray(subjects),
        np.asarray(trial_ids),
    )


def scale_features(Xtr, Xte):
    """학습 윈도우로만 표준화 통계를 계산한다."""
    feature_count = Xtr.shape[-1]
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr.reshape(-1, feature_count))
    Xte_scaled = scaler.transform(Xte.reshape(-1, feature_count))
    return (
        Xtr_scaled.reshape(Xtr.shape),
        Xte_scaled.reshape(Xte.shape),
        scaler,
    )


def print_distribution(name, labels):
    """시행 단위 클래스 분포를 출력한다."""
    print(f"{name} 클래스 분포:")
    for label, count in sorted(Counter(labels).items()):
        print(f"  {label}: {count}")


def main():
    files = sorted(INPUT_DIR.glob("*.csv"))
    train_trials, test_trials = split_trials_by_file(files)

    Xtr, ytr, subject_tr, trial_id_tr = build(train_trials)
    Xte, yte, subject_te, trial_id_te = build(test_trials)
    Xtr, Xte, scaler = scale_features(Xtr, Xte)

    np.savez(
        OUTPUT_PATH,
        Xtr=Xtr,
        ytr=ytr,
        Xte=Xte,
        yte=yte,
        subject_tr=subject_tr,
        subject_te=subject_te,
        trial_id_tr=trial_id_tr,
        trial_id_te=trial_id_te,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )

    print("입력 CSV 파일 수:", len(files))
    print("학습 시행 수:", len(train_trials))
    print("테스트 시행 수:", len(test_trials))
    print("학습 데이터:", Xtr.shape, ytr.shape)
    print("테스트 데이터:", Xte.shape, yte.shape)
    print_distribution("학습", ytr)
    print_distribution("테스트", yte)
    print("학습 특징 평균:", Xtr.mean(axis=(0, 1)).round(6))
    print("학습 특징 표준편차:", Xtr.std(axis=(0, 1)).round(6))
    print("저장 파일:", OUTPUT_PATH)
    print("저장 완료")


if __name__ == "__main__":
    main()
