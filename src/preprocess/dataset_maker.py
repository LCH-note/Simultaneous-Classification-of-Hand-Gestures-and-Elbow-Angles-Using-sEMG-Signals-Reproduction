"""시간축 정렬이 끝난 sEMG 데이터에서 25차원 특징 데이터셋을 만든다.

겹치는 윈도우가 학습셋과 테스트셋에 나뉘어 들어가지 않도록 시행을 먼저
층화 분할한다. 이후 각 집합 안에서 필터링, 세그먼트 추출, 윈도우 분할,
시간영역 특징 추출을 순서대로 수행한다.
"""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
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
INPUT_DIR = PROJECT_ROOT / 'data' / 'data_pre'
OUTPUT_PATH = PROJECT_ROOT / 'data' / 'dataset.npz'
TEST_SIZE = 0.2
RANDOM_STATE = 42


def label_of(path):
    """정렬 파일명에서 각도와 동작으로 구성된 클래스명을 반환한다."""
    stem = Path(path).stem.removesuffix('_pre')
    return stem.split('_', 1)[1]


def collect_trials(files):
    """정렬 CSV의 모든 시행을 ``(경로, 시작점, 번호, 라벨)``로 수집한다."""
    trials = []
    for path in files:
        marks = np.loadtxt(
            path,
            delimiter=',',
            skiprows=1,
            usecols=7,
        )
        starts = np.flatnonzero(marks == 1)
        label = label_of(path)
        for trial_number, start in enumerate(starts, start=1):
            trials.append((str(path), int(start), trial_number, label))
    return trials


def build(trials):
    """한 집합에 속한 시행에서 윈도우별 25차원 특징과 라벨을 만든다."""
    trials_by_file = defaultdict(list)
    for path, start, trial_number, label in trials:
        trials_by_file[path].append((start, trial_number, label))

    features = []
    labels = []
    for path, file_trials in trials_by_file.items():
        signals = np.loadtxt(
            path,
            delimiter=',',
            skiprows=1,
            usecols=range(5),
        )
        for start, _, label in file_trials:
            trial = signals[start:start + TRIAL_SAMPLES]
            segment = cut(preprocess(trial))
            for window in make_windows(segment):
                features.append(td_features(window))
                labels.append(label)

    return np.stack(features), np.asarray(labels)


def print_distribution(name, labels):
    """데이터셋의 클래스별 윈도우 수를 출력한다."""
    distribution = Counter(labels)
    print(f'{name} 클래스 분포:')
    for label in sorted(distribution):
        print(f'  {label}: {distribution[label]}')


def main():
    files = sorted(INPUT_DIR.glob('*_pre.csv'))
    trials = collect_trials(files)
    trial_labels = [trial[3] for trial in trials]

    train_trials, test_trials = train_test_split(
        trials,
        test_size=TEST_SIZE,
        stratify=trial_labels,
        random_state=RANDOM_STATE,
    )

    Xtr, ytr = build(train_trials)
    Xte, yte = build(test_trials)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    np.savez(
        OUTPUT_PATH,
        Xtr=Xtr_s,
        ytr=ytr,
        Xte=Xte_s,
        yte=yte,
    )

    print('정렬 CSV 파일 수:', len(files))
    print('전체 시행 수:', len(trials))
    print('학습 시행 수:', len(train_trials))
    print('테스트 시행 수:', len(test_trials))
    print('학습 데이터:', Xtr_s.shape, ytr.shape)
    print('테스트 데이터:', Xte_s.shape, yte.shape)
    print_distribution('학습', ytr)
    print_distribution('테스트', yte)
    print('학습셋 평균:', Xtr_s.mean(axis=0).round(6))
    print('학습셋 표준편차:', Xtr_s.std(axis=0).round(6))
    print('저장 파일:', OUTPUT_PATH)
    print('저장 완료')


if __name__ == '__main__':
    main()
