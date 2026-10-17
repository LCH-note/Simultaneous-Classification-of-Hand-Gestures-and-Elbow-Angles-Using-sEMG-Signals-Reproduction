"""sEMG 시행의 근활성 시작점을 5초 구간 내 2초로 정렬한다.

이 모듈은 ``data_preprocessing_rule.txt``에 정의된 규칙을 구현한다.
원본 CSV는 수정하지 않으며, 결과는 기본적으로 ``data/data_pre``에
``원본파일명_pre.csv`` 형식으로 저장한다.

Onset 검출은 mark 직전 1초 baseline과 200 ms RMS를 사용한다. 먼저
``bandpass 2~4``에 P1(평균 + 5 SD), P2(중앙값 + 5 x 1.4826 x MAD)를
적용하고, 채널별 100 ms 지속 onset의 차이가 300 ms 이내인 순차 판정과
2개 채널의 100 ms 동시 판정을 차례로 수행한다. 이 단계가 모두 실패한
시행에만 ``bandpass 1~5`` 전체 채널로 같은 검출 체인을 다시 수행한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


FS = 1000  # 샘플링 주파수(Hz): 1초에 수집되는 샘플 수
TRIAL_SECONDS = 5  # 시행 1회의 길이(초)
TRIAL_SAMPLES = FS * TRIAL_SECONDS  # 시행 1회에 포함되는 전체 샘플 수
TARGET_ONSET_SECONDS = 2.0  # 정렬 후 근활성 시작점의 목표 시간(초)
TARGET_ONSET_SAMPLE = int(FS * TARGET_ONSET_SECONDS)  # 목표 onset의 샘플 위치

SIGNAL_COLUMNS = tuple(  # 전처리 대상 sEMG 신호 열 Ch1~Ch5
    f"bandpass {index}" for index in range(1, 6)
)
TIME_COLUMN = "time"  # 시간축 열 이름
ONSET_COLUMN = "onset"  # 정렬된 근활성 시작점 표지 열 이름
MARK_COLUMN = "mark"  # 각 5초 시행의 시작점 표지 열 이름
REQUIRED_COLUMNS = SIGNAL_COLUMNS + (  # 입력 CSV에 반드시 필요한 전체 열과 순서
    TIME_COLUMN,
    ONSET_COLUMN,
    MARK_COLUMN,
)

ONSET_CHANNELS = (1, 2, 3)  # 1차 검출 채널의 0-based 인덱스: 실제 Ch2~Ch4
FALLBACK_ONSET_CHANNELS = (0, 1, 2, 3, 4)  # 1차 실패 시 사용하는 실제 Ch1~Ch5
MIN_ACTIVE_CHANNELS = 2  # onset 판정에 필요한 최소 활성 채널 수
RMS_WINDOW_MS = 200  # 신호 포락선 계산에 사용하는 후행 RMS 구간(ms)
BASELINE_END_SECONDS = 1.0  # 임계값 계산에 사용하는 mark 직전 baseline 길이(초)
THRESHOLD_STD = 5.0  # P1/P2 임계값 계산 시 변동성에 곱하는 계수
ROBUST_MAD_SCALE = 1.4826  # MAD를 표준편차 수준으로 환산하는 보정 계수
SUSTAINED_MS = 100  # 임계값 초과가 지속되어야 하는 최소 시간(ms)
SEQUENTIAL_TOLERANCE_MS = 300  # 두 채널 onset 시작점의 최대 허용 차이(ms)
SEARCH_START_SECONDS = 0.0  # mark 이후 onset 검색 시작 시간(초)
SEARCH_END_SECONDS = 4.0  # mark 이후 onset 검색 종료 시간(초)
PROCESSING_RULE_VERSION = (  # 결과 보고서에 기록되는 전처리 규칙 버전
    "2026-10-16-stepwise-all5-fallback-v3"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 현재 파일을 기준으로 찾은 프로젝트 루트
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "data"  # Run 버튼 실행 시 원본 검색 경로
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "data_pre"  # 기본 전처리 결과 경로


class OnsetDetectionError(ValueError):
    """한 시행에서 규칙에 맞는 onset을 찾지 못했을 때 발생한다."""


@dataclass(frozen=True)
class OnsetParameters:
    """재현 가능한 onset 검출 설정."""

    channels: tuple[int, ...]
    min_active_channels: int
    rms_window_ms: int
    baseline_end_seconds: float
    threshold_std: float
    robust_mad_scale: float
    sustained_ms: int
    sequential_tolerance_ms: int
    search_start_seconds: float
    search_end_seconds: float
    fallback_channels: tuple[int, ...]
    stage_order: tuple[str, ...]


@dataclass(frozen=True)
class TrialAlignmentReport:
    """한 시행의 검출 및 이동 내역."""

    trial_number: int
    mark_index: int
    detected_onset_sample: int | None
    detected_onset_seconds: float | None
    shift_samples: int
    source_start_index: int
    source_end_index_exclusive: int
    filled_prefix_samples: int
    filled_suffix_samples: int
    activation_candidate_count: int
    preactivation_detected: bool | None
    onset_detection_failed: bool
    onset_detection_stage: str
    detection_channels: list[int]
    contributing_channels: list[int]
    contributing_channel_onsets: dict[str, int]
    contributing_onset_difference_samples: int | None
    fallback_attempted: bool
    fallback_used: bool
    alignment_applied: bool
    qc_flags: list[str]


@dataclass(frozen=True)
class PreprocessReport:
    """파일 단위 전처리 결과와 추적 정보."""

    input_path: str
    output_path: str
    report_path: str | None
    sampling_rate_hz: int
    trial_samples: int
    target_onset_sample: int
    trial_count: int
    input_mark_intervals: list[int]
    irregular_mark_intervals: list[int]
    donor_trials: list[int]
    processing_rule_version: str
    onset_detection_policy: str
    onset_parameters: OnsetParameters
    trials: list[TrialAlignmentReport]


@dataclass
class _LoadedCsv:
    header: list[str]
    signal_numbers: np.ndarray
    signal_text: np.ndarray
    time_text: list[str]
    mark_indices: np.ndarray


@dataclass(frozen=True)
class _DetectionResult:
    """한 검출 단계에서 선택된 onset과 감사용 진단 정보."""

    onset_sample: int
    stage: str
    channels: tuple[int, ...]
    contributing_channels: tuple[int, ...]
    contributing_channel_onsets: dict[int, int]
    onset_difference_samples: int | None
    activation_candidate_count: int


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name}은(는) 양수여야 합니다.")


def _load_csv(file_path: Path) -> _LoadedCsv:
    """필요한 열을 검증하고 숫자·원문 신호를 함께 읽는다."""
    signal_rows: list[list[float]] = []
    signal_text_rows: list[list[str]] = []
    time_text: list[str] = []
    marks: list[int] = []

    with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"빈 CSV 파일입니다: {file_path}") from exc

        if tuple(header) != REQUIRED_COLUMNS:
            raise ValueError(
                "CSV 열이 예상 형식과 다릅니다. "
                f"필요한 열: {list(REQUIRED_COLUMNS)}, 실제 열: {header}"
            )

        for row_index, row in enumerate(reader):
            if len(row) != len(REQUIRED_COLUMNS):
                raise ValueError(
                    f"{row_index + 2}행의 열 수가 {len(row)}개입니다. "
                    f"{len(REQUIRED_COLUMNS)}개가 필요합니다."
                )
            try:
                numeric_signal = [float(value) for value in row[:5]]
                mark_value = float(row[7])
                float(row[5])
            except ValueError as exc:
                raise ValueError(f"{row_index + 2}행에 숫자가 아닌 값이 있습니다.") from exc

            if not np.isfinite(numeric_signal).all():
                raise ValueError(f"{row_index + 2}행 신호에 NaN 또는 무한대가 있습니다.")

            signal_rows.append(numeric_signal)
            signal_text_rows.append(row[:5])
            time_text.append(row[5])
            if mark_value == 1:
                marks.append(row_index)

    if not signal_rows:
        raise ValueError(f"데이터 행이 없습니다: {file_path}")
    if not marks:
        raise ValueError(f"mark=1인 행이 없습니다: {file_path}")

    return _LoadedCsv(
        header=header,
        signal_numbers=np.asarray(signal_rows, dtype=np.float64),
        signal_text=np.asarray(signal_text_rows, dtype=object),
        time_text=time_text,
        mark_indices=np.asarray(marks, dtype=np.int64),
    )


def _rolling_rms(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """시간축을 따라 후행 RMS를 계산한다."""
    if signal.ndim != 2:
        raise ValueError("signal은 (샘플 수, 채널 수) 모양이어야 합니다.")
    if not 1 <= window_samples <= len(signal):
        raise ValueError("RMS window가 신호 길이 범위 밖입니다.")

    squared = signal * signal
    cumulative = np.vstack(
        [np.zeros((1, signal.shape[1]), dtype=np.float64), np.cumsum(squared, axis=0)]
    )
    rms = np.empty_like(signal, dtype=np.float64)

    if window_samples > 1:
        early_counts = np.arange(1, window_samples, dtype=np.float64)[:, None]
        rms[: window_samples - 1] = np.sqrt(
            cumulative[1:window_samples] / early_counts
        )

    window_sums = cumulative[window_samples:] - cumulative[:-window_samples]
    rms[window_samples - 1 :] = np.sqrt(window_sums / window_samples)
    return rms


def _sustained_runs(mask: np.ndarray, length: int) -> list[tuple[int, int]]:
    """Boolean 배열에서 ``length`` 이상인 연속 True 구간을 반환한다."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, active in enumerate(mask):
        if active and start is None:
            start = index
        elif not active and start is not None:
            if index - start >= length:
                runs.append((start, index))
            start = None
    if start is not None and len(mask) - start >= length:
        runs.append((start, len(mask)))
    return runs


def detect_onset_legacy(
    trial_signal: np.ndarray,
    *,
    fs: int = FS,
    channels: Sequence[int] = ONSET_CHANNELS,
    min_active_channels: int = MIN_ACTIVE_CHANNELS,
    rms_window_ms: int = RMS_WINDOW_MS,
    baseline_end_seconds: float = BASELINE_END_SECONDS,
    threshold_std: float = THRESHOLD_STD,
    sustained_ms: int = SUSTAINED_MS,
    search_start_seconds: float = SEARCH_START_SECONDS,
    search_end_seconds: float = SEARCH_END_SECONDS,
) -> tuple[int, int, bool]:
    """이전의 시행 내부 baseline 방식으로 onset을 검출한다.

    이 함수는 과거 결과 비교만을 위한 진단용 구현이다. 규칙 기반 전처리에는
    mark 직전 baseline과 Ch2~4 → Ch1~5 fallback을 구현한
    :func:`detect_onset_stepwise`를 사용해야 한다.

    Returns:
        ``(onset 샘플, 검색 범위 내 활성 구간 수, 검색 시작 전 선행활성 여부)``.
    """
    warnings.warn(
        "detect_onset_legacy()는 과거 결과 비교용입니다. "
        "전처리에는 detect_onset_stepwise()를 사용하십시오.",
        FutureWarning,
        stacklevel=2,
    )
    trial_signal = np.asarray(trial_signal, dtype=np.float64)
    if trial_signal.ndim != 2 or trial_signal.shape[1] < 5:
        raise ValueError("trial_signal은 5개 이상의 채널을 가진 2차원 배열이어야 합니다.")
    if len(trial_signal) != TRIAL_SAMPLES * fs // FS:
        raise ValueError(f"시행 길이는 {TRIAL_SECONDS}초여야 합니다.")
    if not np.isfinite(trial_signal).all():
        raise ValueError("시행 신호에 NaN 또는 무한대가 있습니다.")

    _validate_positive("fs", fs)
    _validate_positive("rms_window_ms", rms_window_ms)
    _validate_positive("baseline_end_seconds", baseline_end_seconds)
    _validate_positive("sustained_ms", sustained_ms)
    if threshold_std < 0:
        raise ValueError("threshold_std는 0 이상이어야 합니다.")

    channel_indices = tuple(int(channel) for channel in channels)
    if len(set(channel_indices)) != len(channel_indices):
        raise ValueError("channels에 중복된 채널이 있습니다.")
    if not channel_indices or min(channel_indices) < 0:
        raise ValueError("channels에는 0 이상의 채널 인덱스가 필요합니다.")
    if max(channel_indices) >= trial_signal.shape[1]:
        raise ValueError("channels에 신호 배열 범위 밖의 채널이 있습니다.")
    if not 1 <= min_active_channels <= len(channel_indices):
        raise ValueError("min_active_channels가 channels 수의 범위를 벗어났습니다.")

    window_samples = max(1, round(rms_window_ms * fs / 1000))
    sustained_samples = max(1, round(sustained_ms * fs / 1000))
    baseline_end = round(baseline_end_seconds * fs)
    search_start = round(search_start_seconds * fs)
    search_end = round(search_end_seconds * fs)

    if not window_samples <= baseline_end <= len(trial_signal):
        raise ValueError("baseline 구간은 RMS window 이후이면서 시행 안에 있어야 합니다.")
    if not 0 <= search_start < search_end <= len(trial_signal):
        raise ValueError("onset 검색 범위가 시행 길이를 벗어났습니다.")

    envelope = _rolling_rms(trial_signal[:, channel_indices], window_samples)
    baseline = envelope[window_samples - 1 : baseline_end]
    threshold = baseline.mean(axis=0) + threshold_std * baseline.std(axis=0)
    threshold = np.maximum(threshold, np.finfo(np.float64).eps)
    channel_active = envelope > threshold
    combined_active = channel_active.sum(axis=1) >= min_active_channels

    search_runs_relative = _sustained_runs(
        combined_active[search_start:search_end], sustained_samples
    )
    if not search_runs_relative:
        raise OnsetDetectionError("지정된 검색 범위에서 지속 조건을 만족하는 onset이 없습니다.")

    onset_sample = search_start + search_runs_relative[0][0]
    preactivation_detected = bool(
        _sustained_runs(combined_active[:search_start], sustained_samples)
    )
    return onset_sample, len(search_runs_relative), preactivation_detected


def _thresholds_from_mark_baselines(
    signal: np.ndarray,
    mark_indices: np.ndarray,
    *,
    fs: int,
    rms_window_ms: int,
    baseline_seconds: float,
    threshold_std: float,
    robust_mad_scale: float,
) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """시행별 mark 직전 baseline에서 P1/P2 채널 임계값을 계산한다."""
    _validate_positive("fs", fs)
    _validate_positive("rms_window_ms", rms_window_ms)
    _validate_positive("baseline_seconds", baseline_seconds)
    _validate_positive("robust_mad_scale", robust_mad_scale)
    if threshold_std < 0:
        raise ValueError("threshold_std는 0 이상이어야 합니다.")

    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] < 5:
        raise ValueError("signal은 5개 이상의 채널을 가진 2차원 배열이어야 합니다.")
    if not np.isfinite(signal).all():
        raise ValueError("신호에 NaN 또는 무한대가 있습니다.")

    baseline_samples = max(1, round(baseline_seconds * fs))
    window_samples = max(1, round(rms_window_ms * fs / 1000))
    if window_samples > baseline_samples:
        raise ValueError("RMS window는 mark 직전 baseline 길이보다 길 수 없습니다.")

    p1 = np.full((len(mark_indices), signal.shape[1]), np.nan, dtype=np.float64)
    p2 = np.full_like(p1, np.nan)
    complete_baseline: list[bool] = []

    for trial_index, mark_value in enumerate(mark_indices):
        mark_index = int(mark_value)
        if mark_index < baseline_samples:
            complete_baseline.append(False)
            continue

        baseline_signal = signal[mark_index - baseline_samples : mark_index]
        envelope = _rolling_rms(baseline_signal, window_samples)
        baseline = envelope[window_samples - 1 :]

        mean = baseline.mean(axis=0)
        standard_deviation = baseline.std(axis=0)
        p1[trial_index] = mean + threshold_std * standard_deviation

        median = np.median(baseline, axis=0)
        mad = np.median(np.abs(baseline - median), axis=0)
        p2[trial_index] = median + threshold_std * robust_mad_scale * mad
        complete_baseline.append(True)

    valid = np.asarray(complete_baseline, dtype=bool)
    if not valid.any():
        raise ValueError(
            "mark 직전 baseline이 충분한 시행이 없어 onset 임계값을 계산할 수 없습니다."
        )

    p1_donor_median = np.median(p1[valid], axis=0)
    p2_donor_median = np.median(p2[valid], axis=0)
    for trial_index, is_complete in enumerate(complete_baseline):
        if not is_complete:
            p1[trial_index] = p1_donor_median
            p2[trial_index] = p2_donor_median

    epsilon = np.finfo(np.float64).eps
    return np.maximum(p1, epsilon), np.maximum(p2, epsilon), complete_baseline


def _validate_channel_indices(
    channels: Sequence[int], signal_channel_count: int
) -> tuple[int, ...]:
    indices = tuple(int(channel) for channel in channels)
    if not indices:
        raise ValueError("channels가 비어 있습니다.")
    if len(set(indices)) != len(indices):
        raise ValueError("channels에 중복된 채널이 있습니다.")
    if min(indices) < 0 or max(indices) >= signal_channel_count:
        raise ValueError("channels에 신호 배열 범위 밖의 채널이 있습니다.")
    return indices


def _sequential_detection(
    envelope: np.ndarray,
    thresholds: np.ndarray,
    *,
    channels: tuple[int, ...],
    sustained_samples: int,
    tolerance_samples: int,
    search_start: int,
    search_end: int,
    stage: str,
) -> _DetectionResult | None:
    run_starts: dict[int, list[int]] = {}
    for channel in channels:
        relative_runs = _sustained_runs(
            envelope[search_start:search_end, channel] > thresholds[channel],
            sustained_samples,
        )
        run_starts[channel] = [search_start + start for start, _ in relative_runs]

    candidates: list[tuple[int, int, int, int, int, int]] = []
    available_channels = sorted(
        channel for channel, starts in run_starts.items() if starts
    )
    for left_index, left_channel in enumerate(available_channels):
        for right_channel in available_channels[left_index + 1 :]:
            for left_onset in run_starts[left_channel]:
                for right_onset in run_starts[right_channel]:
                    difference = abs(left_onset - right_onset)
                    if difference <= tolerance_samples:
                        candidates.append(
                            (
                                min(left_onset, right_onset),
                                max(left_onset, right_onset),
                                left_channel,
                                right_channel,
                                left_onset,
                                right_onset,
                            )
                        )

    if not candidates:
        return None

    candidates.sort()
    onset, _, left_channel, right_channel, left_onset, right_onset = candidates[0]
    return _DetectionResult(
        onset_sample=onset,
        stage=stage,
        channels=channels,
        contributing_channels=(left_channel, right_channel),
        contributing_channel_onsets={
            left_channel: left_onset,
            right_channel: right_onset,
        },
        onset_difference_samples=abs(left_onset - right_onset),
        activation_candidate_count=len(candidates),
    )


def _simultaneous_detection(
    envelope: np.ndarray,
    thresholds: np.ndarray,
    *,
    channels: tuple[int, ...],
    min_active_channels: int,
    sustained_samples: int,
    search_start: int,
    search_end: int,
    stage: str,
) -> _DetectionResult | None:
    selected_active = envelope[:, channels] > thresholds[list(channels)]
    combined = selected_active.sum(axis=1) >= min_active_channels
    relative_runs = _sustained_runs(
        combined[search_start:search_end], sustained_samples
    )
    if not relative_runs:
        return None

    onset = search_start + relative_runs[0][0]
    active_positions = np.flatnonzero(selected_active[onset])
    contributing = tuple(channels[int(position)] for position in active_positions)
    if len(contributing) > min_active_channels:
        contributing = contributing[:min_active_channels]
    return _DetectionResult(
        onset_sample=onset,
        stage=stage,
        channels=channels,
        contributing_channels=contributing,
        contributing_channel_onsets={channel: onset for channel in contributing},
        onset_difference_samples=0,
        activation_candidate_count=len(relative_runs),
    )


def _detect_stage_chain(
    envelope: np.ndarray,
    p1_thresholds: np.ndarray,
    p2_thresholds: np.ndarray,
    *,
    channels: tuple[int, ...],
    prefix: str,
    min_active_channels: int,
    sustained_samples: int,
    tolerance_samples: int,
    search_start: int,
    search_end: int,
) -> _DetectionResult | None:
    for threshold_name, thresholds in (
        ("p1", p1_thresholds),
        ("p2", p2_thresholds),
    ):
        sequential = _sequential_detection(
            envelope,
            thresholds,
            channels=channels,
            sustained_samples=sustained_samples,
            tolerance_samples=tolerance_samples,
            search_start=search_start,
            search_end=search_end,
            stage=f"{prefix}_{threshold_name}_sequential",
        )
        if sequential is not None:
            return sequential

        simultaneous = _simultaneous_detection(
            envelope,
            thresholds,
            channels=channels,
            min_active_channels=min_active_channels,
            sustained_samples=sustained_samples,
            search_start=search_start,
            search_end=search_end,
            stage=f"{prefix}_{threshold_name}_simultaneous",
        )
        if simultaneous is not None:
            return simultaneous
    return None


def detect_onset_stepwise(
    trial_signal: np.ndarray,
    p1_thresholds: np.ndarray,
    p2_thresholds: np.ndarray,
    *,
    fs: int = FS,
    channels: Sequence[int] = ONSET_CHANNELS,
    fallback_channels: Sequence[int] | None = FALLBACK_ONSET_CHANNELS,
    min_active_channels: int = MIN_ACTIVE_CHANNELS,
    rms_window_ms: int = RMS_WINDOW_MS,
    sustained_ms: int = SUSTAINED_MS,
    sequential_tolerance_ms: int = SEQUENTIAL_TOLERANCE_MS,
    search_start_seconds: float = SEARCH_START_SECONDS,
    search_end_seconds: float = SEARCH_END_SECONDS,
) -> tuple[_DetectionResult, bool]:
    """Ch2-4 우선, 전체 채널 fallback 순서로 한 시행의 onset을 검출한다.

    Returns:
        ``(검출 결과, fallback 시도 여부)``. 모든 단계가 실패하면
        :class:`OnsetDetectionError`를 발생시킨다.
    """
    trial_signal = np.asarray(trial_signal, dtype=np.float64)
    if trial_signal.ndim != 2 or trial_signal.shape[1] < 5:
        raise ValueError("trial_signal은 5개 이상의 채널을 가진 2차원 배열이어야 합니다.")
    if len(trial_signal) != TRIAL_SAMPLES * fs // FS:
        raise ValueError(f"시행 길이는 {TRIAL_SECONDS}초여야 합니다.")
    if not np.isfinite(trial_signal).all():
        raise ValueError("시행 신호에 NaN 또는 무한대가 있습니다.")

    _validate_positive("fs", fs)
    _validate_positive("rms_window_ms", rms_window_ms)
    _validate_positive("sustained_ms", sustained_ms)
    _validate_positive("sequential_tolerance_ms", sequential_tolerance_ms)
    primary = _validate_channel_indices(channels, trial_signal.shape[1])
    fallback = (
        None
        if fallback_channels is None
        else _validate_channel_indices(fallback_channels, trial_signal.shape[1])
    )
    if not 1 <= min_active_channels <= len(primary):
        raise ValueError("min_active_channels가 기본 channels 수의 범위를 벗어났습니다.")
    if fallback is not None and min_active_channels > len(fallback):
        raise ValueError("min_active_channels가 fallback channels 수를 초과합니다.")

    p1_thresholds = np.asarray(p1_thresholds, dtype=np.float64)
    p2_thresholds = np.asarray(p2_thresholds, dtype=np.float64)
    expected_shape = (trial_signal.shape[1],)
    if p1_thresholds.shape != expected_shape or p2_thresholds.shape != expected_shape:
        raise ValueError(f"임계값 배열은 {expected_shape} 모양이어야 합니다.")
    if not np.isfinite(p1_thresholds).all() or not np.isfinite(p2_thresholds).all():
        raise ValueError("임계값에 NaN 또는 무한대가 있습니다.")

    window_samples = max(1, round(rms_window_ms * fs / 1000))
    sustained_samples = max(1, round(sustained_ms * fs / 1000))
    tolerance_samples = max(1, round(sequential_tolerance_ms * fs / 1000))
    search_start = round(search_start_seconds * fs)
    search_end = round(search_end_seconds * fs)
    if not 0 <= search_start < search_end <= len(trial_signal):
        raise ValueError("onset 검색 범위가 시행 길이를 벗어났습니다.")

    envelope = _rolling_rms(trial_signal, window_samples)
    primary_result = _detect_stage_chain(
        envelope,
        p1_thresholds,
        p2_thresholds,
        channels=primary,
        prefix="primary",
        min_active_channels=min_active_channels,
        sustained_samples=sustained_samples,
        tolerance_samples=tolerance_samples,
        search_start=search_start,
        search_end=search_end,
    )
    if primary_result is not None:
        return primary_result, False

    if fallback is not None:
        fallback_result = _detect_stage_chain(
            envelope,
            p1_thresholds,
            p2_thresholds,
            channels=fallback,
            prefix="fallback",
            min_active_channels=min_active_channels,
            sustained_samples=sustained_samples,
            tolerance_samples=tolerance_samples,
            search_start=search_start,
            search_end=search_end,
        )
        if fallback_result is not None:
            return fallback_result, True

    raise OnsetDetectionError(
        "Ch2-4 우선 검출과 전체 채널 fallback에서 onset을 찾지 못했습니다."
    )


def _meaningful_decimal_places(text: str, max_places: int = 12) -> int:
    """부동소수점 꼬리를 제외한 실질 소수 자릿수를 추정한다."""
    value = float(text)
    for places in range(max_places + 1):
        if abs(value - round(value, places)) <= 5e-13:
            return places
    return max_places


def _median_integer(values: Iterable[int]) -> int:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("중앙값을 계산할 값이 없습니다.")
    return ordered[len(ordered) // 2]


def _format_mean(value: float, donor_text: Sequence[str]) -> str:
    places = _median_integer(_meaningful_decimal_places(text) for text in donor_text)
    rounded = 0.0 if abs(value) < 0.5 * 10 ** (-places) else value
    return f"{rounded:.{places}f}"


def _time_precision(time_text: Sequence[str]) -> int:
    sample = time_text[: min(len(time_text), 1000)]
    return max(
        0,
        min(
            9,
            max(
                (
                    len(value.lower().split("e")[0].split(".", 1)[1])
                    if "." in value.lower().split("e")[0]
                    else 0
                )
                for value in sample
            ),
        ),
    )


def _format_time(value: Decimal, places: int) -> str:
    text = f"{value:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def preprocess_time_axis(
    file_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    donor_trials: Sequence[int] | None = None,
    save_report: bool = True,
    fs: int = FS,
    channels: Sequence[int] = ONSET_CHANNELS,
    fallback_channels: Sequence[int] | None = FALLBACK_ONSET_CHANNELS,
    min_active_channels: int = MIN_ACTIVE_CHANNELS,
    rms_window_ms: int = RMS_WINDOW_MS,
    baseline_end_seconds: float = BASELINE_END_SECONDS,
    threshold_std: float = THRESHOLD_STD,
    robust_mad_scale: float = ROBUST_MAD_SCALE,
    sustained_ms: int = SUSTAINED_MS,
    sequential_tolerance_ms: int = SEQUENTIAL_TOLERANCE_MS,
    search_start_seconds: float = SEARCH_START_SECONDS,
    search_end_seconds: float = SEARCH_END_SECONDS,
    strict_onset: bool = False,
) -> PreprocessReport:
    """CSV의 모든 시행 onset을 2초로 정렬하여 새 파일로 저장한다.

    ``donor_trials``는 평균 보완에 사용할 1부터 시작하는 시행 번호다.
    학습/검증/시험 분할이 존재하면 학습 시행 번호만 전달한다. 후보 위치를
    계산하는 기준 mark뿐 아니라 실제 원본 샘플도 지정된 시행의 5초 범위에
    속해야 donor로 인정하므로, 인접한 비-donor 시행의 샘플은 사용하지 않는다.
    지정하지 않으면 같은 파일의 다른 모든 시행을 사용한다.

    ``baseline_end_seconds``는 기존 호출 호환을 위해 이름을 유지하지만,
    단계형 검출에서는 mark 직전 baseline의 길이를 의미한다.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    if file_path.suffix.lower() != ".csv":
        raise ValueError("입력 파일은 CSV여야 합니다.")
    _validate_positive("fs", fs)

    loaded = _load_csv(file_path)
    mark_indices = loaded.mark_indices
    trial_count = len(mark_indices)
    expected_trial_samples = round(TRIAL_SECONDS * fs)
    target_onset_sample = round(TARGET_ONSET_SECONDS * fs)

    mark_intervals = np.diff(mark_indices).astype(int).tolist()
    irregular_intervals = [
        interval for interval in mark_intervals if interval != expected_trial_samples
    ]
    if irregular_intervals:
        warnings.warn(
            f"{file_path.name}: mark 간격 중 {expected_trial_samples}샘플이 아닌 값이 "
            f"{len(irregular_intervals)}개 있습니다. 각 mark를 독립 시행 시작점으로 사용합니다.",
            RuntimeWarning,
            stacklevel=2,
        )

    if donor_trials is None:
        donor_indices = list(range(trial_count))
    else:
        donor_indices = sorted({int(number) - 1 for number in donor_trials})
        if not donor_indices:
            raise ValueError("donor_trials가 비어 있습니다.")
        if donor_indices[0] < 0 or donor_indices[-1] >= trial_count:
            raise ValueError(f"donor_trials는 1~{trial_count} 범위여야 합니다.")

    donor_sample_owner = np.full(len(loaded.signal_numbers), -1, dtype=np.int64)
    for donor_index in donor_indices:
        donor_start = int(mark_indices[donor_index])
        donor_end = min(donor_start + expected_trial_samples, len(donor_sample_owner))
        donor_sample_owner[donor_start:donor_end] = donor_index

    onset_parameters = OnsetParameters(
        channels=tuple(int(channel) + 1 for channel in channels),
        min_active_channels=min_active_channels,
        rms_window_ms=rms_window_ms,
        baseline_end_seconds=baseline_end_seconds,
        threshold_std=threshold_std,
        robust_mad_scale=robust_mad_scale,
        sustained_ms=sustained_ms,
        sequential_tolerance_ms=sequential_tolerance_ms,
        search_start_seconds=search_start_seconds,
        search_end_seconds=search_end_seconds,
        fallback_channels=(
            ()
            if fallback_channels is None
            else tuple(int(channel) + 1 for channel in fallback_channels)
        ),
        stage_order=(
            "primary_p1_sequential",
            "primary_p1_simultaneous",
            "primary_p2_sequential",
            "primary_p2_simultaneous",
        )
        + (
            ()
            if fallback_channels is None
            else (
                "fallback_p1_sequential",
                "fallback_p1_simultaneous",
                "fallback_p2_sequential",
                "fallback_p2_simultaneous",
            )
        ),
    )

    p1_thresholds, p2_thresholds, complete_baseline = (
        _thresholds_from_mark_baselines(
            loaded.signal_numbers,
            mark_indices,
            fs=fs,
            rms_window_ms=rms_window_ms,
            baseline_seconds=baseline_end_seconds,
            threshold_std=threshold_std,
            robust_mad_scale=robust_mad_scale,
        )
    )

    aligned_numbers = np.full(
        (trial_count, expected_trial_samples, 5), np.nan, dtype=np.float64
    )
    aligned_text = np.empty(
        (trial_count, expected_trial_samples, 5), dtype=object
    )
    aligned_text.fill(None)
    reports: list[TrialAlignmentReport] = []
    failed_trial_numbers: list[int] = []

    for trial_index, mark_index_value in enumerate(mark_indices):
        mark_index = int(mark_index_value)
        trial_end = mark_index + expected_trial_samples
        if trial_end > len(loaded.signal_numbers):
            raise ValueError(
                f"{trial_index + 1}번 시행의 mark 뒤에 "
                f"{expected_trial_samples}샘플이 없습니다."
            )

        detection_failed = False
        fallback_attempted = False
        detection_result: _DetectionResult | None = None
        try:
            detection_result, fallback_attempted = detect_onset_stepwise(
                loaded.signal_numbers[mark_index:trial_end],
                p1_thresholds[trial_index],
                p2_thresholds[trial_index],
                fs=fs,
                channels=channels,
                fallback_channels=fallback_channels,
                min_active_channels=min_active_channels,
                rms_window_ms=rms_window_ms,
                sustained_ms=sustained_ms,
                sequential_tolerance_ms=sequential_tolerance_ms,
                search_start_seconds=search_start_seconds,
                search_end_seconds=search_end_seconds,
            )
        except OnsetDetectionError as exc:
            fallback_attempted = fallback_channels is not None
            if strict_onset:
                raise OnsetDetectionError(
                    f"{file_path.name}의 {trial_index + 1}번 시행: {exc}"
                ) from exc
            detection_failed = True
            failed_trial_numbers.append(trial_index + 1)

        if detection_result is None:
            onset_sample: int | None = None
            candidate_count = 0
            preactivation: bool | None = None
            shift_samples = 0
            detection_stage = "failed_manual_review"
            detection_channels: list[int] = []
            contributing_channels: list[int] = []
            contributing_channel_onsets: dict[str, int] = {}
            onset_difference: int | None = None
            fallback_used = False
            alignment_applied = False
            qc_flags = ["onset_detection_failed", "manual_review"]
        else:
            onset_sample = detection_result.onset_sample
            candidate_count = detection_result.activation_candidate_count
            preactivation = None
            shift_samples = target_onset_sample - onset_sample
            detection_stage = detection_result.stage
            detection_channels = [channel + 1 for channel in detection_result.channels]
            contributing_channels = [
                channel + 1 for channel in detection_result.contributing_channels
            ]
            contributing_channel_onsets = {
                f"Ch{channel + 1}": sample
                for channel, sample in detection_result.contributing_channel_onsets.items()
            }
            onset_difference = detection_result.onset_difference_samples
            fallback_used = detection_result.stage.startswith("fallback_")
            alignment_applied = True
            qc_flags = []
            if fallback_used:
                qc_flags.append("fallback_all_channels")
            if onset_sample < round(0.1 * fs):
                qc_flags.append("early_onset_under_100ms")

        if not complete_baseline[trial_index]:
            qc_flags.append("baseline_threshold_imputed")

        source_start = mark_index - shift_samples
        source_offsets = np.arange(expected_trial_samples) - shift_samples
        valid = (source_offsets >= 0) & (source_offsets < expected_trial_samples)
        source_indices = mark_index + source_offsets

        valid_source = source_indices[valid]
        aligned_numbers[trial_index, valid] = loaded.signal_numbers[valid_source]
        aligned_text[trial_index, valid] = loaded.signal_text[valid_source]

        valid_positions = np.flatnonzero(valid)
        if len(valid_positions):
            filled_prefix = int(valid_positions[0])
            filled_suffix = int(expected_trial_samples - valid_positions[-1] - 1)
        else:
            filled_prefix = expected_trial_samples
            filled_suffix = 0

        reports.append(
            TrialAlignmentReport(
                trial_number=trial_index + 1,
                mark_index=mark_index,
                detected_onset_sample=onset_sample,
                detected_onset_seconds=(
                    None if onset_sample is None else round(onset_sample / fs, 6)
                ),
                shift_samples=shift_samples,
                source_start_index=source_start,
                source_end_index_exclusive=source_start + expected_trial_samples,
                filled_prefix_samples=filled_prefix,
                filled_suffix_samples=filled_suffix,
                activation_candidate_count=candidate_count,
                preactivation_detected=preactivation,
                onset_detection_failed=detection_failed,
                onset_detection_stage=detection_stage,
                detection_channels=detection_channels,
                contributing_channels=contributing_channels,
                contributing_channel_onsets=contributing_channel_onsets,
                contributing_onset_difference_samples=onset_difference,
                fallback_attempted=fallback_attempted,
                fallback_used=fallback_used,
                alignment_applied=alignment_applied,
                qc_flags=qc_flags,
            )
        )

    if failed_trial_numbers:
        warnings.warn(
            f"{file_path.name}: {len(failed_trial_numbers)}개 시행에서 전체 검출 단계를 "
            "통과하지 못해 해당 시행을 이동하지 않았습니다. 시행 번호는 처리 보고서의 "
            "onset_detection_failed와 qc_flags에서 확인할 수 있습니다.",
            RuntimeWarning,
            stacklevel=2,
        )

    missing_positions = np.argwhere(np.isnan(aligned_numbers))
    for trial_index, relative_index, channel_index in missing_positions:
        source_offset = int(relative_index) - reports[trial_index].shift_samples
        donor_source_indices = [
            (donor_index, int(mark_indices[donor_index]) + source_offset)
            for donor_index in donor_indices
            if donor_index != trial_index
            and 0
            <= int(mark_indices[donor_index]) + source_offset
            < len(loaded.signal_numbers)
            and donor_sample_owner[
                int(mark_indices[donor_index]) + source_offset
            ]
            not in (-1, trial_index)
        ]
        if not donor_source_indices:
            raise ValueError(
                f"{trial_index + 1}번 시행 {relative_index}샘플 Ch{channel_index + 1}의 "
                f"부족 구간(source offset {source_offset})을 채울 다른 시행의 "
                "원본 donor가 없습니다."
            )

        donor_values = np.asarray(
            [
                loaded.signal_numbers[source_index, channel_index]
                for _, source_index in donor_source_indices
            ],
            dtype=np.float64,
        )
        donor_text = [
            str(loaded.signal_text[source_index, channel_index])
            for _, source_index in donor_source_indices
        ]
        mean_value = float(donor_values.mean())
        aligned_numbers[trial_index, relative_index, channel_index] = mean_value
        aligned_text[trial_index, relative_index, channel_index] = _format_mean(
            mean_value, donor_text
        )

    if np.isnan(aligned_numbers).any() or any(value is None for value in aligned_text.flat):
        raise RuntimeError("부족 구간 보완 후에도 비어 있는 신호값이 있습니다.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{file_path.stem}_pre.csv"
    report_path = output_dir / f"{file_path.stem}_pre_report.json" if save_report else None

    time_places = _time_precision(loaded.time_text)
    first_time = Decimal(loaded.time_text[int(mark_indices[0])])
    time_step = Decimal(1) / Decimal(fs)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(REQUIRED_COLUMNS)
        output_index = 0
        for trial_index in range(trial_count):
            for relative_index in range(expected_trial_samples):
                time_value = first_time + time_step * output_index
                writer.writerow(
                    list(aligned_text[trial_index, relative_index])
                    + [
                        _format_time(time_value, time_places),
                        (
                            "1"
                            if reports[trial_index].alignment_applied
                            and relative_index == target_onset_sample
                            else "0"
                        ),
                        "1" if relative_index == 0 else "0",
                    ]
                )
                output_index += 1

    report = PreprocessReport(
        input_path=str(file_path.resolve()),
        output_path=str(output_path.resolve()),
        report_path=str(report_path.resolve()) if report_path is not None else None,
        sampling_rate_hz=fs,
        trial_samples=expected_trial_samples,
        target_onset_sample=target_onset_sample,
        trial_count=trial_count,
        input_mark_intervals=mark_intervals,
        irregular_mark_intervals=irregular_intervals,
        donor_trials=[index + 1 for index in donor_indices],
        processing_rule_version=PROCESSING_RULE_VERSION,
        onset_detection_policy=(
            "primary_"
            + "_".join(str(channel + 1) for channel in channels)
            + "_stepwise"
            + (
                "_no_fallback"
                if fallback_channels is None
                else "_then_fallback_"
                + "_".join(str(channel + 1) for channel in fallback_channels)
                + "_stepwise"
            )
        ),
        onset_parameters=onset_parameters,
        trials=reports,
    )

    if report_path is not None:
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(report), handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    return report


def preprocess_files(
    file_paths: Iterable[str | Path],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    **kwargs: object,
) -> list[PreprocessReport]:
    """여러 CSV를 각각 독립적으로 전처리한다."""
    return [
        preprocess_time_axis(path, output_dir=output_dir, **kwargs)
        for path in file_paths
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "sEMG CSV의 시행별 근활성 onset을 2초로 정렬합니다. "
            "입력 경로를 생략하면 프로젝트의 data/data/*.csv를 모두 처리합니다."
        )
    )
    parser.add_argument(
        "input_paths",
        nargs="*",
        help=(
            "전처리할 원본 CSV 경로. 생략하면 data/data의 모든 CSV를 사용합니다."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="출력 경로(기본값: 프로젝트의 data/data_pre)",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="처리 내역 JSON을 저장하지 않습니다.",
    )
    parser.add_argument(
        "--strict-onset",
        action="store_true",
        help="기본·전체채널 검출 단계가 모두 실패하면 오류로 중단합니다.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    input_paths = [Path(path) for path in args.input_paths]
    if not input_paths:
        input_paths = sorted(DEFAULT_INPUT_DIR.glob("*.csv"))
        if not input_paths:
            parser.error(f"자동 입력 경로에 CSV 파일이 없습니다: {DEFAULT_INPUT_DIR}")
        print(
            f"입력 경로가 지정되지 않아 {DEFAULT_INPUT_DIR}의 "
            f"CSV {len(input_paths)}개를 처리합니다."
        )

    reports = preprocess_files(
        input_paths,
        output_dir=args.output_dir,
        save_report=not args.no_report,
        strict_onset=args.strict_onset,
    )
    for report in reports:
        print(report.output_path)
        if report.report_path:
            print(report.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
