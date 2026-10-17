"""2주차 실습 단계 1: sEMG 신호 필터링.

60 Hz 전원선 잡음을 노치 필터로 제거한 뒤, 20~450 Hz 대역만
4차 Butterworth 대역통과 필터로 남긴다. 두 필터 모두 ``filtfilt``를
사용해 위상 지연 없이 시간축(axis=0)에 적용한다.
"""

import glob
import os
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

matplotlib.use('Agg')
import matplotlib.pyplot as plt


FS = 1000
NOTCH_FREQUENCY = 60
NOTCH_Q = 30
BANDPASS_LOW = 20
BANDPASS_HIGH = 450
BANDPASS_ORDER = 4
TRIAL_SAMPLES = 5 * FS
CHANNEL_NAMES = ['ECR', 'ED', 'FDS', 'Biceps', 'Triceps']
OUTPUT_PATH = Path('outputs/figures/filter_comparison.png')

START, LEN = 1500, 2000        # 1.5초 지점부터 2초

WIN, HOP = 200, 100


def preprocess(x, fs=FS):
    """sEMG 배열에 60 Hz 노치와 20~450 Hz 대역통과 필터를 적용한다.

    Args:
        x: ``(샘플 수, 채널 수)`` 모양의 sEMG 배열.
        fs: 샘플링 주파수(Hz).

    Returns:
        입력과 같은 모양의 필터링된 배열.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError('x는 (샘플 수, 채널 수) 모양의 2차원 배열이어야 합니다.')
    if x.shape[0] == 0:
        raise ValueError('필터링할 샘플이 없습니다.')
    if not np.isfinite(x).all():
        raise ValueError('입력에 NaN 또는 무한대가 포함되어 있습니다.')

    nyquist = fs / 2
    if not 0 < NOTCH_FREQUENCY < nyquist:
        raise ValueError('노치 주파수는 나이퀴스트 주파수보다 작아야 합니다.')
    if not 0 < BANDPASS_LOW < BANDPASS_HIGH < nyquist:
        raise ValueError('대역통과 범위는 0과 나이퀴스트 주파수 사이여야 합니다.')

    notch_b, notch_a = iirnotch(NOTCH_FREQUENCY, NOTCH_Q, fs=fs)
    notched = filtfilt(notch_b, notch_a, x, axis=0)

    band_b, band_a = butter(
        BANDPASS_ORDER,
        [BANDPASS_LOW / nyquist, BANDPASS_HIGH / nyquist],
        btype='bandpass',
    )
    return filtfilt(band_b, band_a, notched, axis=0)


def load_trial(file_path, trial_number=1, n_samples=TRIAL_SAMPLES):
    """CSV에서 mark=1을 기준으로 지정한 시행의 sEMG 5채널을 읽는다."""
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    mark_indices = np.flatnonzero(data[:, 7] == 1)

    if not 1 <= trial_number <= len(mark_indices):
        raise ValueError(
            f'{os.path.basename(file_path)}의 시행 번호는 '
            f'1~{len(mark_indices)} 사이여야 합니다.'
        )

    start = int(mark_indices[trial_number - 1])
    end = min(start + n_samples, len(data))
    x = data[start:end, :5]
    if len(x) != n_samples:
        raise ValueError(
            f'{trial_number}번 시행의 길이가 {len(x)}샘플입니다. '
            f'{n_samples}샘플이 필요합니다.'
        )
    return x


def plot_filter_comparison(
    x,
    xf,
    fs=FS,
    output_path=OUTPUT_PATH,
    channel_names=CHANNEL_NAMES,
):
    """필터 적용 전후의 5채널 신호를 겹쳐 그리고 PNG로 저장한다.

    Args:
        x: 필터 적용 전 ``(샘플 수, 채널 수)`` 배열.
        xf: 필터 적용 후 배열. ``x``와 모양이 같아야 한다.
        fs: 샘플링 주파수(Hz).
        output_path: 비교 그래프를 저장할 경로.
        channel_names: 그래프에 표시할 채널 이름.

    Returns:
        저장된 그래프의 ``Path`` 객체.
    """
    x = np.asarray(x)
    xf = np.asarray(xf)
    if x.ndim != 2 or xf.ndim != 2:
        raise ValueError('x와 xf는 모두 2차원 배열이어야 합니다.')
    if x.shape != xf.shape:
        raise ValueError('필터 전후 배열의 모양이 같아야 합니다.')
    if x.shape[1] != len(channel_names):
        raise ValueError('채널 수와 channel_names의 길이가 같아야 합니다.')
    if fs <= 0:
        raise ValueError('fs는 양수여야 합니다.')

    time = np.arange(len(x)) / fs
    fig, axes = plt.subplots(
        x.shape[1],
        1,
        figsize=(12, 10),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for channel, axis in enumerate(axes):
        axis.plot(
            time,
            x[:, channel],
            color='tab:red',
            linewidth=0.7,
            alpha=0.7,
            label='Before',
        )
        axis.plot(
            time,
            xf[:, channel],
            color='tab:green',
            linewidth=0.8,
            label='After',
        )
        axis.set_ylabel(channel_names[channel])
        axis.grid(alpha=0.2)

    axes[0].legend(loc='upper right')
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('sEMG Signals Before and After Filtering')
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def cut(x, start=START, n=LEN):
    """배열을 지정한 시작점부터 지정한 길이만큼 잘라 반환한다."""
    return x[start:start+n]

def make_windows(x, win=WIN, hop=HOP):
    """배열을 지정한 윈도우 길이와 홉 길이로 나누어 반환한다."""
    n = (len(x) - win) // hop + 1
    return np.stack([x[i*hop : i*hop+win] for i in range(n)])

def td_features(w, th=0.01):
    """w: (200, 5) -> (25,)"""
    rms = np.sqrt(np.mean(w**2, axis=0))
    mav = np.mean(np.abs(w), axis=0)
    d   = np.diff(w, axis=0)
    wl  = np.sum(np.abs(d), axis=0)
    zc  = np.sum(((w[:-1]*w[1:]) < 0) &
                 (np.abs(d) >= th), axis=0)
    ssc = np.sum((d[:-1]*d[1:]) < 0, axis=0)
    return np.concatenate([rms, mav, wl, zc, ssc])

def main():
    files = sorted(glob.glob('data/**/*.csv', recursive=True))
    if not files:
        raise FileNotFoundError('data 폴더에서 CSV 파일을 찾지 못했습니다.')

    file_path = files[0]
    x = load_trial(file_path)
    xf = preprocess(x)
    graph_path = plot_filter_comparison(x, xf)

    seg = cut(xf)
    print('세그먼트 모양:', seg.shape)   # (2000, 5)

    w = make_windows(seg)
    print('윈도우 배열:', w.shape)      # (19, 200, 5)

    f = td_features(w[0])
    print('특징 벡터:', f.shape)        # (25,)

    # print('파일:', file_path)
    # print('필터 전 배열:', x.shape)
    # print('필터 후 배열:', xf.shape)
    # print('전:', x.std(axis=0).round(4))
    # print('후:', xf.std(axis=0).round(4))
    # print('비교 그래프:', graph_path)


if __name__ == '__main__':
    main()
