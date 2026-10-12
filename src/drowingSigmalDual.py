import os
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


FS = 1000
PARTICIPANTS = ['b', 'c']
GESTURE = 'extor'
TRIAL_NUMBERS = range(1, 31)  # 1~30번 시행을 모두 생성
DATA_DIR = 'data/data'
OUTPUT_DIR = Path('outputs/figures')

CHANNEL_NAMES = ['ECR', 'ED', 'FDS', 'Biceps', 'Triceps']
DATA_CACHE = {}


def load_mark_trial(file_path, trial_number):
    """mark=1을 기준으로 지정한 한 시행의 sEMG 5채널을 반환한다."""
    if file_path not in DATA_CACHE:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        mark_indices = np.flatnonzero(data[:, 7] == 1)
        DATA_CACHE[file_path] = (data, mark_indices)
    else:
        data, mark_indices = DATA_CACHE[file_path]

    if len(mark_indices) == 0:
        raise ValueError(f'mark=1인 행이 없습니다: {file_path}')

    if not 1 <= trial_number <= len(mark_indices):
        raise ValueError(
            f'{os.path.basename(file_path)}의 TRIAL_NUMBER는 '
            f'1~{len(mark_indices)} 사이여야 합니다.'
        )

    trial_index = trial_number - 1
    start = int(mark_indices[trial_index])

    # 다음 mark 직전까지를 한 시행으로 사용한다.
    # 마지막 시행은 다음 mark가 없으므로 최대 5,000샘플을 사용한다.
    if trial_index + 1 < len(mark_indices):
        end = int(mark_indices[trial_index + 1])
    else:
        end = min(start + FS * 5, len(data))

    signal = data[start:end, :5]
    return signal, start, end


def plot_dual_trial(participant, file_30, file_60, trial_number):
    """같은 시행의 30°와 60° 신호를 나란히 그리고 저장한다."""
    signal_30, start_30, end_30 = load_mark_trial(file_30, trial_number)
    signal_60, start_60, end_60 = load_mark_trial(file_60, trial_number)

    signals = [signal_30, signal_60]
    angles = [30, 60]
    colors = ['tab:blue', 'tab:red']
    fig, axes = plt.subplots(5, 2, figsize=(14, 9), sharex=True)

    for channel in range(5):
        # 같은 채널의 30°와 60°에 동일한 y축 범위를 적용한다.
        combined = np.concatenate(
            [signal_30[:, channel], signal_60[:, channel]]
        )
        y_limit = np.max(np.abs(combined)) * 1.05
        if y_limit == 0:
            y_limit = 1.0

        for column, (angle, signal, color) in enumerate(
            zip(angles, signals, colors)
        ):
            time = np.arange(len(signal)) / FS
            axis = axes[channel, column]
            axis.plot(time, signal[:, channel], color=color, linewidth=0.5)

            # 근활성 구간: 1~2초
            axis.axvline(1.0, color='black', linestyle='--', linewidth=1.0)
            axis.axvline(2.0, color='black', linestyle='--', linewidth=1.0)
            axis.set_ylim(-y_limit, y_limit)
            axis.grid(alpha=0.2)

            if column == 0:
                axis.set_ylabel(CHANNEL_NAMES[channel])
            if channel == 0:
                axis.set_title(f'{angle} degrees')

    axes[-1, 0].set_xlabel('time (s)')
    axes[-1, 1].set_xlabel('time (s)')
    fig.suptitle(
        f'{participant} - {GESTURE} - trial {trial_number}: '
        '30 vs 60 degrees'
    )
    fig.text(
        0.5,
        0.01,
        'Dashed lines: muscle activation interval (1-2 s)',
        ha='center',
    )
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))

    output_path = OUTPUT_DIR / (
        f'ch5_dual_{participant}_{GESTURE}_trial{trial_number:02d}.png'
    )
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(
        f'trial {trial_number:02d}: '
        f'30° {start_30}~{end_30} {signal_30.shape}, '
        f'60° {start_60}~{end_60} {signal_60.shape}'
    )
    return output_path


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_paths = []
for participant in PARTICIPANTS:
    file_30 = os.path.join(DATA_DIR, f'{participant}_30 {GESTURE}.csv')
    file_60 = os.path.join(DATA_DIR, f'{participant}_60 {GESTURE}.csv')
    for trial_number in TRIAL_NUMBERS:
        output_paths.append(
            plot_dual_trial(
                participant, file_30, file_60, trial_number
            )
        )

print(f'생성한 이미지 수: {len(output_paths)}')
print('저장 폴더:', OUTPUT_DIR)
