import os
import glob
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


FS = 1000
TRIAL_NUMBER = 1  # 그릴 시행 번호: 1~30
OUTPUT_DIR = Path('outputs/figures')

files = sorted(glob.glob('data/**/*.csv', recursive=True))
file_path = files[0]
data = np.loadtxt(file_path, delimiter=',', skiprows=1)

# 열 7은 mark 열이다. mark=1인 행을 각 시행의 시작점으로 사용한다.
mark_indices = np.flatnonzero(data[:, 7] == 1)
available_trials = len(mark_indices)

if available_trials == 0:
    raise ValueError(f'mark=1인 행이 없습니다: {file_path}')

if not 1 <= TRIAL_NUMBER <= available_trials:
    raise ValueError(
        f'TRIAL_NUMBER는 1~{available_trials} 사이여야 합니다.'
    )

trial_index = TRIAL_NUMBER - 1
start = int(mark_indices[trial_index])

# 다음 mark 직전까지를 현재 시행으로 사용한다.
# 마지막 시행은 다음 mark가 없으므로 최대 5,000샘플을 사용한다.
if trial_index + 1 < available_trials:
    end = int(mark_indices[trial_index + 1])
else:
    end = min(start + FS * 5, len(data))

# 실제 sEMG 열인 bandpass 1~5만 선택한다.
x = data[start:end, :5]
t = np.arange(len(x)) / FS
names = ['ECR','ED','FDS','Biceps','Triceps']

fig, ax = plt.subplots(5, 1, figsize=(10, 7), sharex=True)
for ch in range(5):
    ax[ch].plot(t, x[:, ch], lw=0.5)
    ax[ch].set_ylabel(names[ch], fontsize=8)
    ax[ch].axvline(1.0, color='r', ls='--')
    ax[ch].axvline(2.0, color='r', ls='--')
ax[4].set_xlabel('time (s)')
fig.suptitle(
    f'{os.path.basename(file_path)} - trial {TRIAL_NUMBER} '
    f'({len(x)} samples)'
)
plt.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / 'ch5_example.png'
plt.savefig(output_path, dpi=120)
plt.close(fig)

print('파일:', file_path)
print('시행 번호:', TRIAL_NUMBER)
print('mark 시작 행:', start)
print('다음 mark/종료 행:', end)
print('시행 배열 모양:', x.shape)
print('저장 파일:', output_path)
