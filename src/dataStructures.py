import os
import glob
import numpy as np
from collections import Counter

files = sorted(glob.glob('data/**/*.csv', recursive=True))

def label_of(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    # a_30 extor → 30 extor
    return filename.split('_', 1)[1]

labels = [label_of(f) for f in files]
print(Counter(labels))              # 클래스별 파일 개수

# 첫 파일에서 mark=1인 첫 번째 시행을 찾는다.
data = np.loadtxt(files[0], delimiter=',', skiprows=1)
mark_indices = np.flatnonzero(data[:, 7] == 1)

if len(mark_indices) == 0:
    raise ValueError(f'mark=1인 행이 없습니다: {files[0]}')

# 첫 번째 mark부터 5,000샘플, sEMG 5채널만 선택한다.
start = int(mark_indices[0])
x = data[start:start + 5000, :5]

print('배열 모양:', x.shape)
print('채널 수:', x.shape[1])
