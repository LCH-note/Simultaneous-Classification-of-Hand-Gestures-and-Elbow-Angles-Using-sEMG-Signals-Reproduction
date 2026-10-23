# <center>표면 근전도(sEMG) 신호를 활용한 손동작 및 팔꿈치 각도의 동시 분류 재현</center>

## 1. 코드 설명

본 프로젝트는 8명의 참가자(`a`~`h`)에게서 수집한 5채널 sEMG 신호를 이용해 손동작 4종(`extor`, `fist`, `open`, `pro`)과 팔꿈치 각도 2종(30°, 60°)의 조합인 8개 클래스를 분류한다.

### 사용 데이터 및 전처리 방법

- 샘플링 주파수: 1,000 Hz
- 전체 데이터: 64개 CSV, 1,920개 시행(파일별 30개 시행)
- 시행 구분: `mark=1`을 시작점으로 하는 5초(5,000샘플) 구간
- 신호 채널: ECR, ED, FDS, Biceps, Triceps의 5채널
- 필터: 60 Hz notch filter(Q=30) 적용 후 20~450 Hz 4차 Butterworth band-pass filter 적용
- 필터링 방식: `filtfilt`를 사용한 영위상 필터링
- 분석 구간: 시행 시작 1.5초부터 3.5초까지의 2초 구간(2,000샘플)
- 윈도우: 200 ms(200샘플), 100 ms hop, 50% overlap으로 시행당 19개 생성
- 시간영역 특징: 채널별 RMS, MAV, WL, ZC, SSC를 계산하여 윈도우당 25차원 특징 생성
- 데이터 분할: 각 파일의 시행을 먼저 8:2로 분할하여 학습 1,536개, 테스트 384개 시행 구성
- 표준화: 학습 데이터에만 `StandardScaler`를 적합하고 테스트 데이터에는 변환만 적용
- 클래스 분포: 8개 클래스마다 학습 192개, 테스트 48개 시행으로 균등 구성

시행마다 다른 근활성 시작점을 2초 위치로 맞추는 onset 정렬 코드를 구현했다. Ch2~Ch4 기반 단계형 검출을 우선 적용하고, 실패 시 Ch1~Ch5 전체 채널을 사용하는 fallback과 시행별 QC 보고서를 제공한다. 정렬 효과를 확인하기 위해 데이터 구조, 분할, 특징, 모델 파라미터 및 평가 방법은 같게 유지하고 입력 데이터만 다르게 구성했다.

| 구분 | 데이터셋 | 입력 CSV | 모델 | 5-fold 결과 | 혼동행렬 |
| --- | --- | --- | --- | --- | --- |
| 시간축 미정렬 | `data/dataset_improve.npz` | `data/data` | `data/machine learning/no_time_axis` | `data/five fold/no_time_axis` | `output/figures/no_time_axis` |
| 시간축 정렬 | `data/dataset_improve_time_axis.npz` | `data/data_pre` | `data/machine learning/time_axis` | `data/five fold/time_axis` | `output/figures/time_axis` |

### 사용 모델

- SVM: RBF 커널, `C=1.0`, `gamma=scale`
- Random Forest(RF): 결정 트리 100개, `random_state=42`
- XGBoost(XGB): 결정 트리 400개, 최대 깊이 6, 학습률 0.05

각 모델은 학습 시행의 19개 윈도우를 개별 학습 표본으로 사용한다. 평가할 때는 같은 시행에 속한 19개 윈도우의 확률 또는 결정 점수를 평균하여 시행당 하나의 클래스를 예측한다.

### 학습 및 테스트 방법

최종 hold-out 평가는 학습 1,536개 시행과 테스트 384개 시행을 사용한다. 추가로 전체 1,920개 시행에 대해 `StratifiedKFold` 기반 5-fold 교차검증을 수행하며, 각 fold의 학습 데이터로만 표준화 통계를 계산한다. 평가지표는 Accuracy, macro Precision, macro Recall, macro F1-score이다.

### 실행 방법

프로젝트 루트에서 다음 순서로 실행한다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.preprocess.dataset_maker
.\.venv\Scripts\python.exe -m src.machine_learning
.\.venv\Scripts\python.exe -m src.hold_out
.\.venv\Scripts\python.exe -m src.confusion_matrix
.\.venv\Scripts\python.exe -m src.five_fold
```

onset 정렬 결과와 QC 보고서가 필요하면 다음 명령을 별도로 실행한다.

```powershell
.\.venv\Scripts\python.exe -m src.preprocess.data_time_axis
```

정렬 데이터 비교에서는 `machine_learning.py`, `hold_out.py`, `confusion_matrix.py`, `five_fold.py`의 학습·평가 로직은 수정하지 않고 `DATASET_PATH`, 모델 경로와 출력 경로만 위의 데이터 구분표에 맞게 사용했다. 비교 실행 후 네 평가 파일의 기본 상수는 기존 미정렬 경로로 유지했다.

### 코드 설명

| 파일                                | 설명                                                                           |
| ----------------------------------- | ------------------------------------------------------------------------------ |
| `src/preprocess/data_time_axis.py`  | 시행별 근활성 onset 검출, 2초 기준 시간축 정렬, 빈 구간 보완 및 QC 보고서 생성 |
| `src/preprocess/data_preprocess.py` | notch/band-pass 필터링, 분석 구간 절단, 윈도우 생성 및 시간영역 특징 추출      |
| `src/preprocess/dataset_maker.py`   | 시행 단위 학습·테스트 분할, 특징 데이터셋 생성 및 표준화                       |
| `src/machine_learning.py`           | SVM, RF, XGB 학습과 모델·라벨 인코더 저장                                      |
| `src/hold_out.py`                   | 384개 테스트 시행의 Accuracy, Precision, Recall, F1-score 평가                 |
| `src/five_fold.py`                  | 시행 단위 5-fold 교차검증, 대응표본 t-검정 및 실행시간 비교                    |
| `src/confusion_matrix.py`           | 모델별 8클래스 혼동행렬 생성 및 주요 오분류 출력                               |
| `src/drowingSignal.py`              | 단일 시행의 5채널 신호 시각화                                                  |
| `src/drowingSigmalDual.py`          | 30°와 60° 신호 비교 시각화                                                     |

## 2. 모델 성능 비교

미정렬 결과는 2026년 10월 23일 재실행값이며, 정렬 결과는 `dataset_improve_time_axis.npz`로 같은 날 동일한 모델과 평가 코드를 실행한 값이다.

### Hold-out 테스트 결과

| Model | 데이터 | Accuracy | Precision | Recall | F1-score |
| --- | --- | ---: | ---: | ---: | ---: |
| SVM | 미정렬 | 82.03% | 83.12% | 82.03% | 81.91% |
| SVM | 정렬 | **88.02%** | **89.25%** | **88.02%** | **88.01%** |
| Random Forest | 미정렬 | 97.14% | 97.22% | 97.14% | 97.13% |
| Random Forest | 정렬 | **97.40%** | **97.43%** | **97.40%** | **97.39%** |
| XGBoost | 미정렬 | **97.40%** | **97.46%** | **97.40%** | **97.40%** |
| XGBoost | 정렬 | 96.35% | 96.37% | 96.35% | 96.36% |

### 5-fold 교차검증 결과

| Model | 미정렬 Accuracy | 정렬 Accuracy | 정렬에 따른 변화 |
| --- | ---: | ---: | ---: |
| SVM | 76.77 ± 1.50% | **85.47 ± 2.20%** | **+8.70%p** |
| Random Forest | 97.40 ± 0.44% | **97.50 ± 1.25%** | **+0.10%p** |
| XGBoost | **97.03 ± 0.54%** | 96.67 ± 0.98% | -0.36%p |

왼쪽은 시간축 미정렬, 오른쪽은 시간축 정렬 결과이다.

<p align="center">
  <img src="data/five%20fold/no_time_axis/five_fold_comparison.png" width="48%" alt="미정렬 5-fold 정확도">
  <img src="data/five%20fold/time_axis/five_fold_comparison.png" width="48%" alt="정렬 5-fold 정확도">
</p>

### 성능 분석

시간축 정렬은 SVM의 hold-out 정확도를 5.99%p, 5-fold 평균 정확도를 8.70%p 높여 가장 뚜렷한 개선을 보였다. Random Forest는 hold-out에서 0.26%p, 5-fold에서 0.10%p 향상되어 두 조건 모두 약 97%의 안정적인 성능을 유지했다. 반면 XGBoost는 hold-out에서 1.05%p, 5-fold에서 0.36%p 낮아져 시간축 정렬이 모든 모델에 일관된 이점을 주지는 않았다.

### 통계 검정 결과

각 데이터 조건에서 동일한 5개 fold의 모델별 정확도를 이용해 대응표본 t-검정을 수행했다.

| 비교 모델 | 미정렬 t | 미정렬 p | 미정렬 판정 | 정렬 t | 정렬 p | 정렬 판정 |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| SVM vs RF | -25.3773 | 0.000014 | 유의함 | -11.9129 | 0.000284 | 유의함 |
| SVM vs XGB | -24.4321 | 0.000017 | 유의함 | -9.9437 | 0.000574 | 유의함 |
| RF vs XGB | 5.7155 | 0.004636 | 유의함 | 2.1875 | 0.093957 | 유의하지 않음 |

<p align="center">
  <img src="data/five%20fold/no_time_axis/paired_ttest_comparison.png" width="48%" alt="미정렬 대응표본 t-검정">
  <img src="data/five%20fold/time_axis/paired_ttest_comparison.png" width="48%" alt="정렬 대응표본 t-검정">
</p>

SVM은 두 데이터 조건 모두 RF와 XGB보다 유의하게 낮았다. 미정렬 데이터에서는 RF와 XGB의 차이도 유의했지만, 정렬 데이터에서는 두 모델의 차이가 유의하지 않았다. 단, 검정 표본이 5개 fold로 제한되므로 통계 결과는 보조 지표로 해석한다.

### 계산 비용 결과

학습 시간은 hold-out 학습 데이터 전체로 한 번 학습한 시간이며, 추론 시간은 시행 1개(19개 윈도우)의 예측을 100회 반복한 평균이다.

| Model | 미정렬 학습 | 정렬 학습 | 미정렬 추론 | 정렬 추론 |
| --- | ---: | ---: | ---: | ---: |
| SVM | 18.931초 | 18.634초 | 22.437 ms | 22.573 ms |
| Random Forest | 1.062초 | 1.260초 | 16.224 ms | 16.650 ms |
| XGBoost | 3.635초 | 4.772초 | 2.265 ms | 2.478 ms |

<p align="center">
  <img src="data/five%20fold/no_time_axis/model_timing_comparison.png" width="48%" alt="미정렬 모델 실행시간">
  <img src="data/five%20fold/time_axis/model_timing_comparison.png" width="48%" alt="정렬 모델 실행시간">
</p>

두 조건 모두 Random Forest의 학습이 가장 빨랐고 XGBoost의 시행당 추론이 가장 빨랐다. 실행시간은 운영체제 부하와 하드웨어 상태에 영향을 받으므로 정렬 여부에 따른 작은 차이보다는 모델 간 상대적 계산 비용을 비교하는 참고값으로 사용한다.

## 3. Confusion Matrix 분석

각 그림의 왼쪽은 시간축 미정렬, 오른쪽은 시간축 정렬 결과이며 모든 혼동행렬의 합계는 384개이다.

### SVM

<p align="center">
  <img src="output/figures/no_time_axis/confusion8_svm.png" width="48%" alt="미정렬 SVM 혼동행렬">
  <img src="output/figures/time_axis/confusion8_svm.png" width="48%" alt="정렬 SVM 혼동행렬">
</p>

- 미정렬: `60 extor`, `60 pro`가 각 93.75%로 가장 높고 `30 pro`가 70.83%로 가장 낮았다. 전체 오분류는 69건이었다.
- 정렬: `60 pro`가 100%로 가장 높고 `60 fist`가 75.00%로 가장 낮았다. 전체 오분류는 46건으로 23건 감소했다.
- 주요 변화: 정렬 후 `30 pro` 성능은 70.83%에서 85.42%로 개선됐지만 `60 fist`는 79.17%에서 75.00%로 낮아졌다.

### Random Forest

<p align="center">
  <img src="output/figures/no_time_axis/confusion8_rf.png" width="48%" alt="미정렬 Random Forest 혼동행렬">
  <img src="output/figures/time_axis/confusion8_rf.png" width="48%" alt="정렬 Random Forest 혼동행렬">
</p>

- 미정렬: `30 extor`, `30 open`, `60 extor`를 100% 분류했으며 전체 오분류는 11건이었다.
- 정렬: `30 pro`, `60 extor`를 100% 분류했으며 전체 오분류는 10건이었다.
- 주요 변화: 두 조건 모두 가장 큰 오류는 `60 open → 30 open` 3건으로, 정렬 후에도 동일 동작의 각도 구분이 주요 오분류로 남았다.

### XGBoost

<p align="center">
  <img src="output/figures/no_time_axis/confusion8_xgb.png" width="48%" alt="미정렬 XGBoost 혼동행렬">
  <img src="output/figures/time_axis/confusion8_xgb.png" width="48%" alt="정렬 XGBoost 혼동행렬">
</p>

- 미정렬: `30 extor`, `60 extor`, `60 pro`를 100% 분류했으며 전체 오분류는 10건이었다.
- 정렬: `30 pro`, `60 extor`, `60 pro`가 각 97.92%로 가장 높고 전체 오분류는 14건이었다.
- 주요 변화: 정렬 후 오류가 4건 증가했으며 `extor`, `fist`, `open`에서 주로 30°와 60°를 서로 혼동했다.

## 4. 최종 결과

- 시간축 정렬의 효과: SVM에서 가장 큰 개선이 나타났고 RF는 유사한 성능을 유지했으며 XGB는 소폭 하락했다.
- 미정렬 최고 성능: hold-out은 XGBoost 97.40%, 5-fold 평균은 Random Forest 97.40%였다.
- 정렬 최고 성능: hold-out과 5-fold 평균 모두 Random Forest가 각각 97.40%, 97.50%로 가장 높았다.
- 통계 결과: 정렬 후에도 SVM과 두 트리 모델의 차이는 유의했지만 RF와 XGB의 차이는 유의하지 않았다.
- 남은 확인 사항: 원 논문의 정확한 2초 분석 구간과 19개 윈도우 결합 방식이 명시되지 않아 완전한 수치 재현에는 한계가 있다.

### 전체적인 실험 결과 및 느낀 점

19개 윈도우의 점수를 시행 단위로 결합한 결과 RF와 XGB가 97% 이상의 높은 분류 성능을 보였다. 시행 단위 선분할과 학습 데이터 기반 표준화로 겹치는 윈도우에 의한 직접적인 학습·테스트 누수를 방지했다. 현재 결과전에 윈도우 단위 분할로 데이터셋을 구성하였을때 62
