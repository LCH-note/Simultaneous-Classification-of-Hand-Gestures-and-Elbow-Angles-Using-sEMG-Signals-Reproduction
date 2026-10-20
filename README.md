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
- 분석 구간: 시행 시작 1.5초부터 2초 구간(2,000샘플)
- 윈도우: 200 ms(200샘플), 100 ms hop, 50% overlap으로 시행당 19개 생성
- 시간영역 특징: 채널별 RMS, MAV, WL, ZC, SSC를 계산하여 윈도우당 25차원 특징 생성
- 데이터 분할: 각 파일의 시행을 먼저 8:2로 분할하여 학습 1,536개, 테스트 384개 시행 구성
- 표준화: 학습 데이터에만 `StandardScaler`를 적합하고 테스트 데이터에는 변환만 적용
- 클래스 분포: 8개 클래스마다 학습 192개, 테스트 48개 시행으로 균등 구성

추가로 시행마다 다른 근활성 시작점을 2초 위치로 맞추는 onset 정렬 코드를 구현했다. Ch2~Ch4 기반 단계형 검출을 우선 적용하고, 실패 시 Ch1~Ch5 전체 채널을 사용하는 fallback과 시행별 QC 보고서를 제공한다. 현재 모델 학습용 `dataset_improve.npz`는 `data/data`의 원본 시행에서 생성되므로 onset 정렬은 별도 전처리 실험으로 관리한다.

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

### 코드 설명

|파일|설명|
|---|---|
|`src/preprocess/data_time_axis.py`|시행별 근활성 onset 검출, 2초 기준 시간축 정렬, 빈 구간 보완 및 QC 보고서 생성|
|`src/preprocess/data_preprocess.py`|notch/band-pass 필터링, 분석 구간 절단, 윈도우 생성 및 시간영역 특징 추출|
|`src/preprocess/dataset_maker.py`|시행 단위 학습·테스트 분할, 특징 데이터셋 생성 및 표준화|
|`src/machine_learning.py`|SVM, RF, XGB 학습과 모델·라벨 인코더 저장|
|`src/hold_out.py`|384개 테스트 시행의 Accuracy, Precision, Recall, F1-score 평가|
|`src/five_fold.py`|시행 단위 5-fold 교차검증, 대응표본 t-검정 및 실행시간 비교|
|`src/confusion_matrix.py`|모델별 8클래스 혼동행렬 생성 및 주요 오분류 출력|
|`src/drowingSignal.py`|단일 시행의 5채널 신호 시각화|
|`src/drowingSigmalDual.py`|30°와 60° 신호 비교 시각화|

## 2. 모델 성능 비교

### Hold-out 테스트 결과

|Model|Accuracy|Precision|Recall|F1-score|
|---|---:|---:|---:|---:|
|SVM|82.03%|83.12%|82.03%|81.91%|
|Random Forest|97.14%|97.22%|97.14%|97.13%|
|XGBoost|97.40%|97.46%|97.40%|97.40%|

### 5-fold 교차검증 결과

|Model|평균 Accuracy|표준편차|
|---|---:|---:|
|SVM|76.77%|1.50%p|
|Random Forest|97.40%|0.44%p|
|XGBoost|97.03%|0.54%p|

![5-fold 교차검증 정확도](data/five%20fold/five_fold_comparison.png)

### 성능 분석

Hold-out에서는 XGBoost가 Accuracy와 macro F1-score 모두 97.40%로 가장 높았으며, Random Forest도 97.14%로 비슷한 성능을 보였다. 5-fold 평균 정확도는 Random Forest가 97.40%로 가장 높고 표준편차도 0.44%p로 가장 작아 fold별 성능이 안정적이었다. SVM은 두 트리 기반 모델보다 낮은 성능을 보였으며 특히 동일 동작의 30°와 60°를 구분하는 과정에서 오분류가 많았다.

5-fold 정확도에 대한 대응표본 t-검정에서는 SVM-RF, SVM-XGB, RF-XGB의 세 비교 모두 유의수준 0.05에서 유의한 차이를 보였다. 다만 fold가 5개뿐이므로 통계 결과는 보조 지표로 해석한다.

## 3. Confusion Matrix 분석

### SVM

![SVM Confusion Matrix](output/figures/confusion8_svm.png)

- 가장 잘 분류된 클래스: `60 extor`, `60 pro`(각 45/48, 93.75%)
- 가장 많이 오분류된 클래스: `30 pro`(34/48 정답, 70.83%)
- 주요 오분류 유형: `30 extor → 60 extor` 8건, `60 open → 30 open` 7건, `30 pro → 60 pro` 5건
- 오분류 분석: 같은 동작 안에서 팔꿈치 각도만 다른 클래스로 예측한 경우가 많아 SVM이 손동작 패턴보다 각도 차이를 구분하는 데 상대적으로 어려움을 보였다.

### Random Forest

![Random Forest Confusion Matrix](output/figures/confusion8_rf.png)

- 가장 잘 분류된 클래스: `30 extor`, `30 open`, `60 extor`(각 48/48, 100%)
- 가장 많이 오분류된 클래스: `60 open`(44/48 정답, 91.67%)
- 주요 오분류 유형: `60 open → 30 open` 3건, `30 fist → 60 fist` 2건, `30 pro → 30 extor` 2건
- 오분류 분석: 전체 오류가 11건으로 적으며, 가장 큰 오류 유형은 동일한 `open` 동작에서 팔꿈치 각도를 혼동한 경우였다.

### XGBoost

![XGBoost Confusion Matrix](output/figures/confusion8_xgb.png)

- 가장 잘 분류된 클래스: `30 extor`, `60 extor`, `60 pro`(각 48/48, 100%)
- 가장 많이 오분류된 클래스: `30 pro`(45/48 정답, 93.75%)
- 주요 오분류 유형: `30 pro → 30 extor` 3건, `30 open → 60 open` 2건, `60 open → 30 open` 2건
- 오분류 분석: 전체 오류가 10건으로 세 모델 중 가장 적었다. 일부 `open` 시행에서는 각도를 혼동했고, `30 pro`에서는 같은 각도의 `extor` 동작과 혼동했다.

모든 모델의 혼동행렬 합계는 테스트 시행 수와 같은 384개이다.

## 4. 최종 결과

- 가장 성능이 좋은 모델: hold-out 기준 XGBoost(Accuracy/F1-score 97.40%), 5-fold 평균 기준 Random Forest(97.40 ± 0.44%)
- 가장 성능이 낮은 모델: SVM(hold-out Accuracy 82.03%, 5-fold 평균 76.77%)
- 주요 오분류 클래스: SVM의 `30 pro`, `60 open`; 트리 모델에서는 `open`의 30°·60° 간 혼동과 `30 pro → 30 extor`가 주로 나타남
- 전체 결과: 19개 윈도우의 점수를 시행 단위로 결합한 결과 RF와 XGB가 97% 이상의 높은 분류 성능을 보였다. 시행 단위 선분할과 학습 데이터 기반 표준화로 겹치는 윈도우에 의한 직접적인 학습·테스트 누수를 방지했다.
- 남은 확인 사항: `src/main.py`를 통한 통합 실행은 아직 구성되지 않았으며, onset 정렬 데이터를 현재 모델 학습 파이프라인에 연결한 실험과 원 논문의 정확한 2초 분석 구간 및 윈도우 결합 방식 확인이 필요하다.
