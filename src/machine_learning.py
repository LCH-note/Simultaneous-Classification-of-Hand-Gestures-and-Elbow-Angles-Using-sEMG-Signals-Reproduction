"""3차원 시행 특징으로 윈도우 모델을 학습하고 시행 단위로 평가한다."""

from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "dataset_improve.npz"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "data" / "machine learning"
RANDOM_STATE = 42

MODEL_FILENAMES = {
    "SVM": "svm_model.joblib",
    "RF": "rf_model.joblib",
    "XGB": "xgb_model.joblib",
}
LABEL_ENCODER_FILENAME = "label_encoder.joblib"

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "decision_function_shape": "ovr",
}
RF_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
}


def load_dataset():
    """3차원 특징과 정수로 변환한 시행 라벨을 반환한다."""
    with np.load(DATASET_PATH, allow_pickle=True) as data:
        Xtr = data["Xtr"]
        ytr = data["ytr"]
        Xte = data["Xte"]
        yte = data["yte"]

    label_encoder = LabelEncoder()
    ytr = label_encoder.fit_transform(ytr)
    yte = label_encoder.transform(yte)
    return Xtr, ytr, Xte, yte, label_encoder


def create_models():
    """비교할 분류기들을 생성한다."""
    return {
        "SVM": SVC(**SVM_PARAMS),
        "RF": RandomForestClassifier(**RF_PARAMS),
        "XGB": XGBClassifier(**XGB_PARAMS),
    }


def flatten_windows(features, labels):
    """시행별 윈도우를 모델 학습용 2차원 배열로 펼친다."""
    trial_count, window_count, feature_count = features.shape
    X = features.reshape(trial_count * window_count, feature_count)
    y = np.repeat(labels, window_count)
    return X, y


def aggregate_trial_predictions(model, Xte):
    """윈도우별 결정 점수를 평균해 시행별 클래스를 예측한다."""
    trial_count, window_count, feature_count = Xte.shape
    windows = Xte.reshape(trial_count * window_count, feature_count)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(windows)
    else:
        scores = model.decision_function(windows)

    class_count = scores.shape[1]
    trial_scores = scores.reshape(
        trial_count,
        window_count,
        class_count,
    ).mean(axis=1)
    return model.classes_[np.argmax(trial_scores, axis=1)]


def save_model(model, name):
    """학습된 모델을 지정된 경로에 저장한다."""
    model_path = MODEL_OUTPUT_DIR / MODEL_FILENAMES[name]
    dump(model, model_path)
    return model_path


def main():
    Xtr, ytr, Xte, yte, label_encoder = load_dataset()
    Xtr_windows, ytr_windows = flatten_windows(Xtr, ytr)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dump(label_encoder, MODEL_OUTPUT_DIR / LABEL_ENCODER_FILENAME)

    print("클래스:", list(label_encoder.classes_))
    print("학습 시행 데이터:", Xtr.shape)
    print("테스트 시행 데이터:", Xte.shape)

    for name, model in create_models().items():
        model.fit(Xtr_windows, ytr_windows)
        predictions = aggregate_trial_predictions(model, Xte)
        accuracy = accuracy_score(yte, predictions)
        print(f"{name} 학습 완료 - 시행 단위 테스트 정확도: {accuracy:.4f}")
        print("모델 저장:", save_model(model, name))


if __name__ == "__main__":
    main()
