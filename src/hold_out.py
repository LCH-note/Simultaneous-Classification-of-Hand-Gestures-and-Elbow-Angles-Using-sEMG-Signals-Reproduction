"""저장된 모델의 시행 단위 hold-out 성능을 평가한다."""

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "dataset_improve.npz"
MODEL_DIR = PROJECT_ROOT / "data" / "machine learning"

MODEL_FILENAMES = {
    "SVM": "svm_model.joblib",
    "RF": "rf_model.joblib",
    "XGB": "xgb_model.joblib",
}
LABEL_ENCODER_FILENAME = "label_encoder.joblib"


def load_test_dataset(label_encoder):
    """3차원 테스트 특징과 정수 라벨을 불러온다."""
    with np.load(DATASET_PATH, allow_pickle=True) as data:
        Xte = data["Xte"]
        yte = label_encoder.transform(data["yte"])
    return Xte, yte


def load_models():
    """저장된 모델을 이름별로 불러온다."""
    return {
        name: load(MODEL_DIR / filename)
        for name, filename in MODEL_FILENAMES.items()
    }


def aggregate_trial_predictions(model, Xte):
    """19개 윈도우의 점수를 평균해 시행별 클래스를 예측한다."""
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


def evaluate_models(models, Xte, yte):
    """모델별 시행 단위 평가 지표를 데이터프레임으로 반환한다."""
    rows = []
    for name, model in models.items():
        predictions = aggregate_trial_predictions(model, Xte)
        rows.append([
            name,
            accuracy_score(yte, predictions),
            precision_score(yte, predictions, average="macro"),
            recall_score(yte, predictions, average="macro"),
            f1_score(yte, predictions, average="macro"),
        ])

    return pd.DataFrame(
        rows,
        columns=["model", "acc", "precision", "recall", "f1"],
    )


def main():
    label_encoder = load(MODEL_DIR / LABEL_ENCODER_FILENAME)
    Xte, yte = load_test_dataset(label_encoder)
    models = load_models()
    results = evaluate_models(models, Xte, yte)

    print("테스트 시행 데이터:", Xte.shape)
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
