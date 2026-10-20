"""전처리된 특징 데이터로 여러 머신러닝 분류기를 학습하고 평가한다."""

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.npz"
RANDOM_STATE = 42

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
    """특징 데이터와 정수로 변환한 클래스 라벨을 반환한다."""
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


def main():
    Xtr, ytr, Xte, yte, label_encoder = load_dataset()
    print("클래스:", list(label_encoder.classes_))

    for name, model in create_models().items():
        model.fit(Xtr, ytr)
        predictions = model.predict(Xte)
        accuracy = accuracy_score(yte, predictions)
        print(f"{name} 학습 완료 - 테스트 정확도: {accuracy:.4f}")


if __name__ == "__main__":
    main()
