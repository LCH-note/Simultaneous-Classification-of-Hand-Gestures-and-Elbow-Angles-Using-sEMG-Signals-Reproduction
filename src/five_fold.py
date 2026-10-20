"""세 모델의 시행 단위 5-fold 교차검증 결과를 저장한다."""

import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "dataset_improve.npz"
OUTPUT_DIR = PROJECT_ROOT / "data" / "five fold"
JSON_OUTPUT_PATH = OUTPUT_DIR / "five_fold_results.json"
TEXT_OUTPUT_PATH = OUTPUT_DIR / "five_fold_results.txt"
IMAGE_OUTPUT_PATH = OUTPUT_DIR / "five_fold_comparison.png"

N_SPLITS = 5
RANDOM_STATE = 42
FIGURE_SIZE = (8, 5)
FIGURE_DPI = 120

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


def create_models():
    """교차검증에 사용할 원형 모델을 생성한다."""
    return {
        "SVM": SVC(**SVM_PARAMS),
        "RF": RandomForestClassifier(**RF_PARAMS),
        "XGB": XGBClassifier(**XGB_PARAMS),
    }


def load_dataset():
    """전체 시행 특징을 원래 스케일로 복원하고 라벨을 인코딩한다."""
    with np.load(DATASET_PATH, allow_pickle=True) as data:
        X = np.concatenate([data["Xtr"], data["Xte"]], axis=0)
        labels = np.concatenate([data["ytr"], data["yte"]], axis=0)
        scaler_mean = data["scaler_mean"]
        scaler_scale = data["scaler_scale"]

    X = X * scaler_scale + scaler_mean
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    return X, y, label_encoder


def scale_fold(X_train, X_validation):
    """현재 fold의 학습 윈도우로만 특징을 표준화한다."""
    feature_count = X_train.shape[-1]
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(
        X_train.reshape(-1, feature_count)
    ).reshape(X_train.shape)
    validation_scaled = scaler.transform(
        X_validation.reshape(-1, feature_count)
    ).reshape(X_validation.shape)
    return train_scaled, validation_scaled


def flatten_training_windows(features, labels):
    """시행별 윈도우를 모델 학습용 배열로 펼친다."""
    trial_count, window_count, feature_count = features.shape
    X = features.reshape(trial_count * window_count, feature_count)
    y = np.repeat(labels, window_count)
    return X, y


def aggregate_trial_predictions(model, features):
    """윈도우 점수를 평균해 시행별 클래스를 예측한다."""
    trial_count, window_count, feature_count = features.shape
    windows = features.reshape(trial_count * window_count, feature_count)

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


def run_cross_validation(X, y, models):
    """층화 5-fold 시행 단위 정확도를 계산한다."""
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fold_accuracy = {name: [] for name in models}

    for fold_number, (train_indices, validation_indices) in enumerate(
        splitter.split(X, y),
        start=1,
    ):
        X_train, X_validation = scale_fold(
            X[train_indices],
            X[validation_indices],
        )
        y_train = y[train_indices]
        y_validation = y[validation_indices]
        train_windows, train_labels = flatten_training_windows(
            X_train,
            y_train,
        )

        print(f"Fold {fold_number}/{N_SPLITS}")
        for name, prototype in models.items():
            model = clone(prototype)
            model.fit(train_windows, train_labels)
            predictions = aggregate_trial_predictions(
                model,
                X_validation,
            )
            accuracy = accuracy_score(y_validation, predictions)
            fold_accuracy[name].append(float(accuracy))
            print(f"  {name}: {accuracy * 100:.2f}%")

    return fold_accuracy


def summarize_results(fold_accuracy):
    """모델별 fold 정확도의 평균과 표준편차를 계산한다."""
    return {
        name: {
            "fold_accuracies": values,
            "mean_accuracy": float(np.mean(values)),
            "std_accuracy": float(np.std(values)),
        }
        for name, values in fold_accuracy.items()
    }


def save_json(results, class_names, sample_count):
    """교차검증 설정과 결과를 JSON으로 저장한다."""
    payload = {
        "dataset": str(DATASET_PATH),
        "sample_unit": "trial",
        "sample_count": sample_count,
        "windows_per_trial": 19,
        "class_names": class_names.tolist(),
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "aggregation": "mean window probability or decision score",
        "models": results,
    }
    JSON_OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_text(results):
    """모델별 fold 결과를 읽기 쉬운 텍스트로 저장한다."""
    lines = [
        "5-fold 교차검증 결과",
        f"데이터셋: {DATASET_PATH}",
        "평가 단위: 시행",
        "",
    ]

    for name, result in results.items():
        fold_values = ", ".join(
            f"{value * 100:.2f}%"
            for value in result["fold_accuracies"]
        )
        lines.append(f"{name} fold 정확도: {fold_values}")
        lines.append(
            f"{name}: {result['mean_accuracy'] * 100:.2f} "
            f"± {result['std_accuracy'] * 100:.2f}%"
        )
        lines.append("")

    TEXT_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def save_comparison_image(results):
    """평균 정확도와 표준편차를 막대그래프로 저장한다."""
    names = list(results)
    means = [results[name]["mean_accuracy"] * 100 for name in names]
    standard_deviations = [
        results[name]["std_accuracy"] * 100
        for name in names
    ]

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    bars = axis.bar(
        names,
        means,
        yerr=standard_deviations,
        capsize=6,
        color=["#4C78A8", "#59A14F", "#F28E2B"],
    )
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("5-Fold Cross-Validation Accuracy")
    axis.set_ylim(0, 110)
    axis.grid(axis="y", alpha=0.25)

    for bar, mean, deviation in zip(
        bars,
        means,
        standard_deviations,
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            mean + deviation + 1,
            f"{mean:.2f} ± {deviation:.2f}%",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()
    figure.savefig(
        IMAGE_OUTPUT_PATH,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, label_encoder = load_dataset()
    models = create_models()
    fold_accuracy = run_cross_validation(X, y, models)
    results = summarize_results(fold_accuracy)

    save_json(results, label_encoder.classes_, len(y))
    save_text(results)
    save_comparison_image(results)

    print("최종 결과:")
    for name, result in results.items():
        print(
            f"{name}: {result['mean_accuracy'] * 100:.2f} "
            f"± {result['std_accuracy'] * 100:.2f}%"
        )
    print("JSON 저장:", JSON_OUTPUT_PATH)
    print("텍스트 저장:", TEXT_OUTPUT_PATH)
    print("이미지 저장:", IMAGE_OUTPUT_PATH)


if __name__ == "__main__":
    main()
