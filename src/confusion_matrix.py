"""저장된 모델별 시행 단위 혼동행렬을 생성한다."""

from pathlib import Path

import matplotlib
import numpy as np
from joblib import load
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "dataset_improve.npz"
MODEL_DIR = PROJECT_ROOT / "data" / "machine learning"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"
FIGURE_SIZE = (8, 7)
FIGURE_DPI = 120

MODEL_FILENAMES = {
    "SVM": "svm_model.joblib",
    "RF": "rf_model.joblib",
    "XGB": "xgb_model.joblib",
}
OUTPUT_FILENAMES = {
    "SVM": "confusion8_svm.png",
    "RF": "confusion8_rf.png",
    "XGB": "confusion8_xgb.png",
}


def load_test_dataset(label_encoder):
    """3차원 테스트 특징과 정수 라벨을 불러온다."""
    with np.load(DATASET_PATH, allow_pickle=True) as data:
        Xte = data["Xte"]
        yte = label_encoder.transform(data["yte"])
    return Xte, yte


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


def plot_confusion_matrix(matrix, class_names, model_name, output_path):
    """혼동행렬의 빈도를 표시하고 이미지로 저장한다."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(f"{model_name} Confusion Matrix")

    threshold = matrix.max() / 2
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = matrix[row, column]
            if value:
                color = "white" if value > threshold else "black"
                axis.text(
                    column,
                    row,
                    value,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def print_structure_summary(matrix, class_names, model_name):
    """클래스별 정답률과 주요 오분류 구조를 출력한다."""
    print(f"[{model_name}] 클래스별 정답률:")
    for index, class_name in enumerate(class_names):
        total = matrix[index].sum()
        recall = matrix[index, index] / total
        print(
            f"  {class_name}: {matrix[index, index]}/{total} "
            f"({recall:.2%})"
        )

    errors = matrix.copy()
    np.fill_diagonal(errors, 0)
    error_pairs = np.argwhere(errors > 0)
    error_pairs = sorted(
        error_pairs,
        key=lambda pair: errors[tuple(pair)],
        reverse=True,
    )

    print("주요 오분류:")
    for true_index, predicted_index in error_pairs:
        print(
            f"  {class_names[true_index]} -> "
            f"{class_names[predicted_index]}: "
            f"{errors[true_index, predicted_index]}건"
        )


def main():
    label_encoder = load(LABEL_ENCODER_PATH)
    Xte, yte = load_test_dataset(label_encoder)
    class_names = label_encoder.classes_

    for model_name, model_filename in MODEL_FILENAMES.items():
        model = load(MODEL_DIR / model_filename)
        predictions = aggregate_trial_predictions(model, Xte)
        matrix = confusion_matrix(
            yte,
            predictions,
            labels=np.arange(len(class_names)),
        )
        output_path = OUTPUT_DIR / OUTPUT_FILENAMES[model_name]

        plot_confusion_matrix(
            matrix,
            class_names,
            model_name,
            output_path,
        )
        print_structure_summary(matrix, class_names, model_name)
        print("혼동행렬 합계:", matrix.sum())
        print("이미지 저장:", output_path)


if __name__ == "__main__":
    main()
