"""세 모델의 시행 단위 5-fold 교차검증 결과를 저장한다."""

import json
import time
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from scipy.stats import ttest_rel
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "dataset_improve.npz"
OUTPUT_DIR = PROJECT_ROOT / "data" / "five fold"
JSON_OUTPUT_PATH = OUTPUT_DIR / "five_fold_results.json"
TEXT_OUTPUT_PATH = OUTPUT_DIR / "five_fold_results.txt"
IMAGE_OUTPUT_PATH = OUTPUT_DIR / "five_fold_comparison.png"
TTEST_IMAGE_OUTPUT_PATH = OUTPUT_DIR / "paired_ttest_comparison.png"
TIMING_IMAGE_OUTPUT_PATH = OUTPUT_DIR / "model_timing_comparison.png"

N_SPLITS = 5
RANDOM_STATE = 42
SIGNIFICANCE_LEVEL = 0.05
INFERENCE_REPEATS = 100
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


def load_holdout_dataset(label_encoder):
    """실행시간 측정에 사용할 기존 hold-out 데이터를 불러온다."""
    with np.load(DATASET_PATH, allow_pickle=True) as data:
        Xtr = data["Xtr"]
        ytr = label_encoder.transform(data["ytr"])
        Xte = data["Xte"]
    return Xtr, ytr, Xte


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


def run_paired_t_tests(fold_accuracy):
    """모델 쌍의 fold 정확도에 대응표본 t-검정을 수행한다."""
    test_results = []
    for left, right in combinations(fold_accuracy, 2):
        statistic, p_value = ttest_rel(
            fold_accuracy[left],
            fold_accuracy[right],
        )
        is_significant = bool(
            np.isfinite(p_value) and p_value < SIGNIFICANCE_LEVEL
        )
        verdict = "유의함" if is_significant else "유의하지 않음"
        test_results.append({
            "comparison": f"{left} vs {right}",
            "t_statistic": float(statistic),
            "p_value": float(p_value),
            "alpha": SIGNIFICANCE_LEVEL,
            "significant": is_significant,
            "verdict": verdict,
        })
    return test_results


def measure_model_times(Xtr, ytr, Xte, models):
    """모델별 학습 시간과 시행 1개의 평균 추론 시간을 측정한다."""
    train_windows, train_labels = flatten_training_windows(Xtr, ytr)
    sample = Xte[:1]
    timing_results = {}

    for name, prototype in models.items():
        model = clone(prototype)
        start = time.perf_counter()
        model.fit(train_windows, train_labels)
        training_seconds = time.perf_counter() - start

        aggregate_trial_predictions(model, sample)
        start = time.perf_counter()
        for _ in range(INFERENCE_REPEATS):
            aggregate_trial_predictions(model, sample)
        inference_ms = (
            (time.perf_counter() - start)
            * 1000
            / INFERENCE_REPEATS
        )

        timing_results[name] = {
            "training_seconds": float(training_seconds),
            "inference_ms_per_trial": float(inference_ms),
            "inference_repeats": INFERENCE_REPEATS,
        }
        print(
            f"{name}: 학습 {training_seconds:.3f}s, "
            f"추론 {inference_ms:.3f} ms/trial"
        )

    return timing_results


def save_json(
    results,
    class_names,
    sample_count,
    t_test_results,
    timing_results,
):
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
        "paired_t_tests": t_test_results,
        "timing": timing_results,
    }
    JSON_OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_text(results, t_test_results, timing_results):
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

    lines.extend(["대응표본 t-검정", f"유의수준: {SIGNIFICANCE_LEVEL}"])
    for result in t_test_results:
        lines.append(
            f"{result['comparison']}: "
            f"t = {result['t_statistic']:.4f}, "
            f"p = {result['p_value']:.6f} "
            f"({result['verdict']})"
        )

    lines.extend([
        "",
        "모델 실행시간",
        f"추론 반복 횟수: {INFERENCE_REPEATS}",
    ])
    for name, result in timing_results.items():
        lines.append(
            f"{name}: 학습 {result['training_seconds']:.3f}s, "
            f"추론 {result['inference_ms_per_trial']:.3f} ms/trial"
        )

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


def save_ttest_image(t_test_results):
    """모델 쌍별 p-value를 로그 스케일 막대그래프로 저장한다."""
    names = [result["comparison"] for result in t_test_results]
    p_values = [
        max(result["p_value"], np.finfo(float).tiny)
        for result in t_test_results
    ]

    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    bars = axis.bar(
        names,
        p_values,
        color=["#4C78A8", "#59A14F", "#F28E2B"],
    )
    axis.axhline(
        SIGNIFICANCE_LEVEL,
        color="red",
        linestyle="--",
        label=f"alpha = {SIGNIFICANCE_LEVEL}",
    )
    axis.set_yscale("log")
    axis.set_ylabel("p-value (log scale)")
    axis.set_title("Paired t-test of Fold Accuracies")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)

    for bar, p_value in zip(bars, p_values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            p_value,
            f"p={p_value:.4g}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()
    figure.savefig(
        TTEST_IMAGE_OUTPUT_PATH,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_timing_image(timing_results):
    """모델별 학습·추론 시간을 두 개의 그래프로 저장한다."""
    names = list(timing_results)
    training_seconds = [
        timing_results[name]["training_seconds"]
        for name in names
    ]
    inference_ms = [
        timing_results[name]["inference_ms_per_trial"]
        for name in names
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = ["#4C78A8", "#59A14F", "#F28E2B"]
    train_bars = axes[0].bar(names, training_seconds, color=colors)
    inference_bars = axes[1].bar(names, inference_ms, color=colors)

    axes[0].set_title("Training Time")
    axes[0].set_ylabel("Seconds")
    axes[1].set_title("Inference Time per Trial")
    axes[1].set_ylabel("Milliseconds")

    for axis, bars, values in (
        (axes[0], train_bars, training_seconds),
        (axes[1], inference_bars, inference_ms),
    ):
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    figure.tight_layout()
    figure.savefig(
        TIMING_IMAGE_OUTPUT_PATH,
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
    t_test_results = run_paired_t_tests(fold_accuracy)
    Xtr, ytr, Xte = load_holdout_dataset(label_encoder)
    timing_results = measure_model_times(Xtr, ytr, Xte, models)

    save_json(
        results,
        label_encoder.classes_,
        len(y),
        t_test_results,
        timing_results,
    )
    save_text(results, t_test_results, timing_results)
    save_comparison_image(results)
    save_ttest_image(t_test_results)
    save_timing_image(timing_results)

    print("최종 결과:")
    for name, result in results.items():
        print(
            f"{name}: {result['mean_accuracy'] * 100:.2f} "
            f"± {result['std_accuracy'] * 100:.2f}%"
        )
    print("대응표본 t-검정:")
    for result in t_test_results:
        print(
            f"{result['comparison']}: "
            f"p = {result['p_value']:.4f} "
            f"({result['verdict']})"
        )
    print("JSON 저장:", JSON_OUTPUT_PATH)
    print("텍스트 저장:", TEXT_OUTPUT_PATH)
    print("정확도 이미지 저장:", IMAGE_OUTPUT_PATH)
    print("t-검정 이미지 저장:", TTEST_IMAGE_OUTPUT_PATH)
    print("실행시간 이미지 저장:", TIMING_IMAGE_OUTPUT_PATH)


if __name__ == "__main__":
    main()
