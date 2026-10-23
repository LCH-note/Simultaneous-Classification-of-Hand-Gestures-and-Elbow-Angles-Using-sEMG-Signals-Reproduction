"""피험자별 외부 8-fold / 내부 7-fold nested LOSO를 수행한다.

실행: IDE Run 버튼 또는 python -m src.loso_8_fold
입력 특징은 표준화 전 값이어야 한다. 표준화는 매 학습 분할에서 적합한다.
"""

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 모듈 실행과 IDE의 파일 직접 실행을 모두 지원한다.
if __package__:
    from .five_fold import (
        INFERENCE_REPEATS,
        RANDOM_STATE,
        SIGNIFICANCE_LEVEL,
        aggregate_trial_predictions,
        create_models,
        flatten_training_windows,
        run_paired_t_tests,
        scale_fold,
        summarize_results,
    )
else:
    from five_fold import (
        INFERENCE_REPEATS,
        RANDOM_STATE,
        SIGNIFICANCE_LEVEL,
        aggregate_trial_predictions,
        create_models,
        flatten_training_windows,
        run_paired_t_tests,
        scale_fold,
        summarize_results,
    )


OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "data" / "loso"
DATASETS = {
    "no_time_axis": OUTPUT_ROOT / "dataset_loso.npz",
    "time_axis": OUTPUT_ROOT / "dataset_time_axis_loso.npz",
}
# 기존 설정을 포함한 후보. Test 결과를 보고 후보를 변경하지 않는다.
PARAMETER_CANDIDATES = {
    "SVM": [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}],
    "RF": [{"max_depth": None}, {"max_depth": 10}],
    "XGB": [{"max_depth": 6}, {"max_depth": 3}],
}
PREPROCESSING_NOTE = (
    "Scaling is fitted on training subjects only. Precomputed aligned inputs "
    "may contain same-file, same-class donor-trial imputation; this evaluation "
    "does not establish independent-trial or label-blind online preprocessing."
)


def load_dataset(path):
    """전체 시행 특징을 읽되 표준화나 윈도우 무작위 분할은 하지 않는다."""
    with np.load(path, allow_pickle=False) as data:
        return tuple(data[key] for key in ("X", "y", "subjects", "trial_ids"))


def subject_splits(subjects):
    """한 피험자를 외부 Test로, 나머지 중 한 명을 내부 Validation으로 둔다."""
    for test_subject in np.unique(subjects):
        outer_train = np.flatnonzero(subjects != test_subject)
        outer_test = np.flatnonzero(subjects == test_subject)
        inner_splits = []
        for validation_subject in np.unique(subjects[outer_train]):
            train = outer_train[subjects[outer_train] != validation_subject]
            validation = outer_train[subjects[outer_train] == validation_subject]
            inner_splits.append((str(validation_subject), train, validation))
        yield str(test_subject), outer_train, outer_test, inner_splits


def select_parameters(X, labels, inner_splits, models, candidates):
    """6명 학습 / 1명 검증의 평균으로 모델별 설정을 고른다."""
    scores = {name: [[] for _ in candidates[name]] for name in models}
    for validation_subject, train, validation in inner_splits:
        Xtr, Xval = scale_fold(X[train], X[validation])
        encoder = LabelEncoder().fit(labels[train])
        train_windows, ytr = flatten_training_windows(
            Xtr, encoder.transform(labels[train])
        )
        yval = encoder.transform(labels[validation])
        for name, prototype in models.items():
            for index, params in enumerate(candidates[name]):
                model = clone(prototype).set_params(**params)
                model.fit(train_windows, ytr)
                prediction = aggregate_trial_predictions(model, Xval)
                scores[name][index].append(float(accuracy_score(yval, prediction)))
        print(f"  내부 Validation {validation_subject} 완료", flush=True)

    selection = {}
    for name in models:
        means = [float(np.mean(values)) for values in scores[name]]
        # 동률이면 미리 정한 후보 순서를 따른다.
        best_index = int(np.argmax(means))
        selection[name] = {
            "selected_params": candidates[name][best_index],
            "best_inner_mean_accuracy": means[best_index],
            "candidates": [
                {"params": params, "fold_accuracies": values, "mean_accuracy": mean}
                for params, values, mean in zip(candidates[name], scores[name], means)
            ],
        }
    return selection


def run_nested_loso(X, labels, subjects, trial_ids, models, candidates,
                    inference_repeats=INFERENCE_REPEATS):
    """내부 선택 후 7명으로 재학습하고 외부 Test의 시행 점수를 계산한다."""
    fold_accuracy = {name: [] for name in models}
    folds = []
    for test_subject, train, test, inner_splits in subject_splits(subjects):
        print(f"외부 Test {test_subject}", flush=True)
        search_start = time.perf_counter()
        # 외부 Test의 특징과 라벨은 내부 선택 함수에 전달하지 않는다.
        local_subjects = subjects[train]
        local_splits = [
            (name, np.flatnonzero(local_subjects != name),
             np.flatnonzero(local_subjects == name))
            for name, _, _ in inner_splits
        ]
        selection = select_parameters(
            X[train], labels[train], local_splits, models, candidates
        )
        search_seconds = time.perf_counter() - search_start

        # 선택 종료 후 외부 학습 7명으로 scaler와 모델을 새로 적합한다.
        Xtr, Xtest = scale_fold(X[train], X[test])
        encoder = LabelEncoder().fit(labels[train])
        train_windows, train_labels = flatten_training_windows(
            Xtr, encoder.transform(labels[train])
        )
        fold = {
            "test_subject": test_subject,
            "train_subjects": np.unique(subjects[train]).tolist(),
            "inner_splits": [
                {"validation_subject": name,
                 "train_subjects": np.unique(subjects[indices]).tolist(),
                 "train_trial_count": len(indices),
                 "validation_trial_count": len(validation)}
                for name, indices, validation in inner_splits
            ],
            "train_trial_count": len(train),
            "test_trial_count": len(test),
            "inner_search_seconds": search_seconds,
            "test_trial_ids": trial_ids[test].tolist(),
            "true_labels": labels[test].tolist(),
            "models": {},
        }
        for name, prototype in models.items():
            model = clone(prototype).set_params(**selection[name]["selected_params"])
            start = time.perf_counter()
            model.fit(train_windows, train_labels)
            training_seconds = time.perf_counter() - start
            prediction = aggregate_trial_predictions(model, Xtest)
            predicted_labels = encoder.inverse_transform(prediction)
            accuracy = float(accuracy_score(labels[test], predicted_labels))
            fold_accuracy[name].append(accuracy)

            # 동일 모델로 시행 1개의 추론 비용만 반복 측정한다.
            sample = Xtest[:1]
            aggregate_trial_predictions(model, sample)
            start = time.perf_counter()
            for _ in range(inference_repeats):
                aggregate_trial_predictions(model, sample)
            inference_ms = (time.perf_counter() - start) * 1000 / inference_repeats
            fold["models"][name] = {
                **selection[name], "accuracy": accuracy,
                "predictions": predicted_labels.tolist(),
                "training_seconds": training_seconds,
                "inference_ms_per_trial": inference_ms,
            }
            print(f"  {name}: {accuracy:.2%}", flush=True)
        folds.append(fold)

    timing = {
        name: {
            "training_seconds": float(np.mean([
                fold["models"][name]["training_seconds"] for fold in folds
            ])),
            "inference_ms_per_trial": float(np.mean([
                fold["models"][name]["inference_ms_per_trial"] for fold in folds
            ])),
            "inference_repeats": inference_repeats,
        }
        for name in models
    }
    return fold_accuracy, folds, timing


def save_figures(output_dir, results, tests, timing):
    """기존 five_fold와 같은 정확도·t검정·실행시간 그래프를 저장한다."""
    names = list(results)
    colors = ["#4C78A8", "#59A14F", "#F28E2B"]
    fig, axis = plt.subplots(figsize=(8, 5))
    means = [results[name]["mean_accuracy"] * 100 for name in names]
    deviations = [results[name]["std_accuracy"] * 100 for name in names]
    bars = axis.bar(names, means, yerr=deviations, capsize=6, color=colors)
    axis.bar_label(bars, labels=[f"{m:.2f} ± {s:.2f}%" for m, s in zip(means, deviations)], padding=4)
    axis.set(title="Nested LOSO: Outer Test Accuracy", ylabel="Accuracy (%)", ylim=(0, 110))
    fig.tight_layout()
    fig.savefig(output_dir / "loso_8_fold_comparison.png", dpi=120)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    # NaN p값(모든 fold 차이가 0 등)은 유의함으로 해석하지 않는다.
    pvalues = np.asarray([item["p_value"] for item in tests])
    heights = np.where(np.isfinite(pvalues), np.maximum(pvalues, np.finfo(float).tiny), 1.0)
    bars = axis.bar([item["comparison"] for item in tests], heights, color=colors)
    axis.bar_label(bars, labels=[f"p={p:.4g}" for p in pvalues], padding=3)
    axis.axhline(SIGNIFICANCE_LEVEL, color="red", linestyle="--", label="alpha = 0.05")
    axis.set(yscale="log", ylabel="p-value", title="Paired t-test: Outer Fold Accuracies")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "paired_ttest_comparison.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, key, title, unit in zip(
        axes, ("training_seconds", "inference_ms_per_trial"),
        ("Mean Outer Refit Time (7 subjects)", "Mean Inference Time per Trial"),
        ("Seconds", "Milliseconds"),
    ):
        bars = axis.bar(names, [timing[name][key] for name in names], color=colors)
        axis.bar_label(bars, fmt="%.3f", padding=3)
        axis.set(title=title, ylabel=unit)
    fig.tight_layout()
    fig.savefig(output_dir / "model_timing_comparison.png", dpi=120)
    plt.close(fig)


def save_results(dataset_path, output_dir, X, labels, subjects, models,
                 fold_accuracy, folds, timing):
    """기존 요약 필드에 내부 선택 내역과 참가자별 예측을 추가한다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = summarize_results(fold_accuracy)
    tests = run_paired_t_tests(fold_accuracy)
    payload = {
        "dataset": str(dataset_path), "sample_unit": "trial",
        "sample_count": len(labels), "windows_per_trial": X.shape[1],
        "class_names": np.unique(labels).tolist(), "n_splits": len(folds),
        "inner_n_splits": len(np.unique(subjects)) - 1,
        "random_state": RANDOM_STATE,
        "aggregation": "mean window probability or decision score",
        "selection_metric": "mean inner subject accuracy",
        "base_model_params": {name: model.get_params() for name, model in models.items()},
        "parameter_candidates": {
            name: [item["params"] for item in folds[0]["models"][name]["candidates"]]
            for name in models
        },
        "models": results, "paired_t_tests": tests, "timing": timing,
        "timing_scope": "mean outer refit on 7 subjects; excludes inner search",
        "preprocessing_note": PREPROCESSING_NOTE,
        "statistical_note": "Model-pair t-tests are exploratory: overlapping training folds, 8 subjects, no multiplicity correction.",
        "folds": folds,
    }
    (output_dir / "loso_8_fold_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["Nested LOSO 8-fold 교차검증 결과", f"데이터셋: {dataset_path}",
             "평가 단위: 시행 / fold 순서: " + ", ".join(f["test_subject"] for f in folds)]
    for name, result in results.items():
        lines.extend([
            f"{name} fold 정확도: " + ", ".join(f"{a:.2%}" for a in result["fold_accuracies"]),
            f"{name}: {result['mean_accuracy'] * 100:.2f} ± {result['std_accuracy'] * 100:.2f}%",
        ])
    lines.append("대응표본 t-검정 (모델 간 비교, 탐색적 해석)")
    lines.extend(f"{t['comparison']}: t={t['t_statistic']:.4f}, p={t['p_value']:.6f} ({t['verdict']})" for t in tests)
    lines.append("실행시간: 내부 탐색 제외, 외부 7명 재학습의 fold 평균")
    lines.extend(f"{name}: 학습 {t['training_seconds']:.3f}s, 추론 {t['inference_ms_per_trial']:.3f} ms/trial" for name, t in timing.items())
    lines.append(PREPROCESSING_NOTE)
    (output_dir / "loso_8_fold_results.txt").write_text("\n".join(lines), encoding="utf-8")
    save_figures(output_dir, results, tests, timing)


def main():
    for condition, path in DATASETS.items():
        X, labels, subjects, trial_ids = load_dataset(path)
        models = create_models()
        scores, folds, timing = run_nested_loso(
            X, labels, subjects, trial_ids, models, PARAMETER_CANDIDATES
        )
        save_results(path, OUTPUT_ROOT / condition, X, labels, subjects,
                     models, scores, folds, timing)


if __name__ == "__main__":
    main()
