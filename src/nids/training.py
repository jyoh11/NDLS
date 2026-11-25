"""Reproducible training, validation-only selection, and held-out evaluation.

The final test partition is not used for fitting, calibration, model selection, or
threshold selection. Model preprocessing lives inside each estimator pipeline,
so imputation and encoding statistics are learned only from training rows.
"""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .calibration import calibration_diagnostics, calibrate_fitted_estimator, positive_probability
from .features import split_features_and_labels
from .ingest import read_chunks, validate_frame
from .preprocessing import build_preprocessor, preprocess_frame, split_preprocessed_frame
from .policy import confidence_alert

LOG = logging.getLogger(__name__)


def _models(seed: int, model_parameters: dict | None = None) -> dict:
    """Construct the synopsis baseline and a simple comparison estimator."""
    parameters = model_parameters or {}
    rf = parameters.get("random_forest", {})
    lr = parameters.get("logistic_regression", {})
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=int(rf.get("n_estimators", 160)),
            max_depth=rf.get("max_depth"),
            min_samples_leaf=int(rf.get("min_samples_leaf", 2)),
            max_features=rf.get("max_features", "sqrt"),
            class_weight=rf.get("class_weight", "balanced_subsample"),
            n_jobs=int(rf.get("n_jobs", -1)),
            random_state=seed,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=int(lr.get("max_iter", 1000)),
            class_weight=lr.get("class_weight", "balanced"),
            C=float(lr.get("C", 1.0)),
            random_state=seed,
        ),
    }


def _choose_threshold(y: np.ndarray, probabilities: np.ndarray, target_recall: float) -> float:
    """Choose the highest validation threshold that meets requested recall."""
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    eligible = [float(t) for t in candidates
                if recall_score(y, probabilities >= t, zero_division=0) >= target_recall]
    return max(eligible) if eligible else 0.5


def _metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """Calculate binary and per-class metrics with benign as class 0."""
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    class_report = classification_report(
        y, predicted, labels=[0, 1], target_names=["benign", "attack"],
        output_dict=True, zero_division=0,
    )
    both_classes = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold),
        "confusion_matrix": {
            "labels": ["benign", "attack"],
            "rows_actual_columns_predicted": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        # FPR is the share of actual benign flows incorrectly marked attack.
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_positive_count": int(fp),
        # FNR is the share of actual attack flows incorrectly marked benign.
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "false_negative_count": int(fn),
        "alert_volume": int(predicted.sum()),
        "roc_auc": float(roc_auc_score(y, probabilities)) if both_classes else None,
        "pr_auc": float(average_precision_score(y, probabilities)) if both_classes else None,
        "brier_score": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "per_class": {
            "benign": {key: float(value) for key, value in class_report["benign"].items()},
            "attack": {key: float(value) for key, value in class_report["attack"].items()},
        },
    }


def _policy_decisions(y: np.ndarray, probabilities: np.ndarray, threshold: float,
                      review_threshold: float, minimum_confidence: float) -> tuple[dict, list[dict]]:
    """Evaluate high-priority alerts separately from analyst-review workload."""
    records = [confidence_alert(float(probability), threshold, review_threshold, minimum_confidence)
               for probability in probabilities]
    high_priority = np.asarray([item["policy_decision"] == "alert_attack" for item in records], dtype=bool)
    surfaced = np.asarray([item["requires_attention"] for item in records], dtype=bool)
    tn, fp, fn, tp = confusion_matrix(y, high_priority, labels=[0, 1]).ravel()
    class_report = classification_report(
        y, high_priority, labels=[0, 1], target_names=["benign", "attack"],
        output_dict=True, zero_division=0,
    )
    attention_tn, attention_fp, attention_fn, attention_tp = confusion_matrix(y, surfaced, labels=[0, 1]).ravel()
    summary = {
        "threshold": float(threshold),
        "review_threshold": float(review_threshold),
        "minimum_confidence": float(minimum_confidence),
        "confusion_matrix": {"labels": ["benign", "attack"],
                             "rows_actual_columns_high_priority": [[int(tn), int(fp)], [int(fn), int(tp)]],
                             "rows_actual_columns_attention": [[int(attention_tn), int(attention_fp)],
                                                                [int(attention_fn), int(attention_tp)]]},
        "false_positive_count": int(fp),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "precision": float(precision_score(y, high_priority, zero_division=0)),
        "attack_recall": float(recall_score(y, high_priority, zero_division=0)),
        "recall": float(recall_score(y, high_priority, zero_division=0)),
        "accuracy": float(accuracy_score(y, high_priority)),
        "f1": float(f1_score(y, high_priority, zero_division=0)),
        "per_class": {
            "benign": {key: float(value) for key, value in class_report["benign"].items()},
            "attack": {key: float(value) for key, value in class_report["attack"].items()},
        },
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "brier_score": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, np.clip(probabilities, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "false_negative_count": int(fn),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "attack_recall_attention": float(recall_score(y, surfaced, zero_division=0)),
        "false_positive_count_attention": int(attention_fp),
        "false_positive_rate_attention": float(attention_fp / (attention_fp + attention_tn))
        if attention_fp + attention_tn else None,
        "alert_volume": int(surfaced.sum()),
        "alert_volume_rate": float(surfaced.mean()) if len(surfaced) else 0.0,
        "high_priority_count": int(sum(item["policy_decision"] == "alert_attack" for item in records)),
        "review_count": int(sum(item["policy_decision"] == "review" for item in records)),
        "reviewed_benign_count": int(sum(item["policy_decision"] == "review" and actual == 0
                                           for item, actual in zip(records, y))),
        "reviewed_attack_count": int(sum(item["policy_decision"] == "review" and actual == 1
                                          for item, actual in zip(records, y))),
        "suppressed_benign_count": int(sum(item["policy_decision"] == "suppress_benign" for item in records)),
        "suppressed_attack_count": int(sum(item["policy_decision"] == "suppress_benign" and actual == 1
                                            for item, actual in zip(records, y))),
        "high_priority_false_positive_count": int(sum(item["policy_decision"] == "alert_attack" and actual == 0
                                                       for item, actual in zip(records, y))),
        "high_priority_attack_count": int(sum(item["policy_decision"] == "alert_attack" and actual == 1
                                               for item, actual in zip(records, y))),
    }
    return summary, records


def _choose_policy_threshold(y: np.ndarray, probabilities: np.ndarray, target_recall: float,
                             review_threshold: float, minimum_confidence: float) -> float:
    """Choose the largest alert cutoff meeting recall after benign suppression."""
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])
    eligible = []
    for threshold in candidates:
        summary, _ = _policy_decisions(y, probabilities, float(threshold), review_threshold, minimum_confidence)
        if summary["attack_recall"] >= target_recall:
            eligible.append(float(threshold))
    return max(eligible) if eligible else 0.0


def _split_calibration_validation(validation_frame: pd.DataFrame, seed: int,
                                  calibration_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split validation into disjoint calibration and policy-threshold subsets."""
    if not 0.1 <= calibration_fraction <= 0.9:
        raise ValueError("calibration_fraction must be between 0.1 and 0.9")
    labels = validation_frame["binary_target"].to_numpy(dtype="int8")
    counts = np.bincount(labels, minlength=2)
    if counts.min() < 4:
        raise ValueError("Validation data needs at least four benign and four attack rows so calibration and policy subsets each retain both classes")
    calibration, operating = train_test_split(
        validation_frame, train_size=calibration_fraction, random_state=seed,
        stratify=labels,
    )
    if calibration["binary_target"].nunique() < 2 or operating["binary_target"].nunique() < 2:
        raise ValueError("Calibration and policy-validation partitions must each contain both classes")
    return calibration.reset_index(drop=True), operating.reset_index(drop=True)


def _save_plots(y: np.ndarray, probabilities: np.ndarray, threshold: float,
                output_dir: Path, run_id: str) -> dict[str, str]:
    """Save ROC, precision-recall, and confusion-matrix PNGs from test predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: dict[str, str] = {}
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, probabilities)
        precision, recall, _ = precision_recall_curve(y, probabilities)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(fpr, tpr, label="Test ROC")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
        axes[0].legend(loc="lower right")
        axes[1].plot(recall, precision, label="Test precision-recall")
        axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
        axes[1].legend(loc="best")
        fig.tight_layout()
        path = output_dir / "roc_pr_curves.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths["roc_pr_curves"] = path.name

    predicted = probabilities >= threshold
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Benign", "Attack"],
             yticklabels=["Benign", "Attack"], xlabel="Predicted class", ylabel="Actual class",
             title="Held-out test confusion matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    path = output_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths["confusion_matrix"] = path.name
    return paths


def _save_calibration_plot(y: np.ndarray, raw_probability: np.ndarray,
                          calibrated_probability: np.ndarray, output_dir: Path,
                          n_bins: int) -> str:
    """Save test-set reliability diagrams for raw and calibrated scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = calibration_diagnostics(y, raw_probability, n_bins)
    calibrated = calibration_diagnostics(y, calibrated_probability, n_bins)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect reliability")
    for name, report in (("Uncalibrated", raw), ("Calibrated", calibrated)):
        bins = report["reliability_bins"]
        axis.plot([item["mean_probability"] for item in bins],
                  [item["observed_positive_rate"] for item in bins], marker="o", label=name)
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted P(attack)",
             ylabel="Observed attack fraction", title="Held-out test reliability diagram")
    axis.legend(loc="best")
    fig.tight_layout()
    path = output_dir / "reliability_diagram.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _save_policy_tradeoff_plot(comparison: dict, output_dir: Path) -> str:
    """Plot false-positive counts against missed attacks for the three policies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    false_positives = []
    false_negatives = []
    for name, values in comparison.items():
        labels.append(name.replace("_", "\n"))
        false_positives.append(values["false_positive_count"])
        false_negatives.append(values.get("suppressed_attack_count", values["false_negative_count"]))
    positions = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(7, 4))
    width = 0.36
    axis.bar(positions - width / 2, false_positives, width, label="False positives")
    axis.bar(positions + width / 2, false_negatives, width, label="Missed/suppressed attacks")
    axis.set(xticks=positions, xticklabels=labels, ylabel="Test rows", title="Held-out policy trade-off")
    axis.legend()
    fig.tight_layout()
    path = output_dir / "policy_tradeoff.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path.name


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def load_dataset(path: str | Path, label_column: str | None = None, chunksize: int = 250_000,
                 max_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Read chunks and optionally make a deterministic, class-preserving sample."""
    if max_rows is not None:
        from tempfile import TemporaryDirectory
        from .ingest import prepare_sample
        with TemporaryDirectory(prefix="nids-train-sample-") as temporary:
            sample_path = Path(temporary) / "sample.csv"
            prepare_sample(path, sample_path, max_rows, label_column or "", seed, chunksize)
            frame = pd.read_csv(sample_path, low_memory=False)
        validate_frame(frame, label_column)
        return frame

    chunks = []
    for frame in read_chunks(path, chunksize):
        validate_frame(frame, label_column)
        chunks.append(frame)
    if not chunks:
        raise ValueError(f"No usable CSV rows found under {path}")
    result = pd.concat(chunks, ignore_index=True)
    validate_frame(result, label_column)
    return result


def train_and_evaluate(frame: pd.DataFrame, output_dir: str | Path = "artifacts",
                       label_column: str | None = None, seed: int = 42,
                       target_recall: float = 0.95, review_threshold: float = 0.50,
                       max_train_rows: int | None = 500_000,
                       calibration_method: str = "sigmoid",
                       test_size: float = 0.20, validation_size: float = 0.20,
                       model_names: list[str] | None = None,
                       explicit_exclusions: list[str] | None = None,
                       drop_exact_duplicates: bool = True,
                       model_parameters: dict | None = None,
                       dataset_version: str = "unspecified-local-dataset",
                       collection_note: str | None = None,
                       calibration_fraction: float = 0.5,
                       minimum_confidence: float = 0.8,
                       ece_bins: int = 10,
                       min_isotonic_samples: int = 1000,
                       explanation_background_size: int = 100,
                       explanation_sample_size: int = 100,
                       explanation_top_k: int = 15,
                       explanation_max_dense_mb: float = 512.0,
                       generate_explanations: bool = True) -> dict:
    """Fit candidates, select by validation metrics, and evaluate winner once on test."""
    if not label_column:
        raise ValueError("Pass the verified label column explicitly; inspect the dataset before training")
    started = time.perf_counter()
    processed, quality_report = preprocess_frame(
        frame, label_column, explicit_exclusions=explicit_exclusions,
        drop_exact_duplicates=drop_exact_duplicates,
    )
    splits, split_report = split_preprocessed_frame(processed, test_size, validation_size, seed)
    train_frame = splits["train"]
    validation_frame = splits["validation"]
    test_frame = splits["test"]
    calibration_frame, operating_frame = _split_calibration_validation(
        validation_frame, seed, calibration_fraction
    )
    X_train, y_train_series = split_features_and_labels(train_frame, "binary_target", explicit_exclusions)
    X_calibration, y_calibration_series = split_features_and_labels(calibration_frame, "binary_target", explicit_exclusions)
    X_operating, y_operating_series = split_features_and_labels(operating_frame, "binary_target", explicit_exclusions)
    X_test, y_test_series = split_features_and_labels(test_frame, "binary_target", explicit_exclusions)
    y_train = y_train_series.to_numpy(dtype="int8")
    y_calibration = y_calibration_series.to_numpy(dtype="int8")
    y_operating = y_operating_series.to_numpy(dtype="int8")
    y_test = y_test_series.to_numpy(dtype="int8")

    if max_train_rows is not None and max_train_rows > 0 and len(X_train) > max_train_rows:
        strata = train_frame["normalized_attack_label"].astype(str)
        if strata.nunique() > max_train_rows:
            raise ValueError("max_train_rows is smaller than the observed attack-class count; increase it to retain every class")
        if strata.value_counts().min() < 2:
            strata = pd.Series(y_train, index=X_train.index)
        chosen, _ = train_test_split(np.arange(len(X_train)), train_size=max_train_rows,
                                     random_state=seed, stratify=strata.to_numpy())
        X_train = X_train.iloc[chosen]
        y_train = y_train[chosen]

    candidates = _models(seed, model_parameters)
    if model_names is not None:
        unknown = sorted(set(model_names) - set(candidates))
        if unknown:
            raise ValueError(f"Unsupported model names: {unknown}")
        candidates = {name: candidates[name] for name in model_names}
    if not candidates:
        raise ValueError("Select at least one supported model")

    scores: dict[str, dict] = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocess", build_preprocessor(X_train, scale_numeric=name == "logistic_regression")),
            ("classifier", estimator),
        ])
        LOG.info("Fitting %s on %d training flows", name, len(X_train))
        fit_start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_start
        calibration_start = time.perf_counter()
        calibrated = calibrate_fitted_estimator(
            pipeline, X_calibration, y_calibration, calibration_method, min_isotonic_samples
        )
        calibration_seconds = time.perf_counter() - calibration_start
        operating_raw_probability = positive_probability(pipeline, X_operating)
        operating_probability = positive_probability(calibrated, X_operating)
        threshold = _choose_policy_threshold(
            y_operating, operating_probability, target_recall, review_threshold, minimum_confidence
        )
        validation_metrics = _metrics(y_operating, operating_probability, threshold)
        validation_policy, _ = _policy_decisions(
            y_operating, operating_probability, threshold, review_threshold, minimum_confidence
        )
        validation_metrics["target_recall"] = float(target_recall)
        validation_metrics["target_recall_met"] = validation_policy["attack_recall"] >= target_recall
        validation_metrics["policy_summary"] = validation_policy
        validation_metrics["uncalibrated_at_0_5"] = _metrics(y_operating, operating_raw_probability, 0.5)
        validation_metrics["calibrated_at_0_5"] = _metrics(y_operating, operating_probability, 0.5)
        if explanation_background_size < 1:
            raise ValueError("explanation_background_size must be at least 1")
        background = X_train.sample(n=min(explanation_background_size, len(X_train)), random_state=seed).copy()
        scores[name] = {
            "validation": validation_metrics,
            "fit_seconds": fit_seconds,
            "calibration_seconds": calibration_seconds,
            "bundle": {
                "model": calibrated,
                "uncalibrated_model": pipeline,
                "preprocessor": pipeline.named_steps["preprocess"],
                "features": X_train.columns.tolist(),
                "background": background,
                "threshold": threshold,
                "review_threshold": review_threshold,
                "minimum_confidence": minimum_confidence,
                "model_name": name,
                "calibration_method": calibration_method,
                "calibration_fraction_of_validation": calibration_fraction,
                "feature_dtypes": {
                    column: ("numeric" if pd.api.types.is_numeric_dtype(X_train[column].dtype) else "categorical")
                    for column in X_train.columns
                },
                "artifact_version": 1,
                "explanation_config": {
                    "background_sample_size": explanation_background_size,
                    "explanation_sample_size": explanation_sample_size,
                    "top_k": explanation_top_k,
                    "max_dense_mb": explanation_max_dense_mb,
                },
            },
        }

    selected = min(scores, key=lambda name: (
        not scores[name]["validation"]["target_recall_met"],
        scores[name]["validation"]["policy_summary"]["false_positive_rate"]
        if scores[name]["validation"]["policy_summary"]["false_positive_rate"] is not None else 1.0,
        -scores[name]["validation"]["policy_summary"]["attack_recall"],
        -scores[name]["validation"]["pr_auc"] if scores[name]["validation"]["pr_auc"] is not None else 0.0,
    ))
    bundle = scores[selected]["bundle"]
    inference_start = time.perf_counter()
    raw_test_probability = positive_probability(bundle["uncalibrated_model"], X_test)
    raw_inference_seconds = time.perf_counter() - inference_start
    inference_start = time.perf_counter()
    test_probability = positive_probability(bundle["model"], X_test)
    inference_seconds = time.perf_counter() - inference_start
    confidence_policy_metrics, test_decisions = _policy_decisions(
        y_test, test_probability, bundle["threshold"], bundle["review_threshold"], bundle["minimum_confidence"]
    )
    raw_metrics = _metrics(y_test, raw_test_probability, 0.5)
    calibrated_metrics = _metrics(y_test, test_probability, 0.5)
    for classifier_metrics in (raw_metrics, calibrated_metrics):
        classifier_metrics["attack_recall"] = classifier_metrics["recall"]
        classifier_metrics["suppressed_attack_count"] = classifier_metrics["false_negative_count"]
    metrics = {
        **confidence_policy_metrics,
        "confusion_matrix": {
            "labels": ["benign", "attack"],
            "rows_actual_columns_predicted": confidence_policy_metrics["confusion_matrix"]["rows_actual_columns_high_priority"],
        },
        "baseline_0_5": calibrated_metrics,
        "selected_threshold_calibrated": _metrics(y_test, test_probability, bundle["threshold"]),
        "decision_comparison": {
            "uncalibrated_classifier_at_0_5": raw_metrics,
            "calibrated_classifier_at_0_5": calibrated_metrics,
            "confidence_aware_policy": confidence_policy_metrics,
        },
        "calibration_comparison": {
            "uncalibrated": calibration_diagnostics(y_test, raw_test_probability, ece_bins),
            "calibrated": calibration_diagnostics(y_test, test_probability, ece_bins),
            "note": "Test rows are used only for final reporting; calibration was fitted on a disjoint calibration subset of validation rows.",
        },
        "operating_point_selection": {
            "selected_on": "policy_validation_subset",
            "target_attack_recall": target_recall,
            "target_recall_met_on_validation": scores[selected]["validation"]["target_recall_met"],
            "attack_alert_threshold": bundle["threshold"],
            "benign_review_cutoff": bundle["review_threshold"],
            "minimum_confidence_for_suppression": bundle["minimum_confidence"],
        },
    }
    metrics["selected_model"] = selected
    metrics["split_rows"] = {name: int(len(part)) for name, part in splits.items()}
    metrics["data_rows_after_cleaning"] = int(len(processed))
    metrics["training_rows_used"] = int(len(X_train))
    metrics["calibration_rows"] = int(len(X_calibration))
    metrics["policy_validation_rows"] = int(len(X_operating))
    metrics["target_definition"] = {"0": "benign", "1": "attack"}
    metrics["rate_definitions"] = {
        "false_positive_rate": "FP / (FP + TN): actual benign flows incorrectly classified as high-priority attacks; review rows are reported separately.",
        "false_negative_rate": "FN / (FN + TP): actual attack flows not classified as high-priority; review and suppression are separated in counts.",
        "attention_false_positive_rate": "False positives among all rows requiring analyst attention (high priority or review), divided by all actual benign rows.",
        "confusion_matrix": "Policy matrix rows are actual [benign, attack]; columns are predicted high-priority [benign, attack]. Attention matrix also counts reviews as surfaced.",
    }
    metrics["timing_seconds"] = {
        "selected_model_fit": scores[selected]["fit_seconds"],
        "selected_model_calibration": scores[selected]["calibration_seconds"],
        "uncalibrated_test_batch_inference": raw_inference_seconds,
        "test_batch_inference": inference_seconds,
        "test_inference_per_row": inference_seconds / max(len(X_test), 1),
        "training_and_evaluation_total": time.perf_counter() - started,
    }
    metrics["training_note"] = (
        "Candidate fitting used training rows only. Validation was split into disjoint calibration and policy-validation subsets. "
        "Model/threshold selection used policy-validation rows; test rows were held out until final comparison."
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"run_{timestamp}_seed{seed}"
    metrics["run_id"] = run_id
    root = Path(output_dir)
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    bundle["run_id"] = run_id
    bundle["dataset_version"] = dataset_version
    bundle["selected_features"] = X_train.columns.tolist()
    plot_paths = _save_plots(y_test, test_probability, bundle["threshold"], run_dir, run_id)
    plot_paths["reliability_diagram"] = _save_calibration_plot(
        y_test, raw_test_probability, test_probability, run_dir, ece_bins
    )
    plot_paths["policy_tradeoff"] = _save_policy_tradeoff_plot(
        {name: values for name, values in metrics["decision_comparison"].items()}, run_dir
    )
    metrics["policy_decisions_file"] = f"runs/{run_id}/policy_decisions.csv"
    metrics["plots"] = {name: f"runs/{run_id}/{filename}" for name, filename in plot_paths.items()}
    if generate_explanations:
        from .explain import generate_explanation_artifacts
        explanation_manifest = generate_explanation_artifacts(
            bundle, X_test, test_decisions, run_dir,
            explanation_sample_size=explanation_sample_size,
            top_k=explanation_top_k, seed=seed, max_dense_mb=explanation_max_dense_mb,
        )
        explanation_manifest["artifact_dir"] = f"runs/{run_id}"
    else:
        explanation_manifest = {"status": "disabled", "reason": "Explanation generation is disabled by configuration."}
    metrics["explanations"] = explanation_manifest
    comparison = {
        name: {"validation": result["validation"], "fit_seconds": result["fit_seconds"],
               "calibration_seconds": result["calibration_seconds"]}
        for name, result in scores.items()
    }
    estimator_params = {name: estimator.get_params(deep=True) for name, estimator in candidates.items()}
    metadata = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "dataset_rows_before_cleaning": int(len(frame)),
        "dataset_rows_after_cleaning": int(len(processed)),
        "collection_structure": collection_note or (
            "Rows were stratified at flow level. No verified timestamp, capture-day, or source-file group field was supplied; "
            "a temporal/group holdout could not be constructed."
        ),
        "selected_features": X_train.columns.tolist(),
        "model_parameters": estimator_params,
        "selected_model": selected,
        "training_configuration": {
            "seed": seed, "test_size": test_size, "validation_size": validation_size,
            "target_recall": target_recall, "review_threshold": review_threshold,
            "max_train_rows": max_train_rows, "calibration_method": calibration_method,
            "calibration_fraction_of_validation": calibration_fraction,
            "minimum_confidence": minimum_confidence, "ece_bins": ece_bins,
            "min_isotonic_samples": min_isotonic_samples,
            "explanation_background_size": explanation_background_size,
            "explanation_sample_size": explanation_sample_size,
            "explanation_top_k": explanation_top_k,
            "explanation_max_dense_mb": explanation_max_dense_mb,
            "explanations_enabled": generate_explanations,
            "models_compared": list(candidates), "class_imbalance": "class-weighted estimators; no resampling",
            "threshold_selection": "highest validation threshold meeting target recall",
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                     "joblib": _package_version("joblib")},
        "split_metadata": split_report,
    }

    # Store immutable run artifacts as well as stable latest paths used by API.
    joblib.dump(bundle, run_dir / "model.joblib")
    joblib.dump(bundle["preprocessor"], run_dir / "preprocessor.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "preprocessing_metadata.json").write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
    (run_dir / "split_metadata.json").write_text(json.dumps(split_report, indent=2), encoding="utf-8")
    decision_frame = pd.DataFrame(test_decisions)
    decision_frame.insert(0, "actual_class", np.where(y_test == 1, "attack", "benign"))
    decision_frame.insert(0, "record_number", np.arange(len(test_decisions)))
    decision_frame.to_csv(run_dir / "policy_decisions.csv", index=False)
    joblib.dump(bundle, root / "model.joblib")
    joblib.dump(bundle["preprocessor"], root / "preprocessor.joblib")
    for filename, payload in (("metrics.json", metrics), ("comparison.json", comparison),
                              ("metadata.json", metadata), ("preprocessing_metadata.json", quality_report),
                              ("split_metadata.json", split_report)):
        (root / filename).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for name, filename in plot_paths.items():
        import shutil
        shutil.copy2(run_dir / filename, root / filename)
    for artifact in run_dir.glob("shap_*"):
        import shutil
        shutil.copy2(artifact, root / artifact.name)
    LOG.info("Completed run %s; selected %s; artifacts at %s", run_id, selected, run_dir)
    return metrics
