"""SHAP explanations for the exact persisted NIDS feature/model pipeline.

Explanations target the fitted base classifier before probability calibration.
This avoids attributing opaque calibration-wrapper internals and preserves the
same preprocessing and classifier used by the calibrated prediction bundle.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

LOG = logging.getLogger(__name__)


def _model_parts(bundle: dict):
    pipeline = bundle.get("uncalibrated_model")
    if pipeline is None:
        raise ValueError("Saved model bundle has no uncalibrated training pipeline. Retrain to create compatible SHAP artifacts.")
    if not hasattr(pipeline, "named_steps") or "preprocess" not in pipeline.named_steps or "classifier" not in pipeline.named_steps:
        raise ValueError("Saved base model is not the supported fitted preprocessing/classifier pipeline")
    preprocessor = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["classifier"]
    return pipeline, preprocessor, classifier


def _feature_names(bundle: dict, preprocessor, transformed_width: int) -> list[str]:
    if not hasattr(preprocessor, "get_feature_names_out"):
        raise ValueError("Fitted preprocessing pipeline does not expose transformed feature names")
    names = list(map(str, preprocessor.get_feature_names_out()))
    if len(names) != transformed_width:
        raise ValueError(
            f"Transformed feature count ({transformed_width}) differs from preprocessing schema ({len(names)})"
        )
    expected_raw = list(bundle.get("features", []))
    if not expected_raw:
        raise ValueError("Saved model bundle is missing the raw feature schema")
    return names


def _source_feature_map(preprocessor, transformed_names: list[str]) -> dict[str, str]:
    """Map exact transformed names back to source input columns when possible."""
    mapping: dict[str, str] = {}
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder" or transformer == "drop":
            continue
        if isinstance(columns, slice):
            continue
        source_columns = [str(column) for column in columns]
        if transformer_name == "numeric":
            for source in source_columns:
                candidate = f"numeric__{source}"
                if candidate in transformed_names:
                    mapping[candidate] = source
        elif transformer_name == "categorical":
            encoder = transformer.named_steps.get("encode")
            if encoder is not None:
                for encoded in encoder.get_feature_names_out(source_columns):
                    candidate = f"categorical__{encoded}"
                    if candidate in transformed_names:
                        source = next((column for column in source_columns
                                       if encoded == column or encoded.startswith(column + "_")), encoded)
                        mapping[candidate] = source
    for name in transformed_names:
        mapping.setdefault(name, name)
    return mapping


def _as_dense(matrix, max_dense_mb: float):
    if not sparse.issparse(matrix):
        return np.asarray(matrix)
    required_bytes = int(matrix.shape[0]) * int(matrix.shape[1]) * np.dtype(np.float64).itemsize
    if required_bytes > max_dense_mb * 1024 * 1024:
        raise MemoryError(
            f"Tree SHAP would need about {required_bytes / 1024**2:.1f} MiB of dense features, "
            f"above the configured {max_dense_mb:.1f} MiB limit. Reduce background/explanation samples or raise the limit."
        )
    return matrix.toarray()


def _build_explainer(bundle: dict, background: pd.DataFrame, max_dense_mb: float = 512.0):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is not installed. Install the project dependencies with `python -m pip install -e .`.") from exc
    _, preprocessor, classifier = _model_parts(bundle)
    names = _feature_names(bundle, preprocessor, len(preprocessor.get_feature_names_out()))
    transformed_background = preprocessor.transform(background[bundle["features"]])
    model_name = bundle.get("model_name", "")
    classes = np.asarray(classifier.classes_)
    positive_indexes = np.flatnonzero(classes == 1)
    if len(positive_indexes) != 1:
        raise ValueError(f"The fitted classifier does not contain positive class 1: {classes.tolist()}")
    positive_index = int(positive_indexes[0])

    if model_name == "random_forest" and isinstance(classifier, RandomForestClassifier):
        dense_background = _as_dense(transformed_background, max_dense_mb)
        explainer = shap.TreeExplainer(
            classifier, data=dense_background, feature_perturbation="interventional",
            model_output="probability", feature_names=names,
        )
        output_space = "raw_classifier_probability"
    elif model_name == "logistic_regression" and isinstance(classifier, LogisticRegression):
        # LinearExplainer returns additive log-odds contributions for sklearn LR.
        explainer = shap.LinearExplainer(
            classifier, transformed_background, feature_perturbation="interventional",
        )
        output_space = "raw_classifier_log_odds"
    else:
        raise NotImplementedError(
            f"No model-specific SHAP explainer is registered for bundle model {model_name!r} "
            f"({type(classifier).__name__}). Add an explainer adapter for this estimator."
        )
    source_map = _source_feature_map(preprocessor, names)
    return explainer, names, source_map, positive_index, output_space, preprocessor


def _extract_values(explanation: Any, positive_index: int, expected_rows: int,
                    expected_features: int) -> tuple[np.ndarray, float]:
    values = getattr(explanation, "values", explanation)
    if isinstance(values, list):
        if positive_index >= len(values):
            raise ValueError("SHAP returned fewer output arrays than the classifier has classes")
        values = values[positive_index]
    values = np.asarray(values)
    if values.ndim == 3:
        if values.shape == (expected_rows, expected_features, 2) or values.shape[0] == expected_rows:
            values = values[:, :, positive_index]
        elif values.shape[1:] == (expected_rows, expected_features):
            values = values[positive_index]
    if values.shape != (expected_rows, expected_features):
        raise ValueError(
            f"Unexpected SHAP values shape {values.shape}; expected {(expected_rows, expected_features)}"
        )
    base = getattr(explanation, "base_values", None)
    if base is None:
        expected = getattr(explanation, "expected_value", 0.0)
        base_values = np.asarray(expected)
    else:
        base_values = np.asarray(base)
    if base_values.ndim == 0:
        base_value = float(base_values)
    elif base_values.ndim == 1:
        base_value = float(base_values[positive_index]) if len(base_values) > positive_index else float(base_values[0])
    elif base_values.ndim >= 2:
        base_value = float(base_values.reshape(expected_rows, -1)[0, positive_index])
    else:
        base_value = 0.0
    return values.astype(float), base_value


def explain_many(bundle: dict, rows: pd.DataFrame, top_k: int = 15,
                 max_dense_mb: float = 512.0) -> list[dict]:
    """Explain a bounded set of rows with signed contributions and exact names."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if rows.empty:
        return []
    features = list(bundle.get("features", []))
    if not features:
        raise ValueError("Saved model bundle is missing its input feature list")
    missing = [feature for feature in features if feature not in rows.columns]
    if missing:
        raise ValueError(f"Explanation rows are missing trained features: {missing}")
    ordered = rows.loc[:, features].copy()
    background = bundle.get("background")
    if not isinstance(background, pd.DataFrame) or background.empty:
        raise ValueError("Model bundle has no training-only SHAP background sample")
    explainer, names, source_map, positive_index, output_space, preprocessor = _build_explainer(
        bundle, background, max_dense_mb
    )
    estimated_values_mb = len(ordered) * len(names) * np.dtype(np.float64).itemsize / 1024**2
    if estimated_values_mb > max_dense_mb:
        raise MemoryError(
            f"SHAP values would need about {estimated_values_mb:.1f} MiB, "
            f"above the configured {max_dense_mb:.1f} MiB limit. Reduce explanation_sample_size."
        )
    transformed = preprocessor.transform(ordered)
    if bundle.get("model_name") == "random_forest":
        transformed = _as_dense(transformed, max_dense_mb)
    explanation = explainer(transformed)
    shap_values, base_value = _extract_values(explanation, positive_index, len(ordered), len(names))
    output = []
    for row_index in range(len(ordered)):
        contribution = shap_values[row_index]
        order = np.argsort(np.abs(contribution))[::-1][:top_k]
        contributions = []
        for index in order:
            transformed_value = transformed[row_index, index]
            if sparse.issparse(transformed):
                transformed_value = transformed[row_index, index]
            else:
                transformed_value = np.asarray(transformed)[row_index, index]
            value = float(transformed_value) if np.isscalar(transformed_value) and np.isfinite(transformed_value) else None
            shap_value = float(contribution[index])
            contributions.append({
                "feature": names[index],
                "transformed_feature": names[index],
                "source_feature": source_map.get(names[index], names[index]),
                "transformed_value": value,
                "shap_value": shap_value,
                "direction": "toward_attack" if shap_value > 0 else "away_from_attack" if shap_value < 0 else "neutral",
            })
        output.append({
            "method": "SHAP",
            "model_name": bundle.get("model_name"),
            "output_space": output_space,
            "positive_class": "attack",
            "base_value": base_value,
            "shap_value_sum": float(contribution.sum()),
            "explained_output": float(base_value + contribution.sum()),
            "transformed_feature_count": len(names),
            "transformed_feature_names": names,
            "contributions": contributions,
            "interpretation_note": (
                "Positive values move the base classifier output toward attack; negative values move it away. "
                "These are feature attributions for this model and background, not causal effects. "
                "Probability calibration is applied after this explained base-model output."
            ),
        })
    return output


def explain(bundle: dict, row: dict, top_k: int = 8,
            max_dense_mb: float = 512.0) -> dict:
    """Explain one raw-feature flow; return graceful status instead of failing scoring."""
    try:
        features = bundle["features"]
        sample = pd.DataFrame([{name: row.get(name) for name in features}], columns=features)
        return explain_many(bundle, sample, top_k=top_k, max_dense_mb=max_dense_mb)[0]
    except Exception as exc:  # Scoring must remain available when SHAP cannot explain.
        LOG.warning("SHAP explanation unavailable: %s", exc, exc_info=True)
        return {
            "method": "SHAP",
            "error": str(exc),
            "contributions": [],
            "interpretation_note": "Explanation is unavailable; prediction was produced by the saved model bundle.",
        }


def generate_explanation_artifacts(bundle: dict, rows: pd.DataFrame,
                                   decisions: list[dict], output_dir: str | Path,
                                   explanation_sample_size: int = 100,
                                   top_k: int = 15,
                                   seed: int = 42,
                                   max_dense_mb: float = 512.0) -> dict:
    """Create bounded global importance, representative local explanations, and plots."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if explanation_sample_size < 1:
        raise ValueError("explanation_sample_size must be at least 1")
    if len(rows) != len(decisions):
        raise ValueError("Explanation rows and policy decisions must have identical lengths")
    if rows.empty:
        return {"status": "unavailable", "reason": "No rows were provided for explanation."}
    selected_indices: list[int] = []
    # Include one true example of each available policy state first.
    state_names = {"suppress_benign": "benign", "alert_attack": "attack", "review": "uncertain"}
    for decision_name in ("suppress_benign", "alert_attack", "review"):
        index = next((i for i, item in enumerate(decisions)
                      if item.get("policy_decision") == decision_name), None)
        if index is not None and index not in selected_indices:
            selected_indices.append(index)
    remaining = [index for index in range(len(rows)) if index not in selected_indices]
    rng = np.random.default_rng(seed)
    if len(selected_indices) < explanation_sample_size and remaining:
        take = min(explanation_sample_size - len(selected_indices), len(remaining))
        selected_indices.extend(rng.choice(remaining, size=take, replace=False).tolist())
    selected_indices = selected_indices[:explanation_sample_size]

    try:
        selected_rows = rows.iloc[selected_indices].reset_index(drop=True)
        # Keep every transformed feature for an unbiased global mean-|SHAP| summary.
        explanations = explain_many(bundle, selected_rows, top_k=10**9, max_dense_mb=max_dense_mb)
        enriched_locals = []
        for index, explanation in zip(selected_indices, explanations):
            record = dict(explanation)
            record["example_type"] = state_names.get(decisions[index].get("policy_decision"), "sample")
            record["record_number"] = int(index)
            record["decision"] = decisions[index]
            record["contributions"] = record["contributions"][:top_k]
            enriched_locals.append(record)

        # Summarize mean absolute SHAP magnitude over bounded explanations.
        importance: dict[str, dict[str, Any]] = {}
        for explanation in explanations:
            for item in explanation["contributions"]:
                name = item["transformed_feature"]
                record = importance.setdefault(name, {"feature": name, "source_feature": item["source_feature"],
                                                       "sum_abs": 0.0, "sum_signed": 0.0, "observations": 0})
                record["sum_abs"] += abs(item["shap_value"])
                record["sum_signed"] += item["shap_value"]
                record["observations"] += 1
        global_items = sorted(({
            **item,
            "mean_absolute_shap": item["sum_abs"] / item["observations"],
            "mean_signed_shap": item["sum_signed"] / item["observations"],
        } for item in importance.values()), key=lambda item: item["mean_absolute_shap"], reverse=True)
        global_report = {
            "method": "SHAP",
            "model_name": bundle.get("model_name"),
            "output_space": explanations[0]["output_space"],
            "background_rows": int(len(bundle["background"])),
            "explained_rows": int(len(explanations)),
            "transformed_feature_names": explanations[0]["transformed_feature_names"],
            "feature_importance": global_items,
            "interpretation_note": explanations[0]["interpretation_note"],
        }
        _save_global_plot(global_items, target / "shap_global_importance.png")
        _save_local_plots(enriched_locals, target)
        import json
        (target / "shap_global_importance.json").write_text(json.dumps(global_report, indent=2), encoding="utf-8")
        (target / "shap_local_examples.json").write_text(json.dumps(enriched_locals, indent=2), encoding="utf-8")
        manifest = {
            "status": "complete", "method": "SHAP",
            "model_name": bundle.get("model_name"),
            "background_sample_size": int(len(bundle["background"])),
            "explanation_sample_size": int(len(explanations)),
            "available_example_types": sorted({item["example_type"] for item in enriched_locals}),
            "global_importance_file": "shap_global_importance.json",
            "local_examples_file": "shap_local_examples.json",
            "global_plot": "shap_global_importance.png",
            "local_plots": [path.name for path in target.glob("shap_local_*.png")],
        }
        (target / "shap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    except Exception as exc:
        LOG.warning("Could not generate SHAP artifacts: %s", exc, exc_info=True)
        import json
        manifest = {"status": "unavailable", "method": "SHAP", "reason": str(exc),
                    "model_name": bundle.get("model_name"), "explanation_sample_size": 0}
        (target / "shap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


def _save_global_plot(items: list[dict], path: Path, top_k: int = 20) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shown = items[:top_k]
    fig, axis = plt.subplots(figsize=(9, max(4, 0.35 * len(shown))))
    labels = [item["feature"] for item in reversed(shown)]
    values = [item["mean_absolute_shap"] for item in reversed(shown)]
    axis.barh(labels, values, color="#3b82f6")
    axis.set(xlabel="Mean absolute SHAP value", title="Global feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_local_plots(explanations: list[dict], output_dir: Path, top_k: int = 12) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for position, explanation in enumerate(explanations):
        items = explanation["contributions"][:top_k]
        if not items:
            continue
        items = list(reversed(items))
        labels = [item["transformed_feature"] for item in items]
        values = [item["shap_value"] for item in items]
        colors = ["#dc2626" if value > 0 else "#2563eb" for value in values]
        fig, axis = plt.subplots(figsize=(9, max(4, 0.35 * len(items))))
        axis.barh(labels, values, color=colors)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set(xlabel=f"SHAP value ({explanation['output_space']})",
                 title=f"Local explanation: {explanation['example_type']}")
        fig.tight_layout()
        fig.savefig(output_dir / f"shap_local_{position}_{explanation['example_type']}.png", dpi=140)
        plt.close(fig)
