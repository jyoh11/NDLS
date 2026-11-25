"""Shared prediction contract used by FastAPI and the ``nids predict`` CLI."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .explain import explain
from .policy import confidence_alert


class PredictionError(RuntimeError):
    """A saved model or policy failed while scoring a validated flow."""

    def __init__(self, message: str, code: str = "prediction_failed"):
        super().__init__(message)
        self.code = code


def validate_features(features: dict[str, object], bundle: dict) -> tuple[dict[str, object], list[dict]]:
    """Validate all supplied fields and return values ordered by saved schema."""
    expected = list(bundle["features"])
    expected_set = set(expected)
    received_set = set(features)
    type_schema = bundle.get("feature_dtypes", {})
    errors: list[dict] = []
    for name in expected:
        if name not in received_set:
            errors.append({"feature": name, "code": "missing_feature", "message": "Required feature is missing."})
    for name in sorted(received_set - expected_set):
        errors.append({"feature": name, "code": "unexpected_feature", "message": "Feature is not in the trained schema."})

    for name in expected:
        if name not in features:
            continue
        value = features[name]
        feature_type = type_schema.get(name)
        if value is None:
            errors.append({"feature": name, "code": "missing_value", "message": "Null values are not accepted."})
        elif isinstance(value, bool):
            errors.append({"feature": name, "code": "invalid_type", "message": "Boolean values are not accepted as flow features."})
        elif isinstance(value, (int, float)):
            try:
                finite = math.isfinite(float(value))
            except (ValueError, OverflowError):
                finite = False
            if not finite:
                errors.append({"feature": name, "code": "non_finite", "message": "Numeric values must be finite."})
            elif feature_type == "categorical":
                errors.append({"feature": name, "code": "invalid_type", "message": "This trained feature requires a string category."})
        elif isinstance(value, str):
            if not value.strip():
                errors.append({"feature": name, "code": "missing_value", "message": "Empty strings are not accepted."})
            elif feature_type == "numeric":
                errors.append({"feature": name, "code": "invalid_type", "message": "This trained feature requires a finite number."})
        else:
            errors.append({"feature": name, "code": "invalid_type", "message": "Feature value must be a finite number or string."})
    return {name: features[name] for name in expected if name in features}, errors


def score_bundle(bundle: dict, ordered_features: dict[str, object],
                 include_explanation: bool = False, explanation_fn=None) -> dict:
    """Score one validated row, apply the saved alert policy, and optionally explain it."""
    frame = pd.DataFrame([ordered_features], columns=bundle["features"])
    try:
        probabilities = np.asarray(bundle["model"].predict_proba(frame), dtype=float)
        classes = np.asarray(getattr(bundle["model"], "classes_", [0, 1]))
        positive = np.flatnonzero(classes == 1)
        if probabilities.ndim != 2 or probabilities.shape[0] != 1 or len(positive) != 1:
            raise ValueError("Calibrated model returned an unexpected probability shape or class schema")
        attack_probability = float(probabilities[0, int(positive[0])])
        if not math.isfinite(attack_probability) or not 0 <= attack_probability <= 1:
            raise ValueError("Calibrated model returned a non-finite or out-of-range attack probability")
    except Exception as exc:
        raise PredictionError("The saved model could not score this feature row.") from exc
    try:
        decision = confidence_alert(
            attack_probability, bundle["threshold"], bundle["review_threshold"],
            bundle.get("minimum_confidence", 0.8),
        )
    except (TypeError, ValueError) as exc:
        raise PredictionError("Saved alert policy configuration is invalid.", "policy_not_ready") from exc
    predicted_class = decision["predicted_class"]
    result = {
        "predicted_class": predicted_class,
        "attack_label": "Attack" if predicted_class == "attack" else "Benign",
        "prediction": "malicious" if predicted_class == "attack" else "benign",
        "calibrated_attack_probability": attack_probability,
        "confidence_score": decision["calibrated_confidence"],
        "confidence_definition": decision["confidence_definition"],
        "policy_decision": decision["policy_decision"],
        "alert_status": decision["status"],
        "alert_reason": decision["reason"],
        "threshold": decision["threshold"],
        "review_threshold": decision["review_threshold"],
        "minimum_confidence": decision["minimum_confidence"],
        "model_name": bundle["model_name"],
        "model_version": bundle.get("run_id", bundle.get("artifact_version", "unknown")),
        "feature_validation_errors": [],
        "explanation": None,
    }
    if include_explanation:
        options = bundle.get("explanation_config", {})
        explain_with = explanation_fn or explain
        result["explanation"] = explain_with(
            bundle, ordered_features,
            top_k=int(options.get("top_k", 8)),
            max_dense_mb=float(options.get("max_dense_mb", 512.0)),
        )
    return result


def score_flow(bundle: dict, features: dict[str, object], include_explanation: bool = False) -> dict:
    """Validate and score a raw named-feature mapping using the saved bundle."""
    ordered, errors = validate_features(features, bundle)
    if errors:
        raise ValueError({"feature_validation_errors": errors})
    return score_bundle(bundle, ordered, include_explanation)
