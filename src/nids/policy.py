"""Binary target mapping and confidence-aware analyst decision policy."""
from __future__ import annotations

import math
import pandas as pd

BENIGN_NAMES = {"benign", "normal", "0"}


def binary_target(labels: pd.Series) -> pd.Series:
    """Map Benign/Normal to zero and every attack category to one."""
    normalized = labels.astype(str).str.strip().str.lower()
    return (~normalized.isin(BENIGN_NAMES)).astype("int8")


def _validated_number(name: str, value, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be between {low} and {high}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be between {low} and {high}") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return number


def confidence_alert(probability: float, threshold: float, review_threshold: float = 0.5,
                     minimum_confidence: float = 0.8) -> dict:
    """Return an auditable alert/review/suppression decision for one flow.

    ``probability`` is the calibrated estimate P(attack | flow) from this model.
    ``calibrated_confidence`` is the calibrated probability assigned to the
    argmax class (``p`` for attack, ``1-p`` for benign). This is a model score
    whose empirical reliability depends on calibration and data remaining
    representative; it is not a universal certainty or an out-of-distribution
    detector.

    An attack probability at/above ``threshold`` becomes an alert. Lower values
    are suppressed as benign only below ``review_threshold`` and when their
    predicted-class probability reaches ``minimum_confidence``. Other cases go
    to review. The review state makes uncertainty explicit.
    """
    if isinstance(probability, bool):
        raise ValueError("Probability must be a finite number between 0 and 1")
    try:
        p = float(probability)
    except (TypeError, ValueError) as exc:
        raise ValueError("Probability must be a finite number between 0 and 1") from exc
    if not math.isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("Probability must be a finite number between 0 and 1")
    attack_threshold = _validated_number("Attack threshold", threshold, 0.0, 1.0)
    benign_review_cutoff = _validated_number("Review threshold", review_threshold, 0.0, 1.0)
    confidence_floor = _validated_number("Minimum confidence", minimum_confidence, 0.5, 1.0)
    predicted_class = "attack" if p >= 0.5 else "benign"
    calibrated_confidence = p if predicted_class == "attack" else 1 - p

    if p >= attack_threshold:
        decision = "alert_attack"
        status = "high_priority"
        reason = f"Attack probability {p:.4f} meets or exceeds alert threshold {attack_threshold:.4f}."
    elif p < benign_review_cutoff and calibrated_confidence >= confidence_floor:
        decision = "suppress_benign"
        status = "benign"
        reason = (
            f"Attack probability {p:.4f} is below benign review cutoff {benign_review_cutoff:.4f} "
            f"and predicted-class probability {calibrated_confidence:.4f} meets confidence floor {confidence_floor:.4f}."
        )
    else:
        decision = "review"
        status = "review"
        reason = "Probability is below the alert threshold but does not qualify for confident benign suppression."

    return {
        "predicted_class": predicted_class,
        "attack_probability": p,
        "calibrated_confidence": float(calibrated_confidence),
        "confidence_definition": "calibrated probability of the argmax class (attack if p>=0.5, else benign)",
        "threshold": attack_threshold,
        "review_threshold": benign_review_cutoff,
        "minimum_confidence": confidence_floor,
        "policy_decision": decision,
        "reason": reason,
        "status": status,
        "predicted_attack": decision == "alert_attack",
        "requires_attention": decision in {"alert_attack", "review"},
    }
