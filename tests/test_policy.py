from nids.policy import binary_target, confidence_alert
from nids.training import (
    _choose_policy_threshold,
    _choose_threshold, _policy_decisions,
)
import numpy as np
import pandas as pd
import pytest

def test_binary_target_maps_benign_and_attack():
    assert binary_target(pd.Series(["Benign", "DDoS", "Web Attacks"])).tolist() == [0, 1, 1]

def test_confidence_tiers():
    assert confidence_alert(0.1, 0.8)["status"] == "benign"
    assert confidence_alert(0.6, 0.8)["status"] == "review"
    assert confidence_alert(0.9, 0.8)["status"] == "high_priority"

def test_decision_records_class_confidence_threshold_and_reason():
    decision = confidence_alert(0.1, 0.8, review_threshold=0.5, minimum_confidence=0.8)
    assert decision["predicted_class"] == "benign"
    assert decision["attack_probability"] == 0.1
    assert decision["calibrated_confidence"] == 0.9
    assert decision["threshold"] == 0.8
    assert decision["policy_decision"] == "suppress_benign"
    assert decision["reason"]
    assert decision["requires_attention"] is False

def test_uncertain_benign_prediction_goes_to_review():
    decision = confidence_alert(0.4, 0.8, review_threshold=0.5, minimum_confidence=0.8)
    assert decision["predicted_class"] == "benign"
    assert decision["status"] == "review"
    assert decision["requires_attention"] is True

def test_operating_alert_threshold_can_differ_from_argmax_prediction():
    decision = confidence_alert(0.45, 0.4)
    assert decision["policy_decision"] == "alert_attack"
    assert decision["predicted_class"] == "benign"
    assert decision["requires_attention"] is True

@pytest.mark.parametrize("probability", [None, float("nan"), float("inf"), -0.01, 1.01, "bad"])
def test_missing_or_invalid_probabilities_are_rejected(probability):
    with pytest.raises(ValueError):
        confidence_alert(probability, 0.8)

def test_threshold_meets_validation_recall_target():
    threshold = _choose_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.3, 0.6, 0.9]), 0.5)
    # The documented selection is the highest threshold that still meets
    # target recall; 0.9 retains exactly one of the two attack rows (50%).
    assert threshold == 0.9

def test_policy_threshold_selected_on_validation_recall_and_reports_tradeoff():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.8, 0.9])
    threshold = _choose_policy_threshold(y, probabilities, 0.5, 0.5, 0.8)
    assert threshold == 0.9
    summary, decisions = _policy_decisions(np.array([1, 0, 1, 1]), probabilities, 0.9, 0.5, 0.8)
    assert summary["false_positive_count"] == 0
    assert summary["attack_recall"] == pytest.approx(1 / 3)
    assert summary["alert_volume"] == 3  # one alert and two reviews; benign suppression is excluded
    assert summary["suppressed_attack_count"] == 1
    assert decisions[1]["policy_decision"] == "review"
