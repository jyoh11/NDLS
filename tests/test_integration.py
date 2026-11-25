"""Small offline end-to-end training and API integration checks."""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from nids import api
from nids.training import train_and_evaluate

class PredictableModel:
    classes_ = np.array([0, 1])
    calibrated_classifiers_ = [object()]

    def predict_proba(self, frame):
        probability = np.asarray(frame["bytes"]).astype(float) / 100.0
        probability = np.clip(probability, 0, 1)
        return np.column_stack([1 - probability, probability])

def test_training_pipeline_saves_test_metrics_and_bundle(tmp_path):
    rows = 120
    frame = pd.DataFrame({"bytes": np.arange(rows), "protocol": ["tcp", "udp"] * (rows // 2),
                          "Label": ["Benign"] * 60 + ["DDoS"] * 60})
    result = train_and_evaluate(frame, tmp_path, label_column="Label", seed=7,
                                target_recall=0.8, max_train_rows=80,
                                model_parameters={"random_forest": {"n_estimators": 24, "n_jobs": 1}})
    assert result["split_rows"]["test"] > 0
    assert 0 <= result["false_positive_rate"] <= 1
    import json
    comparison = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert "random_forest" in comparison  # RF is always included by default.
    assert result["per_class"]["benign"]["support"] > 0
    assert result["per_class"]["attack"]["support"] > 0
    assert result["rate_definitions"]["false_positive_rate"].startswith("FP / (FP + TN)")
    assert result["timing_seconds"]["test_inference_per_row"] >= 0
    assert result["calibration_rows"] + result["policy_validation_rows"] == result["split_rows"]["validation"]
    assert set(result["decision_comparison"]) == {
        "uncalibrated_classifier_at_0_5", "calibrated_classifier_at_0_5", "confidence_aware_policy"
    }
    assert "expected_calibration_error" in result["calibration_comparison"]["calibrated"]
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "comparison.json").exists()
    assert (tmp_path / result["plots"]["confusion_matrix"]).exists()
    assert (tmp_path / result["plots"]["reliability_diagram"]).exists()
    assert (tmp_path / result["plots"]["policy_tradeoff"]).exists()
    assert (tmp_path / result["policy_decisions_file"]).exists()
    assert result["explanations"]["status"] in {"complete", "unavailable"}
    assert (tmp_path / "shap_manifest.json").exists()
    assert list((tmp_path / "runs").glob("*/model.joblib"))

def test_prediction_endpoint_returns_confidence_tier(tmp_path, monkeypatch):
    import joblib
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": PredictableModel(), "features": ["bytes"], "feature_dtypes": {"bytes": "numeric"},
                 "threshold": 0.8, "review_threshold": 0.5, "minimum_confidence": 0.8,
                 "calibration_method": "sigmoid", "run_id": "test-run", "model_name": "test"}, model_path)
    monkeypatch.setattr(api, "BUNDLE_PATH", model_path)
    monkeypatch.setattr(api, "METRICS_PATH", tmp_path / "missing-metrics.json")
    monkeypatch.setattr(api, "METADATA_PATH", tmp_path / "missing-metadata.json")
    with TestClient(api.app) as client:
        response = client.post("/predict", json={"features": {"bytes": 90}, "include_explanation": False})
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] == "malicious"
        assert body["alert_status"] == "high_priority"
        assert body["confidence_score"] == 0.9
        assert body["calibrated_attack_probability"] == 0.9
        assert body["policy_decision"] == "alert_attack"
        assert body["alert_reason"]
