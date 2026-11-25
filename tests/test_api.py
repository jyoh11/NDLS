"""Endpoint tests with mocked serialized model/report artifacts."""
import json

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from nids import api


class FakeCalibratedModel:
    classes_ = np.array([0, 1])
    calibrated_classifiers_ = [object()]
    calls = 0

    def predict_proba(self, frame):
        type(self).calls += 1
        probability = np.clip(np.asarray(frame["duration"], dtype=float) / 100.0, 0.0, 1.0)
        return np.column_stack([1 - probability, probability])


def write_artifacts(tmp_path, monkeypatch, *, with_model=True, with_metrics=True):
    bundle_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    metadata_path = tmp_path / "metadata.json"
    if with_model:
        joblib.dump({
            "model": FakeCalibratedModel(),
            "features": ["duration", "protocol"],
            "feature_dtypes": {"duration": "numeric", "protocol": "categorical"},
            "threshold": 0.8,
            "review_threshold": 0.5,
            "minimum_confidence": 0.8,
            "calibration_method": "sigmoid",
            "model_name": "random_forest",
            "run_id": "fixture-run-1",
            "dataset_version": "synthetic-fixture",
            "explanation_config": {"top_k": 5, "max_dense_mb": 10},
        }, bundle_path)
    if with_metrics:
        metrics_path.write_text(json.dumps({"accuracy": 0.91, "source_note": "test fixture"}), encoding="utf-8")
    metadata_path.write_text(json.dumps({"dataset_version": "synthetic-fixture"}), encoding="utf-8")
    monkeypatch.setattr(api, "BUNDLE_PATH", bundle_path)
    monkeypatch.setattr(api, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(api, "METADATA_PATH", metadata_path)
    return bundle_path, metrics_path, metadata_path


@pytest.fixture
def client(tmp_path, monkeypatch):
    write_artifacts(tmp_path, monkeypatch)
    with TestClient(api.app) as test_client:
        yield test_client


def valid_payload(duration=90):
    return {"features": {"duration": duration, "protocol": "tcp"}}


def test_health_info_config_and_saved_metrics(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    info = client.get("/model/info").json()
    assert info["model_name"] == "random_forest"
    assert [item["name"] for item in info["features"]] == ["duration", "protocol"]
    assert client.get("/schema").json()["features"] == ["duration", "protocol"]
    safe_config = client.get("/config").json()
    assert safe_config["attack_threshold"] == 0.8
    assert not any("path" in key.lower() or "secret" in key.lower() for key in safe_config)
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["source"] == "saved_held_out_evaluation_report"
    assert metrics.json()["metrics"]["accuracy"] == 0.91


def test_prediction_returns_policy_fields_and_ordered_validated_features(client):
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    result = response.json()
    assert result["predicted_class"] == "attack"
    assert result["attack_label"] == "Attack"
    assert result["calibrated_attack_probability"] == 0.9
    assert result["confidence_score"] == 0.9
    assert result["policy_decision"] == "alert_attack"
    assert result["alert_reason"]
    assert result["feature_validation_errors"] == []
    assert result["explanation"] is None
    assert response.headers.get("x-request-id")


def test_explain_endpoint_uses_same_prediction_and_saved_bundle(client, monkeypatch):
    observed = {}
    def fake_explain(bundle, row, top_k, max_dense_mb):
        observed.update({"model": bundle["model_name"], "features": tuple(row), "top_k": top_k})
        return {"method": "SHAP", "contributions": [{"feature": "numeric__duration", "shap_value": 0.2}]}
    monkeypatch.setattr(api, "explain", fake_explain)
    response = client.post("/explain", json=valid_payload())
    assert response.status_code == 200
    assert response.json()["policy_decision"] == "alert_attack"
    assert response.json()["explanation"]["method"] == "SHAP"
    assert observed == {"model": "random_forest", "features": ("duration", "protocol"), "top_k": 5}


@pytest.mark.parametrize("features,code", [
    ({"duration": 90}, "missing_feature"),
    ({"duration": 90, "protocol": "tcp", "extra": 1}, "unexpected_feature"),
    ({"duration": float("nan"), "protocol": "tcp"}, "non_finite"),
    ({"duration": float("inf"), "protocol": "tcp"}, "non_finite"),
    ({"duration": True, "protocol": "tcp"}, "invalid_type"),
    ({"duration": "ninety", "protocol": "tcp"}, "invalid_type"),
    ({"duration": 90, "protocol": None}, "missing_value"),
    ({"duration": 90, "protocol": ["tcp"]}, "invalid_type"),
])
def test_prediction_rejects_invalid_or_unexpected_features(client, features, code):
    if code == "non_finite":
        # HTTP JSON helpers correctly refuse NaN/Infinity before the request
        # reaches FastAPI. Send the raw token so API-side validation is tested.
        response = client.post(
            "/predict", content=json.dumps({"features": features}, allow_nan=True),
            headers={"Content-Type": "application/json"},
        )
    else:
        response = client.post("/predict", json={"features": features})
    assert response.status_code == 422
    errors = response.json()["detail"]["feature_validation_errors"]
    assert code in {item["code"] for item in errors}


def test_request_model_rejects_unexpected_top_level_fields(client):
    response = client.post("/predict", json={**valid_payload(), "debug_dump": "secret"})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "request_validation_error"
    assert "secret" not in response.text


def test_artifacts_load_once_during_startup(tmp_path, monkeypatch):
    write_artifacts(tmp_path, monkeypatch)
    original_load = api.joblib.load
    calls = []
    def counted_load(path):
        calls.append(str(path))
        return original_load(path)
    monkeypatch.setattr(api.joblib, "load", counted_load)
    with TestClient(api.app) as test_client:
        assert test_client.get("/health").json()["status"] == "ready"
        test_client.post("/predict", json=valid_payload())
        test_client.get("/model/info")
    assert len(calls) == 1


def test_unavailable_model_returns_clear_readiness_and_prediction_error(tmp_path, monkeypatch):
    write_artifacts(tmp_path, monkeypatch, with_model=False, with_metrics=False)
    with TestClient(api.app) as test_client:
        health = test_client.get("/health").json()
        assert health["status"] == "not_ready"
        assert health["artifacts"]["model"] is False
        response = test_client.post("/predict", json=valid_payload())
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "model_not_ready"
        assert test_client.get("/metrics").status_code == 404


def test_artifacts_from_different_runs_are_not_combined(tmp_path, monkeypatch):
    _, metrics_path, metadata_path = write_artifacts(tmp_path, monkeypatch)
    metrics_path.write_text(json.dumps({"accuracy": 0.91, "run_id": "another-run"}), encoding="utf-8")
    metadata_path.write_text(json.dumps({"dataset_version": "synthetic-fixture", "run_id": "another-run"}), encoding="utf-8")
    with TestClient(api.app) as test_client:
        readiness = test_client.get("/health").json()
        assert readiness["status"] == "not_ready"
        assert readiness["errors"]["model"] in {
            "Model bundle and metadata belong to different training runs.",
            "Model bundle and evaluation report belong to different training runs.",
        }


def test_cors_allows_configured_dashboard_origin(client):
    response = client.options("/health", headers={
        "Origin": "http://localhost:8501",
        "Access-Control-Request-Method": "GET",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"
