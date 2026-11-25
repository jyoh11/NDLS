"""Offline integration from CSV ingestion through training and API inference."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from nids import api
from nids.inference import score_flow
from nids.ingest import inspect_dataset
from nids.training import load_dataset, train_and_evaluate


def test_csv_to_trained_bundle_cli_and_api_share_prediction_contract(tmp_path, monkeypatch):
    rng = np.random.default_rng(181)
    count = 360
    labels = np.resize(np.array(["Benign", "DoS", "DDoS"], dtype=object), count)
    rng.shuffle(labels)
    attack = labels != "Benign"
    dataset = pd.DataFrame({
        "duration": rng.lognormal(np.where(attack, 3.8, 2.7), 0.55),
        "protocol": np.where(attack, "udp", "tcp"),
        "Flow ID": [f"flow-{index}" for index in range(count)],
        "Attack Description": np.where(attack, "test annotation", "normal annotation"),
        "Label": labels,
    })
    csv_path = tmp_path / "flows.csv"
    dataset.to_csv(csv_path, index=False)

    intake = inspect_dataset(csv_path, label_column="Label", chunksize=37)
    assert intake["row_count"] == count
    assert "label" in intake["candidate_label_columns"]
    frame = load_dataset(csv_path, label_column="Label", chunksize=41, max_rows=None, seed=181)
    artifacts = tmp_path / "artifacts"
    report = train_and_evaluate(
        frame, artifacts, label_column="Label", seed=181, target_recall=0.8,
        max_train_rows=None, model_names=["random_forest"],
        model_parameters={"random_forest": {"n_estimators": 18, "min_samples_leaf": 2, "n_jobs": 1}},
        dataset_version="synthetic-integration-fixture", explanation_background_size=8,
        explanation_sample_size=4, generate_explanations=False,
    )
    bundle = joblib.load(artifacts / "model.joblib")
    assert bundle["features"] == ["duration", "protocol"]
    assert not {"label", "binary_target", "original_label", "normalized_attack_label",
                "flow_id", "attack_description"} & set(bundle["features"])
    assert report["run_id"] == bundle["run_id"]
    assert (artifacts / "preprocessor.joblib").is_file()
    assert (artifacts / "metrics.json").is_file()
    assert (artifacts / "metadata.json").is_file()

    sample = {name: (value.item() if hasattr(value, "item") else value)
              for name, value in dataset.loc[0, ["duration", "protocol"]].items()}
    cli_result = score_flow(bundle, sample, include_explanation=True)
    monkeypatch.setattr(api, "BUNDLE_PATH", artifacts / "model.joblib")
    monkeypatch.setattr(api, "METRICS_PATH", artifacts / "metrics.json")
    monkeypatch.setattr(api, "METADATA_PATH", artifacts / "metadata.json")
    runtime = api.ArtifactRuntime(artifacts / "model.joblib", artifacts / "metrics.json", artifacts / "metadata.json")
    assert runtime.readiness()["status"] == "ready"
    ordered, validation_errors = api._validate_features(sample, runtime.bundle)
    assert validation_errors == []

    # Exercise the API scoring adapter without its HTTP transport. The dedicated
    # endpoint tests cover HTTP status/serialization; this test checks the real
    # trained artifact and shared CLI/API inference path end to end.
    monkeypatch.setattr(api, "explain", lambda bundle, row, top_k, max_dense_mb: {
        "method": "SHAP", "transformed_feature_names": ["numeric__duration"],
        "contributions": [{"transformed_feature": "numeric__duration", "shap_value": 0.1,
                           "direction": "toward_attack"}],
    })
    api_result = api._predict(runtime.bundle, ordered, include_explanation=True)

    for key in ("predicted_class", "policy_decision", "alert_status", "model_version"):
        assert cli_result[key] == api_result[key]
    assert cli_result["calibrated_attack_probability"] == pytest.approx(api_result["calibrated_attack_probability"])
    assert cli_result["confidence_score"] == pytest.approx(api_result["confidence_score"])
    explanation = api_result["explanation"]
    assert explanation and explanation.get("method") == "SHAP"
    if explanation.get("contributions"):
        feature_names = set(explanation.get("transformed_feature_names", []))
        assert all(item["transformed_feature"] in feature_names for item in explanation["contributions"])
        assert all(item["direction"] in {"toward_attack", "away_from_attack", "neutral"}
                   for item in explanation["contributions"])
    else:
        # Optional SHAP dependency/environment failures must be explicit, never faked.
        assert explanation.get("error")
