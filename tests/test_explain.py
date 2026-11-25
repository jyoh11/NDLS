"""SHAP adapter output contract tests using a small fitted project pipeline."""
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from nids.explain import explain, explain_many, generate_explanation_artifacts
from nids.preprocessing import build_preprocessor


class _FakeLinearExplainer:
    """Controlled SHAP-compatible object to test naming/shape without SHAP runtime."""

    def __init__(self, model, background, feature_perturbation):
        self.feature_count = background.shape[1]

    def __call__(self, values):
        rows = values.shape[0]
        signed = np.tile(np.arange(1, self.feature_count + 1, dtype=float), (rows, 1))
        signed[:, 0] *= -1
        return SimpleNamespace(values=signed, base_values=np.zeros(rows))


class _FakeTreeExplainer:
    def __init__(self, model, data, feature_perturbation, model_output, feature_names):
        self.feature_count = data.shape[1]

    def __call__(self, values):
        rows = values.shape[0]
        signed = np.zeros((rows, self.feature_count, 2), dtype=float)
        signed[:, 0, 1] = -0.1
        if self.feature_count > 1:
            signed[:, 1, 1] = 0.2
        base = np.tile([0.4, 0.6], (rows, 1))
        return SimpleNamespace(values=signed, base_values=base)


def fitted_bundle():
    train = pd.DataFrame({"duration": np.arange(40, dtype=float),
                          "protocol": ["tcp", "udp"] * 20})
    target = np.array([0] * 20 + [1] * 20)
    preprocessor = build_preprocessor(train, scale_numeric=False)
    transformed = preprocessor.fit_transform(train)
    classifier = LogisticRegression(random_state=3).fit(transformed, target)
    pipeline = Pipeline([("preprocess", preprocessor), ("classifier", classifier)])
    return {"uncalibrated_model": pipeline, "features": train.columns.tolist(),
            "background": train.iloc[:12].copy(), "model_name": "logistic_regression"}, train


def install_fake_shap(monkeypatch):
    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(LinearExplainer=_FakeLinearExplainer))


def test_local_explanation_preserves_transformed_feature_names(monkeypatch):
    install_fake_shap(monkeypatch)
    bundle, rows = fitted_bundle()
    item = explain_many(bundle, rows.iloc[[0]], top_k=8)[0]
    expected = list(bundle["uncalibrated_model"].named_steps["preprocess"].get_feature_names_out())
    assert item["transformed_feature_names"] == expected
    assert {contribution["transformed_feature"] for contribution in item["contributions"]} == set(expected)
    assert any(contribution["direction"] == "toward_attack" for contribution in item["contributions"])
    assert any(contribution["direction"] == "away_from_attack" for contribution in item["contributions"])
    assert item["output_space"] == "raw_classifier_log_odds"


def test_random_forest_uses_tree_shap_probability_output(monkeypatch):
    monkeypatch.setitem(sys.modules, "shap", SimpleNamespace(TreeExplainer=_FakeTreeExplainer))
    bundle, rows = fitted_bundle()
    pipeline = bundle["uncalibrated_model"]
    transformed = pipeline.named_steps["preprocess"].transform(rows)
    classifier = RandomForestClassifier(n_estimators=4, random_state=2).fit(
        transformed, np.array([0] * 20 + [1] * 20)
    )
    pipeline = Pipeline([("preprocess", pipeline.named_steps["preprocess"]), ("classifier", classifier)])
    bundle["uncalibrated_model"] = pipeline
    bundle["model_name"] = "random_forest"
    item = explain_many(bundle, rows.iloc[[0]], top_k=8)[0]
    assert item["output_space"] == "raw_classifier_probability"
    assert item["transformed_feature_names"] == list(pipeline.named_steps["preprocess"].get_feature_names_out())
    assert {entry["direction"] for entry in item["contributions"]} >= {"toward_attack", "away_from_attack"}


def test_single_flow_explanation_has_structured_graceful_contract(monkeypatch):
    install_fake_shap(monkeypatch)
    bundle, rows = fitted_bundle()
    item = explain(bundle, rows.iloc[0].to_dict(), top_k=2)
    assert item["method"] == "SHAP"
    assert len(item["contributions"]) == 2
    assert item["contributions"][0]["feature"] == item["contributions"][0]["transformed_feature"]


def test_global_and_local_artifacts_include_policy_example_types(tmp_path, monkeypatch):
    install_fake_shap(monkeypatch)
    bundle, rows = fitted_bundle()
    decisions = [
        {"policy_decision": "suppress_benign", "status": "benign"},
        {"policy_decision": "alert_attack", "status": "high_priority"},
        {"policy_decision": "review", "status": "review"},
    ] + [{"policy_decision": "review", "status": "review"}] * (len(rows) - 3)
    manifest = generate_explanation_artifacts(bundle, rows, decisions, tmp_path,
                                              explanation_sample_size=5, seed=17)
    assert manifest["status"] == "complete"
    assert set(manifest["available_example_types"]) == {"benign", "attack", "uncertain"}
    assert (tmp_path / manifest["global_importance_file"]).exists()
    assert (tmp_path / manifest["local_examples_file"]).exists()
    assert (tmp_path / manifest["global_plot"]).exists()


def test_missing_shap_returns_error_without_breaking_prediction(monkeypatch):
    monkeypatch.setitem(sys.modules, "shap", None)
    bundle, rows = fitted_bundle()
    result = explain(bundle, rows.iloc[0].to_dict())
    assert result["method"] == "SHAP"
    assert result["error"]
    assert result["contributions"] == []
