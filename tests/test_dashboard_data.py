"""Tests for bounded dashboard input and its FastAPI integration boundary."""
import json

import pytest

from nids.dashboard_data import (
    DashboardInputError,
    MAX_UPLOAD_BYTES,
    alert_queue_category,
    parse_flow_text,
    parse_uploaded_records,
    predict_flow,
)


FEATURES = [
    {"name": "duration", "type": "numeric", "position": 0},
    {"name": "protocol", "type": "categorical", "position": 1},
]


def test_parse_flow_json_enforces_exact_schema():
    assert parse_flow_text('{"duration": 3, "protocol": "tcp"}', FEATURES) == {
        "duration": 3, "protocol": "tcp"
    }
    with pytest.raises(DashboardInputError, match="missing features"):
        parse_flow_text('{"duration": 3}', FEATURES)
    with pytest.raises(DashboardInputError, match="unexpected features"):
        parse_flow_text('{"duration": 3, "protocol": "tcp", "target": "attack"}', FEATURES)


def test_parse_csv_and_json_with_limits():
    csv_rows = parse_uploaded_records("flows.csv", b"duration,protocol\n4,udp\n", FEATURES)
    assert csv_rows == [{"duration": 4, "protocol": "udp"}]
    json_rows = parse_uploaded_records("flows.json", b'[{"duration":5,"protocol":"tcp"}]', FEATURES)
    assert json_rows == [{"duration": 5, "protocol": "tcp"}]
    with pytest.raises(DashboardInputError, match="UTF-8"):
        parse_uploaded_records("flows.csv", b"\xff", FEATURES)
    with pytest.raises(DashboardInputError, match="limit"):
        parse_uploaded_records("flows.json", b" " * (MAX_UPLOAD_BYTES + 1), FEATURES)


def test_parse_csv_rejects_nonfinite_values():
    with pytest.raises(DashboardInputError, match="finite"):
        parse_uploaded_records("flows.csv", b"duration,protocol\ninf,tcp\n", FEATURES)


def test_prediction_is_sent_to_backend(monkeypatch):
    seen = {}

    class Response:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {"policy_decision": "alert_attack", "predicted_class": "attack"}

    def post(url, json, timeout):
        seen.update(url=url, body=json, timeout=timeout)
        return Response()

    monkeypatch.setattr("nids.dashboard_data.requests.post", post)
    result = predict_flow("http://localhost:8000/", {"duration": 2, "protocol": "tcp"}, explain=True)
    assert result["predicted_class"] == "attack"
    assert seen["url"] == "http://localhost:8000/predict"
    assert seen["body"]["include_explanation"] is True


def test_backend_validation_errors_are_presented_as_input_errors(monkeypatch):
    class Response:
        status_code = 422
        def json(self):
            return {"detail": {"feature_validation_errors": [{"message": "Required feature is missing."}]}}
        def raise_for_status(self):
            pass
    monkeypatch.setattr("nids.dashboard_data.requests.post", lambda *args, **kwargs: Response())
    with pytest.raises(DashboardInputError, match="Required feature is missing"):
        predict_flow("http://localhost:8000", {"duration": 1}, explain=False)


def test_dashboard_alert_queue_uses_policy_decision_when_cutoff_differs_from_argmax():
    # A selected alert threshold below 0.5 can raise an attack policy alert
    # while the classifier's ordinary argmax class remains benign.
    result = {"predicted_class": "benign", "policy_decision": "alert_attack"}
    assert alert_queue_category(result) == "Attack alert"
