"""Small, testable helpers used by the Streamlit analyst dashboard.

The dashboard delegates all scoring to FastAPI. This module only handles UI
input parsing, bounded batches, and calls to the service.
"""
from __future__ import annotations

import csv
import io
import json
import math
from typing import Any

import requests

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_BATCH_ROWS = 100
REQUEST_TIMEOUT_SECONDS = 120


class DashboardInputError(ValueError):
    """An uploaded or manually entered flow did not match the model contract."""


def alert_queue_category(result: dict[str, Any]) -> str:
    """Group a score by the operational policy decision, not the 0.5 argmax.

    The validation-selected attack threshold can be below 0.5, so a benign
    argmax prediction may still correctly be escalated as an attack alert.
    """
    decision = result.get("policy_decision")
    return {
        "alert_attack": "Attack alert",
        "suppress_benign": "Benign prediction",
        "review": "Uncertain / review",
    }.get(decision, "Uncategorized")


def parse_flow_text(text: str, features: list[dict[str, Any]]) -> dict[str, Any]:
    """Parse one JSON object and validate it against the model feature schema."""
    try:
        row = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DashboardInputError(f"JSON could not be parsed: {exc.msg} (line {exc.lineno}).") from exc
    if not isinstance(row, dict):
        raise DashboardInputError("Enter one JSON object containing feature names and values.")
    validate_feature_row(row, features)
    return row


def validate_feature_row(row: dict[str, Any], features: list[dict[str, Any]]) -> None:
    expected = {feature["name"]: feature.get("type", "unknown") for feature in features}
    missing = [name for name in expected if name not in row]
    extra = [name for name in row if name not in expected]
    problems = []
    if missing:
        problems.append("missing features: " + ", ".join(missing[:8]))
    if extra:
        problems.append("unexpected features: " + ", ".join(extra[:8]))
    for name, value in row.items():
        kind = expected.get(name)
        if value is None or isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
            problems.append(f"{name} must have a non-empty value")
        elif isinstance(value, (int, float)) and not math.isfinite(float(value)):
            problems.append(f"{name} must be finite")
        elif kind == "numeric" and not isinstance(value, (int, float)):
            problems.append(f"{name} must be numeric")
        elif kind == "categorical" and not isinstance(value, str):
            problems.append(f"{name} must be a category string")
        elif not isinstance(value, (str, int, float)):
            problems.append(f"{name} must be a number or category string")
    if problems:
        raise DashboardInputError("; ".join(problems))


def parse_uploaded_records(filename: str, content: bytes,
                           features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse JSON/CSV upload with strict size and record-count limits."""
    if len(content) > MAX_UPLOAD_BYTES:
        raise DashboardInputError(f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DashboardInputError("Upload must be UTF-8 encoded.") from exc
    try:
        if suffix == "json":
            payload = json.loads(decoded)
            rows = [payload] if isinstance(payload, dict) else payload
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise DashboardInputError("JSON upload must contain an object or a list of objects.")
        elif suffix == "csv":
            reader = csv.DictReader(io.StringIO(decoded, newline=""))
            if not reader.fieldnames:
                raise DashboardInputError("CSV has no header row.")
            if len(set(reader.fieldnames)) != len(reader.fieldnames) or None in reader.fieldnames:
                raise DashboardInputError("CSV contains duplicate or malformed column names.")
            rows = []
            for index, row in enumerate(reader, start=1):
                if index > MAX_BATCH_ROWS:
                    raise DashboardInputError(f"Upload contains more than {MAX_BATCH_ROWS} rows.")
                if None in row or any(value is None for value in row.values()):
                    raise DashboardInputError(f"CSV row {index} has a different number of fields than its header.")
                rows.append({key: _coerce_csv_value(value) for key, value in row.items()})
        else:
            raise DashboardInputError("Choose a .csv or .json file.")
    except (json.JSONDecodeError, csv.Error) as exc:
        raise DashboardInputError(f"Upload could not be parsed: {exc}.") from exc
    if not rows:
        raise DashboardInputError("Upload contains no flow records.")
    if len(rows) > MAX_BATCH_ROWS:
        raise DashboardInputError(f"Upload contains more than {MAX_BATCH_ROWS} rows.")
    for index, row in enumerate(rows, start=1):
        try:
            validate_feature_row(row, features)
        except DashboardInputError as exc:
            raise DashboardInputError(f"Row {index}: {exc}") from exc
    return rows


def _coerce_csv_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    lowered = stripped.lower()
    if lowered in {"nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        # Keep the token intact so validation can report the affected field.
        return float(stripped) if lowered != "nan" else float("nan")
    try:
        return float(stripped) if any(char in stripped for char in ".eE") else int(stripped)
    except ValueError:
        return stripped


def get_json(api_url: str, path: str, timeout: int = 8) -> dict[str, Any]:
    response = requests.get(f"{api_url.rstrip('/')}{path}", timeout=timeout)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError(f"Backend returned an invalid response for {path}.")
    return result


def predict_flow(api_url: str, row: dict[str, Any], explain: bool = True) -> dict[str, Any]:
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json={"features": row, "include_explanation": explain},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code == 422:
        detail = response.json().get("detail", {})
        messages = [item.get("message", "Invalid feature.")
                    for item in detail.get("feature_validation_errors", [])]
        raise DashboardInputError("Backend rejected the feature schema: " + ("; ".join(messages) or detail.get("message", "check the supplied fields")))
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or "policy_decision" not in result:
        raise ValueError("Backend prediction response is missing expected result fields.")
    return result


def predict_records(api_url: str, rows: list[dict[str, Any]], explain: bool = True) -> list[dict[str, Any]]:
    """Score a bounded sequence one at a time; only FastAPI performs inference."""
    if len(rows) > MAX_BATCH_ROWS:
        raise DashboardInputError(f"At most {MAX_BATCH_ROWS} records can be analyzed at once.")
    results = []
    for row in rows:
        results.append(predict_flow(api_url, row, explain=explain))
    return results
