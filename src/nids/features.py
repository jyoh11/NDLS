"""Column normalization, feature separation, and conservative leakage controls."""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

LABEL_ALIASES = {
    "label", "class", "attack", "category", "target", "attack_type",
    "attack_category", "attack_label", "binary_label", "multiclass_label",
    "normalized_attack_label", "original_label", "is_attack", "binary_target",
}
IDENTIFIER_NAMES = {
    "id", "uid", "row_id", "flow_id", "src_ip", "source_ip", "src_address",
    "dst_ip", "dest_ip", "destination_ip", "dst_address", "src_mac",
    "source_mac", "dst_mac", "destination_mac", "flowid", "sourceip", "srcip",
    "destinationip", "destip", "dstip", "srcmac", "dstmac",
    # NF-CSE-CIC-IDS2018-v2's NetFlow schema uses these endpoint address names.
    "ipv4_src_addr", "ipv4_dst_addr",
}
POST_EVENT_NAMES = {
    "description", "attack_description", "attack_name", "alert", "alert_status",
    "incident", "incident_id", "outcome", "response", "result", "verdict",
    "detected", "detection_result", "post_event_label",
}


def normalize_column_name(name: object) -> str:
    """Return a stable snake-case column name without losing meaningful text."""
    text = unicodedata.normalize("NFKC", str(name)).lstrip("\ufeff").strip()
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    if not normalized:
        raise ValueError(f"Column name {name!r} becomes empty after normalization")
    return normalized


def normalize_frame_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize headers and reject duplicate or colliding names before training."""
    source_names = [str(column) for column in frame.columns]
    normalized_names = [normalize_column_name(column) for column in source_names]
    if len(set(source_names)) != len(source_names):
        duplicates = sorted({name for name in source_names if source_names.count(name) > 1})
        raise ValueError(f"Duplicate source column names are not allowed: {duplicates}")
    if len(set(normalized_names)) != len(normalized_names):
        collisions: dict[str, list[str]] = {}
        for source, normalized in zip(source_names, normalized_names):
            collisions.setdefault(normalized, []).append(source)
        collisions = {name: sources for name, sources in collisions.items() if len(sources) > 1}
        raise ValueError(f"Column names collide after normalization: {collisions}")
    result = frame.copy()
    result.columns = normalized_names
    return result, dict(zip(source_names, normalized_names))


def leakage_exclusions(columns: list[str], target_column: str,
                       explicit_exclusions: list[str] | None = None) -> dict[str, str]:
    """Identify clear target, identifier, and post-event leakage fields.

    This uses exact normalized names and unambiguous suffix rules; it deliberately
    does not drop broad substrings such as ``src``, ``dst``, or ``target`` because
    those can occur in valid features like packet/port directions.
    """
    target = normalize_column_name(target_column)
    exclusions: dict[str, str] = {}
    explicit = {normalize_column_name(name) for name in (explicit_exclusions or [])}
    for name in columns:
        if name == target:
            exclusions[name] = "selected target label"
        elif name in explicit:
            exclusions[name] = "explicitly excluded in project configuration"
        elif name in LABEL_ALIASES or name.endswith(("_label", "_class", "_category")):
            exclusions[name] = "additional target or label-like field"
        elif name in IDENTIFIER_NAMES or name.endswith("_id"):
            exclusions[name] = "identifier or endpoint address"
        elif name in POST_EVENT_NAMES or name.startswith("post_event_"):
            exclusions[name] = "description or post-event/outcome field"
    return exclusions


def split_features_and_labels(frame: pd.DataFrame, label_column: str,
                              explicit_exclusions: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Separate safe model features from a target and return excluded fields in attrs."""
    normalized, _ = normalize_frame_columns(frame)
    target = normalize_column_name(label_column)
    if target not in normalized.columns:
        raise ValueError(f"Selected target column {target!r} is absent. Available columns: {normalized.columns.tolist()}")
    y = normalized[target].copy()
    reasons = leakage_exclusions(normalized.columns.tolist(), target, explicit_exclusions)
    X = normalized.drop(columns=list(reasons)).copy()
    X.attrs["excluded_columns"] = reasons
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after excluding target/leakage fields")
    return X, y


def feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names in stable input order."""
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return numeric, categorical
