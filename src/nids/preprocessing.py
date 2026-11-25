"""Data quality, verified label normalization, safe splits, and model preprocessors.

This module never learns imputation/scaling/encoding statistics from validation or
test rows. ``build_preprocessor`` returns an unfitted sklearn transformer that must
be placed inside a model pipeline and fit on training rows only.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .features import (
    feature_types,
    leakage_exclusions,
    normalize_column_name,
    normalize_frame_columns,
    split_features_and_labels,
)

# Class names published for NF-CSE-CIC-IDS2018-v2 on the UQ dataset page.
# Keys normalize punctuation and case, while canonical values keep the source's
# class spelling for multiclass analysis.
VERIFIED_ATTACK_LABELS = {
    "benign": "Benign",
    "bruteforce": "BruteForce",
    "sshbruteforce": "BruteForce",
    "ftpbruteforce": "BruteForce",
    "bruteforceweb": "BruteForce",
    "bruteforcexss": "BruteForce",
    "bot": "Bot",
    "dos": "DoS",
    "dosattackshulk": "DoS",
    "dosattacksslowhttptest": "DoS",
    "dosattacksgoldeneye": "DoS",
    "dosattacksslowloris": "DoS",
    "ddos": "DDoS",
    "ddosattacksloichttp": "DDoS",
    "ddosattackhoic": "DDoS",
    "ddosattackloicudp": "DDoS",
    "infiltration": "Infiltration",
    # The downloaded source spells this label "Infilteration".
    "infilteration": "Infiltration",
    "webattacks": "Web Attacks",
    "sqlinjection": "Web Attacks",
}


def _label_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_attack_labels(labels: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (canonical original attack class, binary benign/attack label).

    Only the seven class names documented for this exact dataset are mapped.
    Unknown/missing labels raise instead of being silently counted as attacks.
    The raw source values are returned separately by ``preprocess_frame``.
    """
    if labels.isna().any():
        raise ValueError(f"Selected label contains {int(labels.isna().sum())} missing rows")
    keys = labels.astype(str).map(_label_key)
    unknown = sorted(set(keys) - set(VERIFIED_ATTACK_LABELS))
    if unknown:
        samples = sorted(set(labels.astype(str)[keys.isin(unknown)]))[:20]
        raise ValueError(f"Unrecognized NF-CSE-CIC-IDS2018-v2 labels {samples}; "
                         f"verified values are {list(VERIFIED_ATTACK_LABELS.values())}")
    attack_label = keys.map(VERIFIED_ATTACK_LABELS).astype("string")
    binary = (attack_label != "Benign").astype("int8").rename("binary_target")
    attack_label.name = "normalized_attack_label"
    return attack_label, binary


def _missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = {}
    for column in frame.columns:
        series = frame[column]
        blank_count = int(series.astype("string").str.strip().eq("").fillna(False).sum()) \
            if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype) else 0
        if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
            missing_tokens = series.astype("string").str.strip().str.lower()
            missing_text_count = int(missing_tokens.isin({"", "na", "n/a", "nan", "null", "none"}).fillna(False).sum())
        else:
            missing_text_count = 0
        counts[column] = int(series.isna().sum()) + missing_text_count
    return counts


def _clean_numeric_features(frame: pd.DataFrame, label_column: str,
                            numeric_parse_threshold: float) -> tuple[pd.DataFrame, dict]:
    """Coerce numeric fields, count malformed/infinite values, and retain nulls.

    An object column is treated as numeric only when at least the configured
    fraction of its nonmissing cells parse as numbers. Invalid cells in a
    numeric-like column become missing and are imputed later from training data.
    """
    result = frame.copy()
    malformed: dict[str, int] = {}
    infinite: dict[str, int] = {}
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in result.columns:
        if column == label_column:
            continue
        series = result[column]
        if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype):
            strings = series.astype("string").str.strip()
            strings = strings.mask(strings.str.lower().isin({"", "na", "n/a", "nan", "null", "none"}), pd.NA)
            nonmissing = strings.dropna()
            converted_nonmissing = pd.to_numeric(nonmissing, errors="coerce")
            numeric_ratio = float(converted_nonmissing.notna().mean()) if len(nonmissing) else 0.0
            is_numeric = numeric_ratio >= numeric_parse_threshold and len(nonmissing) > 0
        else:
            strings = series
            converted_nonmissing = None
            nonmissing = series.dropna()
            is_numeric = pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype)

        if is_numeric:
            converted = pd.to_numeric(strings, errors="coerce")
            malformed_mask = strings.notna() & converted.isna()
            malformed[column] = int(malformed_mask.sum())
            values = converted.to_numpy(dtype="float64", na_value=np.nan, copy=True)
            infinite_mask = np.isinf(values)
            infinite[column] = int(infinite_mask.sum())
            values[infinite_mask] = np.nan
            result[column] = values
            numeric_columns.append(column)
        else:
            # Preserve categories as strings; OneHotEncoder handles unseen values.
            result[column] = strings.astype("string")
            categorical_columns.append(column)
    return result, {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "malformed_numeric_values": malformed,
        "infinite_numeric_values": infinite,
    }


def preprocess_frame(frame: pd.DataFrame, label_column: str,
                     explicit_exclusions: list[str] | None = None,
                     drop_exact_duplicates: bool = True,
                     numeric_parse_threshold: float = 0.95) -> tuple[pd.DataFrame, dict]:
    """Normalize headers/values, analyze duplicates, map verified labels, and remove leaks.

    Returned rows contain safe feature columns plus ``original_label``,
    ``normalized_attack_label``, and ``binary_target``. Numeric nulls remain null
    until a train-only sklearn imputer is fitted. Original source CSVs are never
    modified. Exact duplicates are counted and removed by default to avoid
    duplicate-flow leakage across evaluation splits; set ``drop_exact_duplicates``
    false to retain them after reviewing the quality report.
    """
    if not label_column:
        raise ValueError("A label column must be explicitly selected from the dataset inspection report")
    if not 0.5 <= numeric_parse_threshold <= 1.0:
        raise ValueError("numeric_parse_threshold must be between 0.5 and 1.0")
    normalized, column_mapping = normalize_frame_columns(frame)
    target = normalize_column_name(label_column)
    if target not in normalized.columns:
        raise ValueError(f"Selected target {target!r} is missing after column normalization; "
                         f"available columns: {normalized.columns.tolist()}")
    if normalized[target].isna().any():
        raise ValueError(f"Selected target {target!r} has missing values")

    rows_before = len(normalized)
    missing_before = _missing_counts(normalized)
    duplicate_count = int(normalized.duplicated(keep="first").sum())
    if drop_exact_duplicates and duplicate_count:
        normalized = normalized.drop_duplicates(keep="first").reset_index(drop=True)

    original_labels = normalized[target].astype("string").rename("original_label")
    attack_label, binary_target = normalize_attack_labels(normalized[target])
    raw_features = normalized.drop(columns=[target])
    clean_features, numeric_report = _clean_numeric_features(raw_features, target, numeric_parse_threshold)

    exclusions = leakage_exclusions(clean_features.columns.tolist(), target, explicit_exclusions)
    excluded_columns = dict(exclusions)
    # All-null feature columns contain no training signal and break median/mode
    # imputation. Drop them explicitly and preserve the decision in metadata.
    for column in clean_features.columns:
        if clean_features[column].isna().all():
            excluded_columns[column] = "all values missing; no learnable training value"
    safe_features = clean_features.drop(columns=list(excluded_columns), errors="ignore")
    if safe_features.shape[1] == 0:
        raise ValueError("No model features remain after target/leakage/all-missing exclusions")

    processed = safe_features.reset_index(drop=True)
    processed["original_label"] = original_labels.reset_index(drop=True)
    processed["normalized_attack_label"] = attack_label.reset_index(drop=True)
    processed["binary_target"] = binary_target.reset_index(drop=True)

    original_class_counts = {str(k): int(v) for k, v in original_labels.value_counts(dropna=False).items()}
    attack_class_counts = {str(k): int(v) for k, v in attack_label.value_counts().items()}
    binary_counts = {str(k): int(v) for k, v in binary_target.value_counts().sort_index().items()}
    report = {
        "input_rows": int(rows_before),
        "output_rows": int(len(processed)),
        "exact_duplicate_rows": duplicate_count,
        "exact_duplicate_rows_removed": duplicate_count if drop_exact_duplicates else 0,
        "drop_exact_duplicates": bool(drop_exact_duplicates),
        "selected_source_label_column": target,
        "column_name_mapping": column_mapping,
        "missing_values_before": missing_before,
        "missing_values_after_numeric_cleaning": _missing_counts(clean_features),
        "malformed_numeric_values": numeric_report["malformed_numeric_values"],
        "infinite_numeric_values": numeric_report["infinite_numeric_values"],
        "numeric_columns": numeric_report["numeric_columns"],
        "categorical_columns": numeric_report["categorical_columns"],
        "excluded_columns_and_reasons": excluded_columns,
        "model_feature_columns": safe_features.columns.tolist(),
        "original_label_distribution": original_class_counts,
        "normalized_attack_distribution": attack_class_counts,
        "binary_distribution": binary_counts,
        "class_imbalance_handling": "No resampling. Training estimators use class_weight; validation and test retain observed distributions.",
        "numeric_parse_threshold": numeric_parse_threshold,
    }
    return processed, report


def _choose_stratification(processed: pd.DataFrame, test_size: float,
                           validation_size: float) -> tuple[pd.Series, str, list[str]]:
    """Prefer attack-class stratification, falling back transparently to binary."""
    classes = processed["normalized_attack_label"].astype(str)
    counts = classes.value_counts()
    n = len(classes)
    test_rows = int(np.ceil(n * test_size))
    train_val_rows = n - test_rows
    val_rows = int(np.ceil(n * validation_size))
    train_rows = n - test_rows - val_rows
    if counts.min() >= 3 and min(test_rows, val_rows, train_rows) >= len(counts):
        return classes, "normalized_attack_label", []
    return processed["binary_target"], "binary_target", [
        "At least one attack subtype was too small for reliable presence in every split; stratified by binary target instead."
    ]


def split_preprocessed_frame(processed: pd.DataFrame, test_size: float = 0.20,
                             validation_size: float = 0.20, seed: int = 42) -> tuple[dict[str, pd.DataFrame], dict]:
    """Create stratified train/validation/test row splits with no resampling."""
    from sklearn.model_selection import train_test_split
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive fractions that sum to less than 1")
    if not {"binary_target", "normalized_attack_label", "original_label"}.issubset(processed.columns):
        raise ValueError("Input must come from preprocess_frame and include all target metadata columns")
    rows = np.arange(len(processed))
    strata, stratified_by, warnings = _choose_stratification(processed, test_size, validation_size)
    train_val_idx, test_idx = train_test_split(rows, test_size=test_size, random_state=seed,
                                               stratify=strata.to_numpy())
    val_fraction = validation_size / (1 - test_size)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_fraction, random_state=seed,
                                          stratify=strata.iloc[train_val_idx].to_numpy())
    indices = {"train": train_idx, "validation": val_idx, "test": test_idx}
    splits = {name: processed.iloc[index].reset_index(drop=True) for name, index in indices.items()}
    counts = {name: {
        "rows": int(len(part)),
        "binary": {str(k): int(v) for k, v in part["binary_target"].value_counts().sort_index().items()},
        "attack_classes": {str(k): int(v) for k, v in part["normalized_attack_label"].value_counts().items()},
    } for name, part in splits.items()}
    metadata = {"seed": seed, "stratified_by": stratified_by, "split_sizes": counts,
                "validation_size": validation_size, "test_size": test_size,
                "training_size": 1 - validation_size - test_size,
                "resampling": "none", "warnings": warnings}
    return splits, metadata


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Return an unfitted transformer; fit only inside the training model pipeline.

    Logistic Regression benefits from standardization. Tree models do not, so
    callers set ``scale_numeric=False`` for Random Forest.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric, categorical = feature_types(X)
    transformers = []
    if numeric:
        numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if scale_numeric:
            numeric_steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric))
    if categorical:
        categorical_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical))
    if not transformers:
        raise ValueError("Preprocessor received no usable columns")
    return ColumnTransformer(transformers, remainder="drop")
