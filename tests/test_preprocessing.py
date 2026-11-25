"""Synthetic preprocessing tests; no real traffic files are required."""
import numpy as np
import pandas as pd
import pytest

from nids.features import normalize_frame_columns, split_features_and_labels
from nids.preprocessing import (
    build_preprocessor,
    normalize_attack_labels,
    preprocess_frame,
    split_preprocessed_frame,
)


def representative_flows(rows=90):
    labels = ["Benign", "BruteForce", "Bot", "DoS", "DDoS", "Infiltration", "Web Attacks"]
    return pd.DataFrame({
        "Flow Duration": np.arange(rows, dtype=float),
        "Protocol": ["tcp", "udp", "icmp"] * (rows // 3),
        "Source IP": [f"192.0.2.{i % 20}" for i in range(rows)],
        "Attack Description": ["source text"] * rows,
        "Alert Status": ["post-event"] * rows,
        "Label": [labels[i % len(labels)] for i in range(rows)],
        "Row Marker": np.arange(rows),
    })


def test_normalization_rejects_duplicate_and_colliding_headers():
    with pytest.raises(ValueError, match="collide after normalization"):
        normalize_frame_columns(pd.DataFrame([[1, 2]], columns=["Flow ID", "flow-id"]))


def test_label_mapping_uses_verified_dataset_classes_and_preserves_source():
    canonical, binary = normalize_attack_labels(pd.Series([" benign ", "brute force", "Web-Attacks"]))
    assert canonical.tolist() == ["Benign", "BruteForce", "Web Attacks"]
    assert binary.tolist() == [0, 1, 1]


def test_downloaded_dataset_attack_subcategories_map_to_verified_parent_classes():
    source = [
        "Benign", "SSH-Bruteforce", "FTP-BruteForce", "Brute Force -Web",
        "Brute Force -XSS", "Bot", "DoS attacks-Hulk",
        "DoS attacks-SlowHTTPTest", "DoS attacks-GoldenEye",
        "DoS attacks-Slowloris", "DDoS attacks-LOIC-HTTP",
        "DDOS attack-HOIC", "DDOS attack-LOIC-UDP", "Infilteration",
        "SQL Injection",
    ]
    canonical, binary = normalize_attack_labels(pd.Series(source))
    assert canonical.tolist() == [
        "Benign", "BruteForce", "BruteForce", "BruteForce", "BruteForce",
        "Bot", "DoS", "DoS", "DoS", "DoS", "DDoS", "DDoS", "DDoS",
        "Infiltration", "Web Attacks",
    ]
    assert binary.tolist() == [0] + [1] * 14


def test_companion_binary_label_is_excluded_when_attack_column_is_target():
    frame = pd.DataFrame({
        "IN_BYTES": [10, 20, 30, 40],
        "IPV4_SRC_ADDR": ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
        "IPV4_DST_ADDR": ["10.0.1.1", "10.0.1.2", "10.0.1.3", "10.0.1.4"],
        "Label": [0, 1, 1, 0],
        "Attack": ["Benign", "SSH-Bruteforce", "Bot", "Benign"],
    })
    processed, report = preprocess_frame(frame, "Attack")
    assert processed["binary_target"].tolist() == [0, 1, 1, 0]
    assert not {"label", "ipv4_src_addr", "ipv4_dst_addr"} & set(report["model_feature_columns"])
    assert "in_bytes" in report["model_feature_columns"]
    assert report["excluded_columns_and_reasons"]["ipv4_src_addr"] == "identifier or endpoint address"
    with pytest.raises(ValueError, match="Unrecognized"):
        normalize_attack_labels(pd.Series(["Benign", "UnknownAttack"]))
    canonical, binary = normalize_attack_labels(pd.Series(["Benign", "BruteForce", "Web Attacks"]))
    assert canonical.tolist() == ["Benign", "BruteForce", "Web Attacks"]
    assert binary.tolist() == [0, 1, 1]


def test_missing_infinite_and_malformed_numeric_values_are_reported():
    frame = pd.DataFrame({
        "duration": [1.0, np.inf, np.nan, 4.0],
        "counter": ["1", "bad", "3", "4"],
        "Label": ["Benign", "DoS", "DDoS", "Bot"],
    })
    cleaned, report = preprocess_frame(frame, "Label", numeric_parse_threshold=0.7)
    assert report["infinite_numeric_values"]["duration"] == 1
    assert report["malformed_numeric_values"]["counter"] == 1
    assert cleaned["duration"].isna().sum() == 2
    assert report["missing_values_before"]["duration"] == 1


def test_leakage_columns_are_excluded_with_reasons_and_safe_fields_remain():
    cleaned, report = preprocess_frame(representative_flows(), "Label")
    assert "source_ip" not in cleaned.columns
    assert "attack_description" not in cleaned.columns
    assert "alert_status" not in cleaned.columns
    assert "protocol" in cleaned.columns
    assert "flow_duration" in cleaned.columns
    assert "source_ip" in report["excluded_columns_and_reasons"]
    assert "protocol" in report["model_feature_columns"]
    X, y = split_features_and_labels(cleaned, "binary_target")
    assert not {"original_label", "normalized_attack_label", "binary_target"} & set(X.columns)
    assert set(y.unique()) == {0, 1}


def test_class_distribution_is_preserved_across_splits_and_no_resampling():
    cleaned, quality = preprocess_frame(representative_flows(), "Label")
    splits, metadata = split_preprocessed_frame(cleaned, test_size=0.2, validation_size=0.2, seed=15)
    markers = [set(part["row_marker"]) for part in splits.values()]
    assert markers[0].isdisjoint(markers[1])
    assert markers[0].isdisjoint(markers[2])
    assert markers[1].isdisjoint(markers[2])
    assert sum(item["rows"] for item in metadata["split_sizes"].values()) == len(cleaned)
    assert metadata["resampling"] == "none"
    assert sum(quality["binary_distribution"].values()) == len(cleaned)


def test_unseen_categories_are_ignored_by_training_fitted_encoder():
    cleaned, _ = preprocess_frame(representative_flows(), "Label")
    splits, _ = split_preprocessed_frame(cleaned, seed=21)
    X_train, _ = split_features_and_labels(splits["train"], "binary_target")
    X_test, _ = split_features_and_labels(splits["test"], "binary_target")
    processor = build_preprocessor(X_train, scale_numeric=False)
    processor.fit(X_train)
    unseen = X_test.head(1).copy()
    unseen["protocol"] = "unseen-protocol-value"
    transformed = processor.transform(unseen)
    assert transformed.shape[0] == 1
    assert transformed.shape[1] == processor.transform(X_test.head(1)).shape[1]


def test_all_missing_features_are_removed_and_documented():
    frame = pd.DataFrame({"empty_feature": [None] * 10, "bytes": range(10),
                          "Label": ["Benign", "DoS"] * 5})
    cleaned, report = preprocess_frame(frame, "Label")
    assert "empty_feature" not in cleaned.columns
    assert report["excluded_columns_and_reasons"]["empty_feature"].startswith("all values missing")
