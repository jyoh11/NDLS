import pandas as pd
import pytest
from nids.ingest import validate_frame

def test_schema_validation_requires_label():
    with pytest.raises(ValueError, match="label"):
        validate_frame(pd.DataFrame({"f1": [1]}), "Label")

def test_schema_validation_accepts_labeled_rows():
    validate_frame(pd.DataFrame({"f1": [1], "Label": ["Benign"]}), "Label")

def test_inspection_counts_multiple_files_and_suggests_label(tmp_path):
    from nids.ingest import inspect_dataset
    (tmp_path / "part-a.csv").write_text("Flow Bytes,Traffic Label\n1,Benign\n2,DDoS\n", encoding="utf-8")
    (tmp_path / "part-b.csv").write_text("Flow Bytes,Traffic Label\n3,Benign\n", encoding="utf-8")
    report = inspect_dataset(tmp_path, label_column="Traffic Label", chunksize=1)
    assert report["row_count"] == 3
    assert report["column_count"] == 2
    assert report["file_count"] == 2
    assert report["candidate_label_columns"] == ["traffic_label"]
    assert report["selected_label_counts"] == {"Benign": 2, "DDoS": 1}

def test_inspection_aligns_different_multi_file_headers(tmp_path):
    from nids.ingest import inspect_dataset
    (tmp_path / "part-a.csv").write_text("flow,label\n1,Benign\n", encoding="utf-8")
    (tmp_path / "part-b.csv").write_text("label,bytes\nDDoS,2\n", encoding="utf-8")
    report = inspect_dataset(tmp_path, label_column="label")
    assert report["column_count"] == 3
    assert "flow" in report["files"][1]["missing_from_union"]
    assert report["missing_values"]["flow"] == 1

def test_duplicate_headers_after_normalization_are_rejected(tmp_path):
    from nids.ingest import inspect_dataset
    path = tmp_path / "duplicates.csv"
    path.write_text("Flow ID,flow-id,Label\n1,2,Benign\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate or colliding"):
        inspect_dataset(path)

def test_inspection_fails_on_malformed_rows_instead_of_skipping(tmp_path):
    from nids.ingest import inspect_dataset
    path = tmp_path / "bad.csv"
    path.write_text("flow,label\n1,Benign\n2,DDoS,extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read CSV"):
        inspect_dataset(path, chunksize=1)

def test_sample_retains_each_label_and_is_reusable(tmp_path):
    from nids.ingest import prepare_sample
    source = tmp_path / "flows.csv"
    source.write_text("bytes,Label\n" + "".join(f"{i},{'Benign' if i < 18 else 'DDoS'}\n" for i in range(20)), encoding="utf-8")
    output = tmp_path / "sample.csv"
    first = prepare_sample(source, output, rows=8, label_column="Label", seed=11, chunksize=4)
    second = prepare_sample(source, output, rows=8, label_column="Label", seed=11, chunksize=4)
    assert first["rows"] == 8
    assert set(first["label_counts"]) == {"Benign", "DDoS"}
    assert second["resumed"] is True

def test_sample_refuses_to_omit_rare_attack_categories(tmp_path):
    from nids.ingest import prepare_sample
    source = tmp_path / "flows.csv"
    source.write_text("Label\nBenign\nDDoS\nDoS\n", encoding="utf-8")
    with pytest.raises(ValueError, match="retain all"):
        prepare_sample(source, tmp_path / "sample.csv", rows=2, label_column="Label")

def test_stream_download_checks_size_and_caches_with_resume_support(tmp_path, monkeypatch):
    from nids import ingest

    class FakeResponse:
        def __init__(self, status, headers, body=b""):
            self.status_code, self.headers, self.body = status, headers, body
        def raise_for_status(self):
            return None
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def iter_content(self, chunk_size):
            for offset in range(0, len(self.body), chunk_size):
                yield self.body[offset:offset + chunk_size]

    content = b"synthetic csv"
    monkeypatch.setattr(ingest, "disk_space", lambda _path: (10_000, 10_000))
    monkeypatch.setattr(ingest.requests, "head", lambda *_a, **_k: FakeResponse(200, {"Content-Length": str(len(content))}))
    monkeypatch.setattr(ingest.requests, "get", lambda *_a, **_k: FakeResponse(200, {}, content))
    result = ingest.download_stream("https://example.test/data.csv", tmp_path / "cached.csv", reserve_bytes=0)
    assert result["bytes"] == len(content)
    assert (tmp_path / "cached.csv").read_bytes() == content

def test_kaggle_adapter_estimates_size_and_requires_confirmation(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from nids import ingest
    monkeypatch.setattr(ingest.subprocess, "run", lambda *_a, **_k: SimpleNamespace(
        stdout="name,size\nflows.csv,2000\n", stderr=""))
    estimate = ingest.estimate_kaggle_download(ingest.EXPECTED_KAGGLE_DATASET)
    assert estimate["compressed_bytes"] == 2000
    with pytest.raises(PermissionError, match="opt-in"):
        ingest.download_kaggle(ingest.EXPECTED_KAGGLE_DATASET, tmp_path)
