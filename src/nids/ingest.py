"""Streaming, inspectable dataset intake for the synopsis's NF-CSE-CIC-IDS2018-v2.

The University of Queensland (UQ) source advertises CSV and links to an open
UQRDM file-delivery page. It does not publish a dataset-specific API or direct
download URL in its documentation. Kaggle CLI support is provided separately
for the Kaggle mirror named in the synopsis. No source file is modified.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Iterator
from urllib.parse import urlparse

import pandas as pd
import numpy as np
import requests
from .features import normalize_column_name, normalize_frame_columns

EXPECTED_KAGGLE_DATASET = "sahangunasekara92/nf-cse-cic-ids2018-v2-full"
OFFICIAL_DATASET_PAGE = "https://staff.itee.uq.edu.au/marius/NIDS_datasets/"
OFFICIAL_UQRDM_FILE_PAGE = "https://rdm.uq.edu.au/files/ce5161d0-ef9c-11ed-827d-e762de186848"
LABEL_NAME_HINTS = {"label", "class", "attack", "category", "target", "traffic_type",
                    "attack_label", "attack_type", "is_attack", "traffic_label"}
SUPPORTED_CSV_SUFFIXES = (".csv", ".csv.gz")
TRUNCATED_LABEL_MARKER = "[additional distinct values omitted]"


def disk_space(path: str | Path) -> tuple[int, int]:
    """Return free and total bytes on the volume containing ``path``."""
    path = Path(path)
    probe = path if path.exists() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe.resolve())
    return usage.free, usage.total


def _valid_csv_path(path: Path) -> bool:
    lowered = path.name.lower()
    return any(lowered.endswith(suffix) for suffix in SUPPORTED_CSV_SUFFIXES)


def list_csv_files(path: str | Path, exclude_paths: set[str | Path] | None = None) -> list[Path]:
    """Find CSV or gzip-compressed CSV files in a file or directory."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    excluded = {Path(item).expanduser().resolve() for item in (exclude_paths or set())}
    if path.is_file():
        if path.resolve() in excluded:
            return []
        if not _valid_csv_path(path):
            raise ValueError(f"Unsupported file format: {path.name}; expected .csv or .csv.gz")
        return [path]
    files = sorted(item for item in path.rglob("*") if item.is_file() and _valid_csv_path(item)
                   and item.resolve() not in excluded)
    if not files:
        raise FileNotFoundError(f"No .csv or .csv.gz files found under {path}")
    return files


def _read_csv_chunks(path: Path, chunksize: int, validate_structure: bool = True) -> Iterator[pd.DataFrame]:
    """Read strict CSV chunks; malformed records raise instead of being skipped."""
    try:
        if validate_structure:
            _validate_csv_structure(path)
        yield from pd.read_csv(path, chunksize=chunksize, low_memory=False,
                               on_bad_lines="error", index_col=False, encoding="utf-8-sig")
    except Exception as exc:
        raise ValueError(f"Could not read CSV file {path}: {exc}") from exc


def _validate_csv_structure(path: Path) -> None:
    """Reject malformed records with a field count that differs from the header.

    Pandas can infer an index for over-wide records in some CSV shapes. This
    lightweight streaming preflight uses Python's strict CSV parser to ensure
    no extra or missing fields are silently accepted before pandas reads chunks.
    """
    opener = gzip.open if path.name.lower().endswith(".csv.gz") else open
    try:
        with opener(path, "rt", encoding="utf-8-sig", errors="strict", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if not header:
                raise ValueError("CSV has no header")
            normalized_header = [normalize_column_name(name) for name in header]
            if len(set(normalized_header)) != len(normalized_header):
                raise ValueError(f"duplicate or colliding column names after normalization: {header}")
            width = len(header)
            for row_number, record in enumerate(reader, start=2):
                if not record:  # Blank lines are also skipped by pandas.
                    continue
                if len(record) != width:
                    raise ValueError(f"record {row_number} has {len(record)} fields; header has {width}")
    except (csv.Error, UnicodeDecodeError) as exc:
        raise ValueError(f"CSV structure is malformed: {exc}") from exc


def _read_header(path: Path) -> tuple[list[str], list[str]]:
    """Read one CSV header and return source names plus normalized names."""
    opener = gzip.open if path.name.lower().endswith(".csv.gz") else open
    with opener(path, "rt", encoding="utf-8-sig", errors="strict", newline="") as stream:
        header = next(csv.reader(stream, strict=True), None)
    if not header:
        raise ValueError(f"CSV file has no header: {path}")
    normalized = [normalize_column_name(name) for name in header]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"duplicate or colliding source column names after normalization in {path}: {header}")
    return header, normalized


def read_chunks(path: str | Path, chunksize: int = 250_000,
                exclude_paths: set[str | Path] | None = None,
                validate_structure: bool = True) -> Iterator[pd.DataFrame]:
    """Yield aligned chunks using the union of fields found across input files.

    Fields absent from an individual source CSV are represented as missing in
    that file's rows; `inspect_dataset` reports these schema differences.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    files = list_csv_files(path, exclude_paths)
    union_columns: list[str] = []
    for file_path in files:
        _, names = _read_header(file_path)
        for name in names:
            if name not in union_columns:
                union_columns.append(name)
    for file_path in files:
        for chunk in _read_csv_chunks(file_path, chunksize, validate_structure):
            normalized, _ = normalize_frame_columns(chunk)
            yield normalized.reindex(columns=union_columns)


def _candidate_label_columns(columns: list[str]) -> list[str]:
    """Suggest human-review candidates by header name; never select implicitly."""
    candidates = []
    for column in columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", column.strip().lower()).strip("_")
        if normalized in LABEL_NAME_HINTS or normalized.endswith(("_label", "_class", "_category")):
            candidates.append(column)
    return candidates


def inspect_dataset(path: str | Path, label_column: str | None = None,
                    chunksize: int = 250_000,
                    exclude_paths: set[str | Path] | None = None) -> dict:
    """Inspect files incrementally and report schemas, counts, missingness and labels.

    Candidate labels are header-name suggestions only. The selected label name
    must be supplied explicitly after the user checks the report.
    """
    if chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    files = list_csv_files(path, exclude_paths)
    headers = {file_path: _read_header(file_path) for file_path in files}
    union_columns = list(dict.fromkeys(name for _, names in headers.values() for name in names))
    normalized_label = normalize_column_name(label_column) if label_column else None
    if normalized_label is not None and normalized_label not in union_columns:
        raise ValueError(f"Selected label column {normalized_label!r} is absent. Available columns: {union_columns}")
    candidates = _candidate_label_columns(union_columns)
    count_columns = set(candidates)
    if normalized_label:
        count_columns.add(normalized_label)
    candidate_counts: dict[str, dict[str, int]] = {name: {} for name in union_columns if name in count_columns}
    missing_values = dict.fromkeys(union_columns, 0)
    report_files = []
    total_rows = 0
    first_schema = set(headers[files[0]][1])

    for file_path in files:
        source_columns, file_columns = headers[file_path]
        file_column_set = set(file_columns)
        row_count = 0
        for raw_chunk in _read_csv_chunks(file_path, chunksize):
            chunk, _ = normalize_frame_columns(raw_chunk)
            if chunk.columns.tolist() != file_columns:
                raise ValueError(f"CSV columns changed while reading {file_path}")
            row_count += len(chunk)
            total_rows += len(chunk)
            for column in union_columns:
                if column in file_column_set:
                    missing_values[column] += int(chunk[column].isna().sum())
                else:
                    # Structural missingness: this file does not provide the field.
                    missing_values[column] += len(chunk)
            for candidate in candidate_counts:
                if candidate not in file_column_set:
                    continue
                counts = chunk[candidate].dropna().astype(str).value_counts()
                current = candidate_counts[candidate]
                for value, count in counts.items():
                    if value not in current and len(current) >= 1000:
                        value = TRUNCATED_LABEL_MARKER
                    current[value] = current.get(value, 0) + int(count)
        if row_count == 0:
            raise ValueError(f"CSV file contains a header but no data records: {file_path}")
        report_files.append({
            "path": str(file_path), "size_bytes": file_path.stat().st_size,
            "rows": row_count, "columns": len(file_columns),
            "source_columns": source_columns, "normalized_columns": file_columns,
            "missing_from_union": [name for name in union_columns if name not in file_column_set],
            "unique_to_file": [name for name in file_columns if name not in first_schema],
        })

    selected_distribution = candidate_counts.get(normalized_label, {}) if normalized_label else None
    return {
        "source_path": str(Path(path).expanduser()),
        "file_count": len(report_files),
        "files": report_files,
        "total_size_bytes": sum(item["size_bytes"] for item in report_files),
        "row_count": total_rows,
        "column_count": len(union_columns),
        "columns": union_columns,
        "missing_values": missing_values,
        "candidate_label_columns": candidates,
        "selected_label_column": normalized_label,
        "selected_label_counts": selected_distribution,
        "label_candidate_counts": candidate_counts,
        "label_counts_truncated": {name: TRUNCATED_LABEL_MARKER in counts
                                   for name, counts in candidate_counts.items()},
        "csv_format": "CSV (UTF-8 or pandas-detectable text; gzip CSV also supported)",
    }


def validate_frame(frame: pd.DataFrame, label_column: str | None = None) -> None:
    """Validate records and (when supplied) an explicitly selected label column."""
    if frame.empty:
        raise ValueError("Dataset contains no data records")
    if len(frame.columns) == 0:
        raise ValueError("Dataset has no columns")
    if label_column is not None:
        normalized, _ = normalize_frame_columns(frame)
        selected = normalize_column_name(label_column)
        if selected not in normalized.columns:
            raise ValueError(f"Missing selected label column {selected!r}")
        if normalized[selected].isna().all():
            raise ValueError(f"Selected label column {selected!r} has no non-null values")


def _atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _fingerprint(files: list[Path], label_column: str, rows: int, seed: int) -> str:
    facts = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    raw = json.dumps([facts, label_column, rows, seed], separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sample_quotas(counts: dict[str, int], sample_size: int) -> dict[str, int]:
    """Allocate a proportional sample while retaining every observed label."""
    if sample_size < len(counts):
        raise ValueError(f"Sample size {sample_size} is too small to retain all {len(counts)} labels")
    total = sum(counts.values())
    quotas = {label: max(1, math.floor(sample_size * count / total)) for label, count in counts.items()}
    # Cap over-allocation, then distribute any remainder by largest fractional part.
    fractions = sorted(counts, key=lambda label: (-(sample_size * counts[label] / total % 1), label))
    while sum(quotas.values()) > sample_size:
        label = max((key for key in quotas if quotas[key] > 1), key=lambda key: quotas[key])
        quotas[label] -= 1
    for label in fractions:
        if sum(quotas.values()) >= sample_size:
            break
        if quotas[label] < counts[label]:
            quotas[label] += 1
    return quotas


def prepare_sample(source: str | Path, destination: str | Path, rows: int,
                   label_column: str, seed: int = 42, chunksize: int = 250_000,
                   resume: bool = True) -> dict:
    """Create a reproducible, class-preserving sample without changing inputs.

    A completed sample with a matching input fingerprint is reused on reruns.
    Two chunked passes first count labels, then apply per-label reservoir sampling.
    """
    if rows < 1:
        raise ValueError("rows must be a positive integer")
    label_column = normalize_column_name(label_column)
    output = Path(destination).expanduser()
    source_path = Path(source).expanduser()
    if source_path.is_dir():
        try:
            output.resolve().relative_to(source_path.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("Write development samples outside the source-data directory (for example, under data/cache/)")
    excluded = {output, output.with_suffix(output.suffix + ".part")}
    files = list_csv_files(source, excluded)
    initial_report = inspect_dataset(source, label_column, chunksize, excluded)
    label_counts = initial_report["selected_label_counts"]
    if not label_counts:
        raise ValueError(f"Selected label column {label_column!r} has no non-null labels")
    if TRUNCATED_LABEL_MARKER in label_counts:
        raise ValueError("Selected column has over 1,000 distinct values; it does not look like a safe label column")
    if initial_report["row_count"] < rows:
        rows = initial_report["row_count"]
    quotas = _sample_quotas(label_counts, rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    fingerprint = _fingerprint(files, label_column, rows, seed)
    if resume and output.exists() and manifest_path.exists():
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old_manifest.get("input_fingerprint") == fingerprint:
                return {"output": str(output), "rows": old_manifest["output_rows"],
                        "label_counts": old_manifest["output_label_counts"], "resumed": True}
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    seen = dict.fromkeys(quotas, 0)
    reservoirs: dict[str, pd.DataFrame] = {label: pd.DataFrame() for label in quotas}
    rngs = {label: np.random.default_rng(seed + sum(ord(char) for char in label))
            for label in quotas}
    random_key_column = "__nids_sample_random_key__"
    for chunk in read_chunks(source, chunksize, excluded, validate_structure=False):
        validate_frame(chunk, label_column)
        labels = chunk[label_column].astype("string")
        if labels.isna().any():
            raise ValueError("Selected label column contains missing values; fix or explicitly review them before sampling")
        for label, group in chunk.assign(__nids_label=labels).groupby("__nids_label", sort=False, dropna=False):
            key = str(label)
            if key not in quotas:
                raise ValueError(f"Label {key!r} appeared after inspection; source may have changed")
            quota = quotas[key]
            seen[key] += len(group)
            fresh = group.drop(columns="__nids_label").copy()
            fresh[random_key_column] = rngs[key].random(len(fresh))
            combined = pd.concat([reservoirs[key], fresh], ignore_index=True)
            if len(combined) > quota:
                keys = combined[random_key_column].to_numpy()
                chosen = np.argpartition(keys, quota - 1)[:quota]
                reservoirs[key] = combined.iloc[chosen].reset_index(drop=True)
            else:
                reservoirs[key] = combined

    sampled = pd.concat([reservoirs[label] for label in quotas], ignore_index=True)
    sampled = sampled.drop(columns=random_key_column)
    if len(sampled) != sum(quotas.values()):
        raise ValueError("Could not satisfy all per-label sample quotas; source data may have changed")
    if _fingerprint(files, label_column, rows, seed) != fingerprint:
        raise ValueError("Input files changed while sampling; no output was finalized. Re-run inspection and sample again.")
    temp = output.with_suffix(output.suffix + ".part")
    sampled.to_csv(temp, index=False)
    os.replace(temp, output)
    output_counts = {str(key): int(value) for key, value in sampled[label_column].astype(str).value_counts().items()}
    manifest = {"input_fingerprint": fingerprint, "source_path": str(source), "label_column": label_column,
                "requested_rows": rows, "output_rows": len(sampled), "seed": seed,
                "source_label_counts": label_counts, "output_label_counts": output_counts,
                "input_files": [str(path) for path in files]}
    _atomic_json(manifest_path, manifest)
    return {"output": str(output), "rows": len(sampled), "label_counts": output_counts, "resumed": False}


def _content_length(response: requests.Response) -> int | None:
    value = response.headers.get("Content-Length")
    return int(value) if value and value.isdigit() else None


def estimate_url_download(url: str, destination: str | Path,
                         reserve_bytes: int = 1_000_000_000,
                         timeout: tuple[int, int] = (20, 120)) -> dict:
    """Read a direct URL's size and compare it to free space without downloading."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Dataset URL must be an explicit HTTP or HTTPS URL")
    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    head = requests.head(url, headers={"Accept-Encoding": "identity"},
                         allow_redirects=True, timeout=timeout)
    head.raise_for_status()
    if "text/html" in head.headers.get("Content-Type", "").lower():
        raise ValueError("Configured URL is an HTML page, not a direct dataset download; use the source's manual download action")
    expected_size = _content_length(head)
    if expected_size is None:
        raise ValueError("The server did not provide Content-Length; refusing a large download without a size estimate")
    free, total = disk_space(destination.parent)
    required = expected_size + reserve_bytes
    return {"url": url, "content_length_bytes": expected_size, "reserve_bytes": reserve_bytes,
            "estimated_required_bytes": required, "free_bytes": free, "total_bytes": total,
            "enough_space": free >= required}


def download_stream(url: str, destination: str | Path, reserve_bytes: int = 1_000_000_000,
                    timeout: tuple[int, int] = (20, 120),
                    expected_size: int | None = None) -> dict:
    """Stream an explicitly configured HTTP(S) URL to a resumable local cache.

    Resume uses HTTP Range when the server supports it. Content-Length (if
    supplied) is checked against free storage plus the configured extraction
    reserve before any body is written.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Dataset URL must be an explicit HTTP or HTTPS URL")
    if expected_size is not None and expected_size < 0:
        raise ValueError("expected_size cannot be negative")
    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    estimate = estimate_url_download(url, destination, reserve_bytes, timeout) if expected_size is None else None
    expected_size = expected_size if expected_size is not None else estimate["content_length_bytes"]
    free, _ = disk_space(destination.parent)
    required = expected_size + reserve_bytes
    if free < required:
        raise OSError(f"Estimated download plus reserve needs {required:,} bytes; only {free:,} bytes are free")
    if destination.exists() and destination.stat().st_size == expected_size:
        return {"path": str(destination), "bytes": expected_size, "cached": True, "resumed": False}
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        partial.unlink()
        offset = 0

    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=timeout) as response:
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", "").lower():
            raise ValueError("URL returned HTML instead of a dataset file; use the source's manual download action")
        append = offset > 0 and response.status_code == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
        if offset and response.status_code == 206 and not append:
            raise IOError("Server returned an unexpected byte range; retained the partial cache unchanged")
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as output_file:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    output_file.write(block)
    actual_size = partial.stat().st_size
    if actual_size != expected_size:
        raise IOError(f"Download incomplete: expected {expected_size:,} bytes, received {actual_size:,}. "
                      f"Partial file is retained at {partial} for a later resume.")
    os.replace(partial, destination)
    return {"path": str(destination), "bytes": actual_size, "cached": False, "resumed": bool(offset)}


def estimate_kaggle_download(dataset_ref: str) -> dict:
    """Query file sizes through Kaggle CLI metadata; never guess the archive size."""
    if dataset_ref != EXPECTED_KAGGLE_DATASET:
        raise ValueError(f"Dataset ref must match synopsis: {EXPECTED_KAGGLE_DATASET}")
    result = subprocess.run(["kaggle", "datasets", "files", dataset_ref, "--csv"],
                            check=True, capture_output=True, text=True)
    reader = csv.DictReader(result.stdout.splitlines())
    if not reader.fieldnames:
        raise ValueError("Kaggle CLI returned no file metadata; cannot estimate storage safely")
    size_column = next((name for name in reader.fieldnames if name.lower() in {"size", "totalbytes", "total_bytes"}), None)
    if size_column is None:
        raise ValueError(f"Kaggle file metadata has no recognized size field: {reader.fieldnames}")
    records = list(reader)
    sizes = [int(row[size_column]) for row in records if row.get(size_column, "").isdigit()]
    if not sizes:
        raise ValueError("Kaggle CLI returned no usable file sizes")
    compressed_bytes = sum(sizes)
    # Allow space for compressed data, extracted CSVs (conservatively 4x), and temp files.
    estimated_required = compressed_bytes * 5 + 1_000_000_000
    return {"files": records, "compressed_bytes": compressed_bytes,
            "estimated_required_bytes": estimated_required}


def download_kaggle(dataset_ref: str, destination: str | Path,
                    confirm_large_download: bool = False,
                    estimate: dict | None = None) -> dict:
    """Download synopsis Kaggle mirror only after explicit consent and size check."""
    if not confirm_large_download:
        raise PermissionError("Full Kaggle download is opt-in. Re-run with --confirm-large-download after reviewing storage needs.")
    if dataset_ref != EXPECTED_KAGGLE_DATASET:
        raise ValueError(f"Dataset ref must match synopsis: {EXPECTED_KAGGLE_DATASET}")
    metadata = estimate or estimate_kaggle_download(dataset_ref)
    destination = Path(destination).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    free, _ = disk_space(destination)
    required = metadata["estimated_required_bytes"]
    if free < required:
        raise OSError(f"Estimated storage requirement is {required:,} bytes; only {free:,} bytes are free")
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_ref,
                    "-p", str(destination), "--unzip"], check=True)
    return {"path": str(destination), "estimated_required_bytes": required,
            "files": metadata["files"]}
