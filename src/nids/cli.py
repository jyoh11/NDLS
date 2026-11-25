"""Command-line entry points for the reproducible NIDS workflow."""
import argparse
import json
import logging
import os
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(prog="nids")
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample", help="Create a bounded sample CSV")
    sample.add_argument("--config", default=None)
    sample.add_argument("--path", help="Source CSV file or directory; defaults to configured data.path")
    sample.add_argument("--output", default="data/cache/dev_sample.csv")
    sample.add_argument("--rows", type=int, default=None)
    sample.add_argument("--label-column", default=None, help="Required explicit label column after inspection")
    sample.add_argument("--seed", type=int, default=None)
    inspect = sub.add_parser("inspect", help="Inspect CSV files, columns, missingness and candidate labels")
    inspect.add_argument("--config", default=None)
    inspect.add_argument("--path", help="CSV file or directory; defaults to configured data.path")
    inspect.add_argument("--label-column", default=None, help="Optional explicitly selected label column")
    inspect.add_argument("--chunksize", type=int, default=None)
    preprocess = sub.add_parser("preprocess", help="Clean labels/features and save train/validation/test CSVs")
    preprocess.add_argument("--config", default=None)
    preprocess.add_argument("--path", help="Input CSV file or directory; defaults to configured data.path")
    preprocess.add_argument("--label-column", default=None, help="Verified target column from `nids inspect`")
    preprocess.add_argument("--output-dir", default=None)
    preprocess.add_argument("--max-rows", type=int, default=None,
                            help="Class-preserving development sample size; default from preprocessing config")
    preprocess.add_argument("--all-data", action="store_true",
                            help="Process every row in memory; check system RAM and disk first")
    preprocess.add_argument("--keep-duplicates", action="store_true",
                            help="Retain exact duplicate records after the quality report is generated")
    train = sub.add_parser("train", help="Train, calibrate and evaluate models")
    train.add_argument("--config", default=None)
    train.add_argument("--mode", choices=["development", "full"], default=None,
                       help="Development uses a reproducible bounded sample; full reads all configured CSV rows")
    train.add_argument("--max-rows", type=int, default=None,
                       help="Override the development sample row count")
    train.add_argument("--max-train-rows", type=int, default=None,
                       help="Cap fitted training rows; omit in full mode to fit every training row")
    explain = sub.add_parser("explain", help="Generate bounded SHAP artifacts from the saved trained model")
    explain.add_argument("--config", default=None)
    explain.add_argument("--path", help="CSV file or directory used as an explanation sample source")
    explain.add_argument("--bundle", default=None, help="Saved model bundle; defaults to configured artifacts/model.joblib")
    explain.add_argument("--output-dir", default=None)
    explain.add_argument("--max-rows", type=int, default=None,
                         help="Bound the source sample size; defaults to 10x the explanation sample size")
    download = sub.add_parser("download-kaggle", help="Download the synopsis dataset with Kaggle CLI")
    download.add_argument("--config", default=None)
    download.add_argument("--confirm-large-download", action="store_true",
                          help="Required opt-in after reviewing the printed storage estimate")
    url_download = sub.add_parser("download-url", help="Stream an explicitly configured direct CSV URL")
    url_download.add_argument("--config", default=None)
    url_download.add_argument("--url", default=os.getenv("NIDS_DATA_URL"),
                              help="Direct URL; set NIDS_DATA_URL instead of saving credentials in config")
    url_download.add_argument("--output", default="data/cache/dataset.csv")
    url_download.add_argument("--confirm-download", action="store_true",
                              help="Required opt-in; the download is never automatic")
    space = sub.add_parser("check-space", help="Compare free disk space to an estimated download requirement")
    space.add_argument("--path", default="data/raw", help="Directory on the target volume")
    space.add_argument("--download-bytes", type=int, required=True,
                       help="File size shown by the source download page, in bytes")
    space.add_argument("--expansion-factor", type=float, default=5.0,
                       help="Storage multiplier for archive, extraction and temporary copies")
    demo = sub.add_parser("demo", help="Generate synthetic flows and run the bounded end-to-end demo")
    demo.add_argument("--config", default=None)
    demo.add_argument("--output-dir", default="reports/generated/synthetic_demo",
                      help="Generated dataset and model artifacts directory")
    demo.add_argument("--rows", type=int, default=900, help="Synthetic row count (minimum 120)")
    demo.add_argument("--seed", type=int, default=None)
    predict = sub.add_parser("predict", help="Score a JSON flow with the exact saved API model and policy")
    predict.add_argument("--config", default=None)
    predict.add_argument("--input", required=True, help="Path to one JSON object containing model feature names")
    predict.add_argument("--bundle", default=None, help="Saved model bundle; defaults to configured artifacts/model.joblib")
    predict.add_argument("--explain", action="store_true", help="Include a local SHAP explanation when available")
    args = parser.parse_args()
    # Lazy imports keep `python -m nids --help` useful before installation.
    from .logging_config import configure_logging
    configure_logging()
    cfg = {"seed": 42, "data": {"path": "data/raw", "label_column": None,
                                 "chunksize": 250_000, "sample_rows": 100_000},
           "training": {}}
    if args.command == "check-space":
        from .ingest import disk_space
        if args.download_bytes < 1 or args.expansion_factor < 1:
            raise SystemExit("Download bytes must be positive and expansion factor must be at least 1")
        free, total = disk_space(args.path)
        required = int(args.download_bytes * args.expansion_factor)
        report = {"volume_path": str(args.path), "free_bytes": free, "total_bytes": total,
                  "source_file_bytes": args.download_bytes, "estimated_required_bytes": required,
                  "expansion_factor": args.expansion_factor, "enough_space": free >= required}
        print(json.dumps(report, indent=2))
        if free < required:
            raise SystemExit("Insufficient free space for the estimated download and extraction footprint")
        return
    use_config = not (
        (args.command == "demo" and not args.config and not os.getenv("NIDS_CONFIG")) or
        (args.command == "predict" and args.bundle and not args.config and not os.getenv("NIDS_CONFIG"))
    )
    if ((args.command not in {"inspect", "sample"} and use_config) or args.config or
            (args.command in {"inspect", "sample"} and not args.path)):
        from .config import load_config
        cfg = load_config(args.config or os.getenv("NIDS_CONFIG", "config.yaml"))
    data_cfg = cfg["data"]
    if os.getenv("NIDS_LABEL_COLUMN") and not data_cfg.get("label_column"):
        data_cfg["label_column"] = os.environ["NIDS_LABEL_COLUMN"]
    if args.command == "demo":
        import numpy as np
        import pandas as pd
        from .ingest import inspect_dataset
        from .training import load_dataset, train_and_evaluate
        if args.rows < 120:
            raise SystemExit("Synthetic demo needs at least 120 rows for stratified calibration and evaluation splits.")
        seed = args.seed if args.seed is not None else cfg.get("seed", 42)
        root = Path(args.output_dir)
        root.mkdir(parents=True, exist_ok=True)
        dataset_path = root / "synthetic_flows.csv"
        rng = np.random.default_rng(seed)
        labels = np.resize(np.array(["Benign", "DoS", "DDoS"], dtype=object), args.rows)
        rng.shuffle(labels)
        attack = labels != "Benign"
        duration = rng.lognormal(mean=np.where(attack, 4.0, 3.0), sigma=0.65)
        fwd_packets = rng.poisson(lam=np.where(attack, 28, 9)).astype(int)
        bwd_packets = rng.poisson(lam=np.where(attack, 12, 7)).astype(int)
        bytes_per_flow = (duration * (fwd_packets + bwd_packets) * rng.uniform(8, 80, args.rows)).astype(float)
        protocols = np.where(attack, rng.choice(["udp", "tcp"], args.rows, p=[0.75, 0.25]),
                             rng.choice(["tcp", "udp"], args.rows, p=[0.8, 0.2]))
        demo_frame = pd.DataFrame({
            "Flow Duration": duration,
            "Total Fwd Packets": fwd_packets,
            "Total Bwd Packets": bwd_packets,
            "Flow Bytes/s": bytes_per_flow,
            "Protocol": protocols,
            "Flow ID": [f"synthetic-{index:06d}" for index in range(args.rows)],
            "Attack Description": np.where(attack, "synthetic attack annotation", "synthetic benign annotation"),
            "Label": labels,
        })
        demo_frame.to_csv(dataset_path, index=False)
        ingestion_report = inspect_dataset(dataset_path, "Label", data_cfg.get("chunksize", 250_000))
        frame = load_dataset(dataset_path, "Label", data_cfg.get("chunksize", 250_000), max_rows=None, seed=seed)
        artifacts_dir = root / "artifacts"
        train_cfg = cfg.get("training", {})
        explanation_cfg = cfg.get("explanations", {})
        result = train_and_evaluate(
            frame, artifacts_dir, "Label", seed=seed,
            target_recall=float(train_cfg.get("target_recall", 0.90)),
            review_threshold=float(train_cfg.get("review_threshold", 0.5)),
            max_train_rows=None,
            calibration_method=train_cfg.get("calibration_method", "sigmoid"),
            test_size=float(train_cfg.get("test_size", 0.20)),
            validation_size=float(train_cfg.get("validation_size", 0.20)),
            model_names=["random_forest"],
            explicit_exclusions=cfg.get("preprocessing", {}).get("excluded_columns", []),
            drop_exact_duplicates=bool(cfg.get("preprocessing", {}).get("drop_exact_duplicates", True)),
            model_parameters={"random_forest": {"n_estimators": 48, "min_samples_leaf": 2,
                                                   "class_weight": "balanced_subsample", "n_jobs": -1}},
            dataset_version="synthetic-development-demo",
            collection_note="Synthetic generated flow records for offline pipeline smoke testing; not network capture data.",
            calibration_fraction=float(train_cfg.get("calibration_fraction", 0.5)),
            minimum_confidence=float(train_cfg.get("minimum_confidence", 0.8)),
            ece_bins=int(train_cfg.get("ece_bins", 10)),
            min_isotonic_samples=int(train_cfg.get("min_isotonic_samples", 1000)),
            explanation_background_size=min(32, max(4, args.rows // 10)),
            explanation_sample_size=min(30, max(4, args.rows // 20)),
            explanation_top_k=10,
            explanation_max_dense_mb=float(explanation_cfg.get("max_dense_mb", 512.0)),
            generate_explanations=True,
        )
        import joblib
        bundle = joblib.load(artifacts_dir / "model.joblib")
        from .features import split_features_and_labels
        from .preprocessing import preprocess_frame
        processed, _ = preprocess_frame(frame, "Label",
            explicit_exclusions=cfg.get("preprocessing", {}).get("excluded_columns", []),
            drop_exact_duplicates=bool(cfg.get("preprocessing", {}).get("drop_exact_duplicates", True)))
        raw_features, _ = split_features_and_labels(processed, "binary_target")
        first = raw_features.loc[:, bundle["features"]].iloc[0]
        safe_row = {name: value.item() if hasattr(value, "item") else value for name, value in first.items()}
        sample_path = root / "sample_flow.json"
        sample_path.write_text(json.dumps(safe_row, indent=2), encoding="utf-8")
        (root / "ingestion_report.json").write_text(json.dumps(ingestion_report, indent=2), encoding="utf-8")
        print(json.dumps({"dataset": str(dataset_path), "rows": args.rows,
                          "artifacts_dir": str(artifacts_dir), "run_id": result["run_id"],
                          "selected_model": result["selected_model"], "sample_flow": str(sample_path),
                          "explanation_status": result["explanations"].get("status")}, indent=2))
    elif args.command == "predict":
        import joblib
        artifact_dir = Path(cfg.get("training", {}).get("artifacts_dir", "artifacts"))
        bundle_path = Path(args.bundle) if args.bundle else artifact_dir / "model.joblib"
        if not bundle_path.is_file():
            raise SystemExit(f"Trained bundle is unavailable: {bundle_path}. Run `nids train` or `nids demo` first.")
        try:
            record = json.loads(Path(args.input).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Could not read a JSON flow object: {exc}") from exc
        if not isinstance(record, dict):
            raise SystemExit("Input JSON must be one object mapping model feature names to values.")
        bundle = joblib.load(bundle_path)
        # FastAPI and this command call the same shared inference/scoring implementation.
        from .inference import score_bundle, validate_features
        ordered, errors = validate_features(record, bundle)
        if errors:
            print(json.dumps({"feature_validation_errors": errors}, indent=2))
            raise SystemExit(2)
        result = score_bundle(bundle, ordered, include_explanation=args.explain)
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "inspect":
        from .ingest import inspect_dataset
        result = inspect_dataset(args.path or data_cfg["path"], args.label_column or data_cfg.get("label_column"),
                                 args.chunksize or data_cfg.get("chunksize", 250_000))
        print(json.dumps(result, indent=2))
    elif args.command == "sample":
        from .ingest import prepare_sample
        label_column = args.label_column or data_cfg.get("label_column")
        if not label_column:
            raise SystemExit("Choose a label from `nids inspect` and pass --label-column or set NIDS_LABEL_COLUMN.")
        result = prepare_sample(args.path or data_cfg["path"], args.output,
            args.rows or data_cfg.get("sample_rows", 100_000), label_column,
            args.seed if args.seed is not None else cfg.get("seed", 42),
            data_cfg.get("chunksize", 250_000))
        print(json.dumps(result, indent=2))
    elif args.command == "preprocess":
        from .ingest import inspect_dataset, prepare_sample
        from .preprocessing import preprocess_frame, split_preprocessed_frame
        from .training import load_dataset
        label_column = args.label_column or data_cfg.get("label_column")
        if not label_column:
            raise SystemExit("Choose the label shown by `nids inspect`; pass --label-column or set NIDS_LABEL_COLUMN.")
        source_path = args.path or data_cfg["path"]
        prep_cfg = cfg.get("preprocessing", {})
        output_dir = Path(args.output_dir or prep_cfg.get("output_dir", "data/processed"))
        source_report = inspect_dataset(source_path, label_column, data_cfg.get("chunksize", 250_000))
        if args.all_data:
            frame = load_dataset(source_path, label_column, data_cfg.get("chunksize", 250_000),
                                 max_rows=None, seed=cfg.get("seed", 42))
            sampling_report = {"mode": "all_data", "rows": len(frame)}
        else:
            sample_rows = args.max_rows or prep_cfg.get("development_sample_rows", 100_000)
            sample_path = Path("data/cache/preprocess_sample.csv")
            sampling_report = prepare_sample(source_path, sample_path, sample_rows, label_column,
                seed=cfg.get("seed", 42), chunksize=data_cfg.get("chunksize", 250_000))
            frame = load_dataset(sample_path, label_column, data_cfg.get("chunksize", 250_000),
                                 max_rows=None, seed=cfg.get("seed", 42))
            sampling_report["mode"] = "class_preserving_development_sample"
        processed, quality_report = preprocess_frame(frame, label_column,
            explicit_exclusions=prep_cfg.get("excluded_columns", []),
            drop_exact_duplicates=not args.keep_duplicates and prep_cfg.get("drop_exact_duplicates", True),
            numeric_parse_threshold=prep_cfg.get("numeric_parse_threshold", 0.95))
        splits, split_report = split_preprocessed_frame(processed,
            cfg["training"]["test_size"], cfg["training"]["validation_size"], cfg.get("seed", 42))
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_frame in splits.items():
            split_frame.to_csv(output_dir / f"{split_name}.csv", index=False)
        (output_dir / "source_inspection.json").write_text(json.dumps(source_report, indent=2), encoding="utf-8")
        (output_dir / "quality_report.json").write_text(json.dumps(quality_report, indent=2), encoding="utf-8")
        (output_dir / "split_metadata.json").write_text(json.dumps(split_report, indent=2), encoding="utf-8")
        (output_dir / "run_metadata.json").write_text(json.dumps({"sampling": sampling_report,
            "label_column": label_column, "seed": cfg.get("seed", 42),
            "features": quality_report["model_feature_columns"]}, indent=2), encoding="utf-8")
        print(f"Processed splits and metadata saved under {output_dir}")
    elif args.command == "explain":
        import joblib
        from .features import split_features_and_labels
        from .policy import confidence_alert
        from .preprocessing import preprocess_frame
        from .training import load_dataset
        from .explain import generate_explanation_artifacts
        train_cfg = cfg["training"]
        explain_cfg = cfg.get("explanations", {})
        label_column = data_cfg.get("label_column")
        if not label_column:
            raise SystemExit("Set the inspected label column before generating a stratified explanation sample.")
        artifact_dir = Path(train_cfg.get("artifacts_dir", "artifacts"))
        bundle_path = Path(args.bundle) if args.bundle else artifact_dir / "model.joblib"
        if not bundle_path.exists():
            raise SystemExit(f"Trained model bundle not found: {bundle_path}")
        bundle = joblib.load(bundle_path)
        source_path = args.path or data_cfg["path"]
        max_rows = args.max_rows or int(explain_cfg.get("explanation_sample_size", 100)) * 10
        source = load_dataset(source_path, label_column, data_cfg.get("chunksize", 250_000),
                              max_rows=max_rows, seed=cfg.get("seed", 42))
        processed, _ = preprocess_frame(source, label_column,
            explicit_exclusions=cfg.get("preprocessing", {}).get("excluded_columns", []),
            drop_exact_duplicates=cfg.get("preprocessing", {}).get("drop_exact_duplicates", True),
            numeric_parse_threshold=cfg.get("preprocessing", {}).get("numeric_parse_threshold", 0.95))
        features, _ = split_features_and_labels(processed, "binary_target")
        expected = list(bundle.get("features", []))
        missing = [name for name in expected if name not in features.columns]
        if missing:
            raise SystemExit(f"Explanation source is missing features required by the saved model: {missing}")
        features = features.loc[:, expected]
        probabilities = bundle["model"].predict_proba(features)[:, 1]
        decisions = [confidence_alert(float(probability), bundle["threshold"],
            bundle.get("review_threshold", 0.5), bundle.get("minimum_confidence", 0.8))
            for probability in probabilities]
        output_dir = Path(args.output_dir or artifact_dir / "explanations" / "latest")
        manifest = generate_explanation_artifacts(bundle, features, decisions, output_dir,
            explanation_sample_size=explain_cfg.get("explanation_sample_size", 100),
            top_k=explain_cfg.get("top_k", 15), seed=cfg.get("seed", 42),
            max_dense_mb=explain_cfg.get("max_dense_mb", 512.0))
        print(json.dumps({"output_dir": str(output_dir), **manifest}, indent=2))
    elif args.command == "download-kaggle":
        from .ingest import download_kaggle, estimate_kaggle_download, inspect_dataset
        dataset_ref = data_cfg["dataset_ref"]
        estimate = estimate_kaggle_download(dataset_ref)
        print(json.dumps(estimate, indent=2), flush=True)
        if not args.confirm_large_download:
            raise SystemExit("Review the estimate, then re-run with --confirm-large-download to proceed.")
        result = download_kaggle(dataset_ref, data_cfg["path"], confirm_large_download=True,
                                 estimate=estimate)
        result["dataset_inspection"] = inspect_dataset(data_cfg["path"], data_cfg.get("label_column"),
                                                        data_cfg.get("chunksize", 250_000))
        print(json.dumps(result, indent=2))
    elif args.command == "download-url":
        from .ingest import download_stream, estimate_url_download, inspect_dataset
        args.url = args.url or os.getenv("NIDS_DATA_URL")
        if not args.url:
            raise SystemExit("No direct URL configured. Set NIDS_DATA_URL only after verifying it is a legitimate CSV download URL.")
        estimate = estimate_url_download(args.url, args.output)
        print(json.dumps(estimate, indent=2), flush=True)
        if not estimate["enough_space"]:
            raise SystemExit("Insufficient free space for the estimated download footprint")
        if not args.confirm_download:
            raise SystemExit("Download is opt-in. Review the source and re-run with --confirm-download.")
        result = download_stream(args.url, args.output,
                                 expected_size=estimate["content_length_bytes"])
        result["dataset_inspection"] = inspect_dataset(args.output, data_cfg.get("label_column"),
                                                       data_cfg.get("chunksize", 250_000))
        print(json.dumps(result, indent=2))
    else:
        from .training import load_dataset, train_and_evaluate
        train_cfg = cfg["training"]
        if not data_cfg.get("label_column"):
            raise SystemExit("Set the confirmed label column with NIDS_LABEL_COLUMN after running `nids inspect`.")
        mode = args.mode or train_cfg.get("mode", "development")
        if mode == "full":
            max_rows = None
            configured_train_cap = train_cfg.get("full_max_train_rows")
        else:
            max_rows = args.max_rows or data_cfg.get("max_rows") or train_cfg.get("development_rows", 100_000)
            configured_train_cap = train_cfg.get("max_train_rows", 500_000)
        if args.max_train_rows is not None:
            configured_train_cap = args.max_train_rows
        frame = load_dataset(data_cfg["path"], data_cfg["label_column"], data_cfg["chunksize"], max_rows, cfg["seed"])
        output_dir = train_cfg.get("artifacts_dir", "artifacts")
        result = train_and_evaluate(
            frame, output_dir, data_cfg["label_column"], cfg["seed"],
            train_cfg["target_recall"], train_cfg["review_threshold"], configured_train_cap,
            train_cfg["calibration_method"], train_cfg["test_size"], train_cfg["validation_size"],
            train_cfg["models"], cfg.get("preprocessing", {}).get("excluded_columns", []),
            cfg.get("preprocessing", {}).get("drop_exact_duplicates", True),
            model_parameters=train_cfg.get("model_parameters", {}),
            dataset_version=data_cfg.get("dataset_version", data_cfg.get("dataset_ref", "local")),
            collection_note=train_cfg.get("collection_note"),
            calibration_fraction=train_cfg.get("calibration_fraction", 0.5),
            minimum_confidence=train_cfg.get("minimum_confidence", 0.8),
            ece_bins=train_cfg.get("ece_bins", 10),
            min_isotonic_samples=train_cfg.get("min_isotonic_samples", 1000),
            explanation_background_size=cfg.get("explanations", {}).get("background_sample_size", 100),
            explanation_sample_size=cfg.get("explanations", {}).get("explanation_sample_size", 100),
            explanation_top_k=cfg.get("explanations", {}).get("top_k", 15),
            explanation_max_dense_mb=cfg.get("explanations", {}).get("max_dense_mb", 512.0),
            generate_explanations=cfg.get("explanations", {}).get("enabled", True),
        )
        print(f"Evaluation written to {output_dir}/metrics.json; selected model: {result['selected_model']}")

if __name__ == "__main__":
    main()
