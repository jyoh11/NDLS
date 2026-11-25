# Explainable IDS with Confidence-Aware False-Positive Reduction

Student-friendly research implementation of the synopsis using the exact primary dataset **NF-CSE-CIC-IDS2018-v2**. The supplied UQ archive has been checksum-verified and its 3.22 GB flow CSV is stored under Git-ignored `data/raw/`. The actual CSV schema and label counts have been scanned, and a bounded 100,000-row real-data training/evaluation run has completed. This is a development sample result, not a full-dataset result.

**Share bundle note:** The shareable project ZIP includes source code, tests, documentation, and the saved 100,000-sample model and reports, but excludes the 3.22 GB raw CSV, virtual environment, caches, and secrets. The bundled API/dashboard can use the saved model without retraining. To train on NF-CSE-CIC-IDS2018-v2, obtain the dataset and place the CSV at the configured path as described below.

## Requirements checklist and progress

- [x] Primary dataset: supplied official UQ archive; exact CSV checked against its SHA-1 manifest. Optional Kaggle mirror retrieval remains available.
- [x] Chunked CSV access, schema validation, streaming URL helper, bounded sample support, disk-space guard.
- [x] Cleaning, binary label mapping, categorical encoding, numeric imputation, duplicate removal.
- [x] Stratified train/validation/test partitions; preprocessing/model fitting use training rows; validation is split into disjoint calibration and policy-selection subsets.
- [x] Logistic Regression and Random Forest comparison, calibration scores, FPR/FNR, precision, recall, F1, accuracy, confusion matrix, Brier score, log loss, ROC AUC.
- [x] Confidence-aware three-tier decision policy and a same-test-set comparison with the standard 0.50 cutoff.
- [x] SHAP local explanation, FastAPI scoring API, Streamlit analyst dashboard.
- [x] Unit tests and student/viva notes below.
- [x] Windows/Python 3.11 dependency bounds, environment overrides, rotating logs and Git exclusions for local data/secrets/artifacts.
- [x] Dataset-free smoke test and CLI help command.
- [x] Bounded real-data development run completed with a 100,000-row sample; sample report and trained artifacts are in Git-ignored `artifacts/`.
- [ ] Train and evaluate on the complete 18.9-million-row dataset; full-data metrics remain pending.

## Architecture

`src/nids/ingest.py` handles local files, chunked CSVs and safe download helpers. `features.py` separates target/identifier fields from model inputs; `preprocessing.py` creates the train-only transformations; `training.py` creates stratified partitions, fits model pipelines, selects a threshold against a validation recall target, and evaluates once on the untouched test partition. `calibration.py` fits validation-only probability calibration. `inference.py` is the single feature-validation, model-scoring, alert-policy and explanation path used by both FastAPI and `nids predict`. `policy.py` holds confidence tiers. `explain.py` computes per-flow SHAP contributions. `api.py` exposes model readiness, predictions, feature schema and saved metrics endpoints. `dashboard_data.py` bounds and validates dashboard inputs; `dashboard.py` is the analyst interface. `logging_config.py` sets up console and rotating file logs.

## Project directory tree

```text
.
├── .env.example              # Safe local configuration template
├── .gitignore                # Excludes private and generated files
├── config.yaml               # Reproducible experiment settings
├── dashboard.py              # Streamlit analyst dashboard
├── pyproject.toml             # Python 3.11 package and dependency bounds
├── README.md                  # Setup, dataset, architecture and runbook
├── docs/VIVA_GUIDE.md         # Student explanation, research decisions and viva Q&A
├── src/nids/
│   ├── __main__.py            # `python -m nids` entry point
│   ├── api.py                 # FastAPI routes
│   ├── calibration.py         # Validation-only probability calibration
│   ├── cli.py                 # Dataset, sample, demo, train and predict commands
│   ├── config.py              # YAML plus environment overrides
│   ├── explain.py             # SHAP local explanations
│   ├── features.py            # Label/identifier and feature schema handling
│   ├── ingest.py              # Dataset access and schema checks
│   ├── inference.py           # Shared validation and CLI/API scoring path
│   ├── dashboard_data.py      # Bounded dashboard input and API requests
│   ├── logging_config.py      # Rotating local logs
│   ├── policy.py              # Binary labels and confidence-aware alerts
│   ├── preprocessing.py       # Numeric/categorical transformations
│   └── training.py            # Splits, model comparison and metrics
├── tests/                     # Smoke, unit and integration tests
├── data/raw/                  # Main flow CSV; ignored by Git
│   └── NF-CSE-CIC-IDS2018-v2.csv
├── docs/dataset/              # Small feature reference from the UQ archive
├── data/cache/                # Samples and derived cache; ignored by Git
├── artifacts/                 # Saved model bundle; ignored by Git
├── reports/generated/         # Generated reports; ignored by Git
└── logs/                      # Rotating runtime logs; ignored by Git
```

The `.gitkeep` placeholders retain the empty folder layout. `.env`, credentials, virtual environments, Python caches, raw datasets, and generated model/report files are excluded by `.gitignore`.

## End-to-end PowerShell runbook

Use this section as the canonical command sequence. Start in the repository root; use Python 3.11. The development demo generates synthetic rows, does not access the real dataset, and saves its bundle under `reports/generated/synthetic_demo/artifacts` so it cannot overwrite `artifacts/` used by a real run.

### 1. Create and activate the environment

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation in this window, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.

### 2. Install application, dashboard, and test dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"
$env:PYTHONPATH = "src"
Copy-Item .env.example .env
```

### 3. Run automated tests

```powershell
python -m pytest -q
```

### 4. Run the offline synthetic-data demonstration

```powershell
python -m nids demo --output-dir reports/generated/synthetic_demo --rows 900 --seed 42
python -m nids predict --input reports/generated/synthetic_demo/sample_flow.json --bundle reports/generated/synthetic_demo/artifacts/model.joblib --explain
```

`demo` generates a CSV, inspects and validates it, applies leakage-aware preprocessing and train/validation/test splitting, trains Random Forest, fits calibration, selects its alert threshold from validation data, evaluates once on held-out test data, and saves versioned artifacts/reports. `predict` uses the shared inference module also called by FastAPI. All generated labels and metrics are synthetic and are not evidence about the primary dataset. If SHAP is unsupported or unavailable, prediction still returns and the explanation field explicitly carries an error.

### 5. Configure and inspect the real synopsis dataset

The supplied archive has been extracted to `data/raw/`. The 3,221,378,977-byte CSV has **18,893,708 rows and 45 columns**: 43 NetFlow features, numeric binary `Label`, and detailed `Attack`. The `Attack` field is the training target; the companion `Label` column is explicitly excluded from model features to prevent target leakage. The small feature-description CSV is not treated as a flow file.

```powershell
$env:NIDS_DATA_PATH = "$PWD/data/raw/NF-CSE-CIC-IDS2018-v2.csv"
$env:NIDS_LABEL_COLUMN = "Attack"
python -m nids inspect --path $env:NIDS_DATA_PATH --label-column Attack
```

The default `config.yaml` and `.env.example` already point to this local CSV and select `Attack`; the environment variables above only need to be set again if your shell has older overrides. Keep the raw CSV in `data/raw/` because that whole directory is Git-ignored.

The completed real-data development run uses seed 42 and a class-preserving sample of 100,000 flows. It split into 60,000 training, 10,000 calibration, 10,000 policy-validation, and 20,000 held-out test flows. The selected Random Forest had validation FPR 0 and attack recall 95.06%; Logistic Regression at the same validation recall target had FPR 9.70%, so Random Forest was selected. On the sampled test set, the calibrated 0.50 baseline made 5 false-positive alerts and had attack recall 95.98% (96 missed attacks). The validation-selected policy threshold (0.9874) made 0 high-priority false positives, with attack recall 95.48% (108 missed attacks): 12 additional attacks were not promoted to high priority. It placed 49 flows into review, including 36 benign flows; therefore the number of benign flows needing analyst attention was 36, higher than the baseline's 5 false positives. The policy removed high-priority false alarms in this sample but increased analyst review workload; it did not reduce all analyst-visible false positives. Calibrated Brier score improved from 0.00604 to 0.00500 and ECE from 0.00691 to 0.00151. These are sample-specific results, not a claim about the full dataset or live traffic.

The run report and plots are in `artifacts/metrics.json`, `artifacts/comparison.json`, and `artifacts/runs/run_20261120T152535469753Z_seed42/`. The saved bundle in `artifacts/model.joblib` uses 40 features; it excludes `Attack`, the companion `Label`, both IPv4 endpoint fields, and the DNS query ID. The complete source CSV remains unchanged. To rerun this bounded experiment:

```powershell
$env:NIDS_DATA_PATH = "$PWD/data/raw/NF-CSE-CIC-IDS2018-v2.csv"
$env:NIDS_LABEL_COLUMN = "Attack"
$env:NIDS_ARTIFACTS_DIR = "artifacts"
python -m nids train --mode development --max-rows 100000
```

### 6. Train with reproducible configuration

```powershell
python -m nids train --mode development
```

The configured development sample and training-row cap are in `config.yaml`. To train on every configured CSV row, after checking RAM/disk and class counts:

```powershell
python -m nids train --mode full
```

Both modes write the bundle, fitted preprocessor, calibration/model/policy settings, feature schema, held-out metrics, plots, and run metadata under `artifacts/`. The bundle's `features` and `feature_dtypes` are the inference contract; labels, identifiers, and post-event fields are excluded before fitting. The fitted transformations live inside the saved pipelines, so inference applies the training-fitted preprocessing rather than refitting it.

### 7. Start FastAPI (PowerShell window 1)

For the synthetic demo, point the service at the demo artifact folder:

```powershell
$env:PYTHONPATH = "src"
$env:NIDS_ARTIFACTS_DIR = "reports/generated/synthetic_demo/artifacts"
python -m uvicorn nids.api:app --app-dir src --host 127.0.0.1 --port 8000
```

For the real dataset, use `$env:NIDS_ARTIFACTS_DIR = "artifacts"` instead. Confirm readiness at `http://127.0.0.1:8000/health`; interactive docs are at `/docs`. The API scores supplied features; it does not capture live network traffic.

### 8. Start Streamlit (PowerShell window 2)

Set `NIDS_ARTIFACTS_DIR` to the same value as the API process, then run:

```powershell
$env:PYTHONPATH = "src"
$env:NIDS_API_URL = "http://127.0.0.1:8000"
$env:NIDS_ARTIFACTS_DIR = "reports/generated/synthetic_demo/artifacts"
python -m streamlit run dashboard.py
```

For the real model, change the last environment value to `artifacts`. The dashboard only reads status/schema/reports from FastAPI and uses the API for inference; it never trains a model during startup. It reports historical held-out metrics separately from session-only prediction counts.

The 0.50 cutoff is a reference point. The operating threshold is selected on the policy-validation subset to meet `target_recall` while minimizing false-positive rate when possible. Calibration affects probability reliability; it does not by itself reduce false positives. Compare the raw, calibrated, and confidence-aware policy results on the held-out test report.

## Dataset access and storage

The primary source is the [University of Queensland NIDS dataset page](https://staff.itee.uq.edu.au/marius/NIDS_datasets/). Its V2 page reports 18,893,708 records and seven parent classes. The supplied UQ archive contains a 45-column CSV: 43 flow features plus numeric `Label` and fine-grained `Attack`. A complete local streaming scan found 18,893,708 rows, no rows with an unexpected number of columns, and these source `Attack` value counts: Benign 16,635,567; SSH-Bruteforce 94,979; FTP-BruteForce 25,933; Brute Force -Web 2,143; Brute Force -XSS 927; Bot 143,097; DoS attacks-Hulk 432,648; DoS attacks-SlowHTTPTest 14,116; DoS attacks-GoldenEye 27,723; DoS attacks-Slowloris 9,512; DDoS attacks-LOIC-HTTP 307,300; DDOS attack-HOIC 1,080,858; DDOS attack-LOIC-UDP 2,112; Infilteration 116,361; SQL Injection 432. These map to the seven documented parent classes. The CSV SHA-1 matches the archive's `manifest-sha1.txt`. UQ lists a commercial-use restriction; follow the dataset terms and cite Sarhan et al. (2022) for academic use.

The synopsis also names the Kaggle mirror `sahangunasekara92/nf-cse-cic-ids2018-v2-full`. For that mirror, the documented Kaggle CLI is available as an optional adapter. It first queries file metadata for a size estimate and requires a second explicit `--confirm-large-download` flag before transferring data. Credentials remain in the user's Kaggle profile. A direct URL can be configured only if you have independently verified that it is a legitimate CSV download URL; the streaming helper requires a server size estimate, checks disk space, and retains partial data for HTTP Range resume.

### Obtain and configure the official dataset manually

1. Open the UQ dataset page above, select **NF-CSE-CIC-IDS2018-v2**, then follow its **click here to download** link to the UQ RDM delivery page.
2. Read the file size shown on the RDM page before downloading. Use five times that size as a conservative allowance for the archive, extracted CSVs and temporary copies, then check the target drive:

   ```powershell
   $DownloadBytes = 1000000000 # Replace with the size shown by UQRDM, in bytes
   python -m nids check-space --path data/raw --download-bytes $DownloadBytes
   ```

3. Only after the check reports enough space, use the RDM page's **Download file** action. If the downloaded item is an archive, extract the main `NF-CSE-CIC-IDS2018-v2.csv` file. Do not rename or edit the original CSV.
4. Place that CSV at `data/raw/NF-CSE-CIC-IDS2018-v2.csv` or set `NIDS_DATA_PATH` to its exact path. The separate `NetFlow_v2_Features.csv` reference belongs in `docs/dataset/`, not beside flow CSVs. No UQ API key or guessed URL is used.
5. The real archive used in this workspace was scanned and verified: selected target is `Attack`; the companion `Label` field is excluded from features. To repeat inspection:

   ```powershell
   python -m nids inspect --path data/raw/NF-CSE-CIC-IDS2018-v2.csv --label-column Attack
   ```

6. Run the development training mode first. It creates a bounded, class-preserving sample and never reads the entire CSV into RAM. Full training is deliberately not the default.

If using Kaggle instead, install its CLI, create the Kaggle API token in account settings, and store `kaggle.json` only under `%USERPROFILE%\.kaggle\`. Review the estimate before opting in:

```powershell
python -m nids download-kaggle
python -m nids download-kaggle --confirm-large-download
```

For a direct, verified CSV URL, use `NIDS_DATA_URL` in your local `.env`. First run `python -m nids download-url` to print the remote size and storage check without downloading; after reviewing it, run `python -m nids download-url --confirm-download`. No URL is prefilled because the official UQ metadata does not publish one. A non-downloadable landing page will fail validation instead of being treated as data.

### Inspection and development sample

`nids inspect` processes chunks without retaining full files in memory. It reports each file's byte size, rows, columns, total counts, missing values, candidate-label value counts, and fields missing from or unique to each CSV. Multiple files with differing columns or order are aligned by normalized name; structurally absent fields are reported as missing. Malformed records and duplicate normalized headers raise errors instead of being silently dropped. `nids sample` performs chunked, two-pass, per-label reservoir sampling, retains every observed label, writes a sidecar manifest, and reuses a matching completed sample on reruns. Requesting fewer rows than there are label categories is rejected instead of dropping categories.

```powershell
python -m nids inspect --path data/raw/NF-CSE-CIC-IDS2018-v2.csv --label-column Attack
$LabelColumn = "Attack"
python -m nids sample --path data/raw/NF-CSE-CIC-IDS2018-v2.csv --output data/cache/dev_sample.csv --rows 100000 --label-column $LabelColumn
```

The optional Kaggle download checks file metadata and disk space first; the direct-URL downloader checks `Content-Length` plus a configurable 1 GB extraction reserve. Full downloads are never triggered by inspect, sample, train, or application startup.

## Quick start on Windows PowerShell

```powershell
python --version  # Use Python 3.11.x
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"
$env:PYTHONPATH = "src"
Copy-Item .env.example .env
python -m nids --help
python -m pytest -q
```

To smoke-test the CLI without dataset access, run `python -m nids --help` and `python -m pytest -q tests/test_smoke.py`. The real dataset path and label are configured in `config.yaml`; start with the bounded development training command above, which streams the large CSV to prepare a sample before loading model rows.

```powershell
python -m nids.cli train
python -m pytest
uvicorn nids.api:app --app-dir src --reload
```

In a second PowerShell window, activate the environment and launch the dashboard:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run dashboard.py
```

For a full-data run, set `data.path` back to the extracted data directory and `data.max_rows: null`. The training config caps fitting rows at `max_train_rows` to keep student hardware use manageable; to fit all available training rows set it to `null`. A capped training subset is still split without overlap with validation/test; the resulting test set is held out from fitting.

## FastAPI backend and analyst dashboard

The service loads the saved model, metrics, and metadata once during application startup. It scores caller-supplied flow features only; it does not capture packets or watch live traffic. Requests must include every saved raw feature with the same name and compatible value type. Extra fields, nulls, booleans, malformed values, and non-finite numbers receive a structured `422` response without echoing submitted values. Feature ordering follows the saved bundle, regardless of JSON key order.

```powershell
# From the activated Python 3.11 virtual environment and project root
python -m uvicorn nids.api:app --app-dir src --host 127.0.0.1 --port 8000 --reload
```

Open interactive API documentation at `http://127.0.0.1:8000/docs`. The Streamlit dashboard uses the same service. Endpoints:

- `GET /health` reports model, saved metrics, and metadata readiness. If the calibrated model is unavailable, prediction endpoints return `503` with a readiness code.
- `GET /model/info` returns model/dataset version, ordered raw feature names/types, and safe policy settings. `GET /schema` remains as a compatibility alias.
- `GET /config` returns only policy/calibration and explanation settings; it omits filesystem paths and secrets.
- `GET /metrics` returns the saved held-out evaluation report and identifies it as a saved report, not live performance.
- `POST /predict` scores features and returns the predicted class/label, calibrated attack probability, confidence score/definition, policy decision, alert reason, thresholds, and empty feature-validation errors on success. Set `include_explanation: true` to include local SHAP data when supported.
- `POST /explain` scores the same flow and requests a local SHAP explanation. If SHAP is unavailable or the model is unsupported, the response still contains the prediction and reports the explanation error.

Example PowerShell request (replace feature names and values using `GET /model/info`):

```powershell
$Body = @{ features = @{ flow_duration = 1200; protocol = "tcp" }; include_explanation = $true } | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body $Body
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/explain -ContentType "application/json" -Body $Body

The dashboard is a five-section analyst console: Overview, Analyze flows, Alert review, Model evaluation, and Dataset & model. It fetches model readiness, schema, safe training metadata, and the saved held-out evaluation report from FastAPI. Predictions always go through `POST /predict`; the dashboard does not run a separate model. Each CSV/JSON upload is limited to 2 MB and 100 rows, and inputs are checked against the current feature schema before submission. Predictions in the review queue are session-only; the dashboard has no persistent alert database or live traffic feed.

### Launch both apps on Windows

In the first PowerShell window, from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m uvicorn nids.api:app --app-dir src --host 127.0.0.1 --port 8000
```

In a second PowerShell window:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
$env:NIDS_API_URL = "http://127.0.0.1:8000"
$env:NIDS_ARTIFACTS_DIR = "artifacts"
python -m streamlit run dashboard.py
```

Open `http://localhost:8501`. The API reads model and evaluation artifacts from `NIDS_ARTIFACTS_DIR` (default `artifacts`); keep the same setting in both windows. If no trained artifacts exist, the dashboard shows service/model readiness and explains that evaluation is unavailable without inventing values. To verify dashboard helpers and backend request integration after installing development dependencies, run `python -m pytest -q tests/test_dashboard_data.py tests/test_api.py`. To smoke-check the UI script, run `python -m py_compile dashboard.py` and launch it with Streamlit while the API is running.
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/model/info
Invoke-RestMethod http://127.0.0.1:8000/config
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

Set `NIDS_CORS_ORIGINS` to a comma-separated list of trusted dashboard origins if the UI runs on another address. The default allows local Streamlit origins on port 8501 and does not enable credentialed CORS. API logging records request IDs, paths, status codes, and duration; it does not log submitted flow values.

## Major modules and viva notes

### Ingestion and validation (`ingest.py`)

Purpose: find CSV data, stream explicitly configured downloads, check storage and inspect candidate labels without assuming their names. Chunked CSV reading bounds memory use; the sample helper performs per-label reservoir sampling, preserves all observed labels, writes a reproducibility manifest, and reuses a completed matching sample. Important functions: `inspect_dataset`, `prepare_sample`, `download_kaggle`, `download_stream`, `read_chunks`, and `validate_frame`. Explain that Kaggle authentication is kept in the user's profile and no credential is embedded in code. Limitation: the official UQ download is a manual RDM file-delivery action; archive extraction is left to the Kaggle CLI or user, and schema checks cannot validate the semantic meaning of each network feature.

### Preprocessing and training (`preprocessing.py`, `features.py`, `training.py`)

Purpose: turn labeled flows into a tested binary detector. The seven verified dataset classes are preserved for analysis; Benign maps to 0 and documented attack classes map to 1. Unknown labels fail for review. Median imputation and scaling are fitted in a pipeline only on training rows; categorical values are imputed and one-hot encoded. Logistic Regression is an interpretable linear baseline; Random Forest captures nonlinear feature interactions. Class weights address imbalance without synthetically creating flows. Important parameters: `seed`, `target_recall`, `review_threshold`, `max_train_rows`, and calibration method (currently sigmoid). Viva: stratified 60/20/20 split preserves class ratios; the validation partition calibrates probabilities and selects threshold; the test partition is untouched until final evaluation. Limitation: random stratification does not enforce temporal or host-based separation, so correlated flows could occur across partitions; future work should evaluate time/host grouped splits. The current model comparison selects by validation FPR with recall as tie-breaker, constrained by target recall.

### Confidence policy (`training.py` and `api.py`)

Purpose: make an explicit analyst action from calibrated attack probability. A separate policy-validation subset selects thresholds; an uncertain band routes flows to review, and confident benign cases may be suppressed. Test reports compare raw, calibrated, and policy decisions and count review as surfaced traffic. Calibration does not promise fewer false positives. Limitation: no dataset-derived guarantee transfers to live traffic, and probability calibration may drift under a changed network.

### Explainability (`explain.py`)

Purpose: rank flow features by their contribution to the predicted class for one flow. SHAP is selected because the synopsis requires Shapley-based local explanations. Important function: `explain(bundle, row, top_k)`. Viva: contributions describe the local model output relative to an explainer baseline; they are not causal evidence. The generic explainer favors model compatibility but can be slow on high-cardinality features, and one-row background explanations are less stable than a representative saved background set. Improve by storing a representative training background sample and using a model-specific TreeExplainer where appropriate.

### API and dashboard (`api.py`, `dashboard.py`)

Purpose: expose predictions, confidence tier, metrics and explanation to an analyst. FastAPI provides typed endpoints and input errors; Streamlit gives a compact interactive UI. Viva: interface components call the same persisted model bundle, preventing separate prediction logic. Limitation: this is a local research prototype without authentication, audit storage, streaming sensor integration or production hardening.

## Testing and reproducibility

Set a fixed `seed` in `config.yaml`; all splits, sampling, and estimators use it. Run `python -m pytest -q`. Tests cover package/config smoke imports, label conversion, confidence tiers, threshold selection, schema validation, a small synthetic training pipeline, and API prediction. The integration tests use synthetic in-memory rows and do not download the primary dataset.

## Known implementation notes

- The synopsis describes the dataset as roughly 43 features and 18.9 million rows; the official UQ V2 page confirms the same scale and class totals. Source labels/feature columns are validated at run time rather than assumed from this summary.
- Dataset scale can exceed ordinary laptop memory. This prototype reads chunks but gathers the selected modeling subset in memory; start with the bounded sample and increase only after monitoring memory.
- The confidence tier is a review workflow distinction; it does not erase or discard low-confidence predictions from evaluation.
- Current model candidates are Logistic Regression and Random Forest. Calibration is sigmoid. Future scope can add multiclass outputs, temporal splits, cost-sensitive policies, live flow collectors, SIEM integration and online drift monitoring.


## Preprocessing command and decisions

Preprocessing uses the verified `Attack` target from the supplied CSV. The fine-grained source categories are grouped into UQ's seven parent classes; punctuation and case are normalized, and unknown labels stop processing. The exact source text is preserved in `original_label`, the parent class in `normalized_attack_label`, and `binary_target` is 0 for Benign and 1 for attacks. The companion binary `Label` column is excluded as target-like metadata, preventing it from leaking the answer into training features. The source and destination IPv4 address columns are excluded as endpoint identifiers: retaining them would encourage endpoint memorization, expose addresses, and create thousands of one-hot categories without representing generalizable flow behavior. The other documented flow statistics remain eligible features.

The command writes `train.csv`, `validation.csv`, `test.csv`, `source_inspection.json`, `quality_report.json`, `split_metadata.json`, and `run_metadata.json` under `preprocessing.output_dir`. By default it first creates a reproducible, class-preserving sample in `data/cache/` using `preprocessing.development_sample_rows`. `--all-data` opts into loading the complete dataset into memory; start with a sample on student hardware. Original CSV files are never changed.

```powershell
$LabelColumn = "Attack"
python -m nids preprocess --path data/raw/NF-CSE-CIC-IDS2018-v2.csv --label-column $LabelColumn
# Opt into all rows only after checking available RAM and disk:
python -m nids preprocess --path data/raw/NF-CSE-CIC-IDS2018-v2.csv --label-column $LabelColumn --all-data
```

Preprocessing decisions:

- Header normalization applies Unicode normalization, trims whitespace, converts punctuation/spacing to lowercase snake case, and rejects duplicate or colliding names.
- Missing cells and textual missing markers are reported. Numeric-like columns are recognized when at least `numeric_parse_threshold` (default 0.95) of non-missing values parse as numbers. Invalid numeric cells and positive/negative infinity become missing and are reported; train-fitted median imputation handles them later. Categorical features remain categorical and are one-hot encoded.
- Exact duplicate rows are counted and removed by default before splitting, reducing risk that duplicate flows land in separate partitions. Configure `preprocessing.drop_exact_duplicates: false` or pass `--keep-duplicates` only when retaining them is justified.
- Only the selected label, other label-like columns, exact identifiers/endpoints (including this source's IPv4 endpoints), post-event descriptions/outcomes, configured exclusions, and all-null columns are excluded. Directional flow features such as ports and packet counts are retained. Every exclusion and reason appears in the quality report.
- Binary targets and both label metadata columns are never model inputs. All feature transformations are unfitted until the model pipeline fits on training rows; validation/test data are only transformed. Standard scaling is used for Logistic Regression, while Random Forest gets unscaled numeric values. One-hot encoding ignores categories first encountered at inference.
- Splits are stratified by attack subtype when feasible, otherwise by binary target with a recorded warning. No resampling is performed: training estimators use class weights, and validation/test retain the observed class distribution.

Important functions: `preprocess_frame` produces cleaned rows and a quality report; `split_preprocessed_frame` makes reproducible partitions; `build_preprocessor` creates an unfitted sklearn transformer; `split_features_and_labels` enforces target/leakage separation. Limitation: the selected sample (or all rows under `--all-data`) is materialized in memory; chunked CSV intake bounds ingestion memory, but conventional model training requires a bounded modeling frame. A future extension could persist partitions in Parquet and use incremental estimators.

Ingestion aligns multiple CSVs by the union of normalized column names and reports missing/unique fields per file. Missing fields in a file become null values for those rows. Malformed CSV records and header collisions raise errors instead of being silently dropped.

## Model training and evaluation

The training command defaults to `development` mode: it makes a reproducible, class-preserving sample with `training.development_rows` (100,000 by default), then trains on the configured `max_train_rows` cap. Random Forest is the primary baseline (`class_weight=balanced_subsample`); Logistic Regression is the comparison candidate (`class_weight=balanced`). Both candidates are compared using validation data only. Selection first minimizes validation false-positive rate among models meeting the recall target, then favors higher recall and PR-AUC. No test metrics influence the choice. XGBoost is not added because the synopsis does not require it and the existing Random Forest/Logistic Regression comparison is simpler to reproduce on Windows.

```powershell
# Bounded development experiment (default)
python -m nids train --mode development
# Change development sample size
python -m nids train --mode development --max-rows 250000
# Full input and all training rows; this can be expensive and is never the default
python -m nids train --mode full
# Full input, but cap model fitting rows to control memory/time
python -m nids train --mode full --max-train-rows 500000
```

The split is stratified by documented attack subtype when sample counts permit, with binary-label stratification as fallback. Class weighting handles imbalance without synthesizing traffic, and validation/test keep their natural distributions. The dataset is supplied as one or more CSVs, but currently has no verified per-flow capture-group/time field carried into the project. Therefore the reproducible baseline uses stratified flow-level splits; run metadata records this limitation. If a verified capture day, source partition, or time field becomes available, a grouped/time-based holdout should be added to assess generalization between collection periods.

For binary metrics, class 0 means benign and class 1 means attack. The confusion matrix rows are actual `[benign, attack]` and columns predicted `[benign, attack]`. **False-positive rate (FPR) = FP / (FP + TN)**, so it is the share of actual benign flows incorrectly called attacks. **False-negative rate (FNR) = FN / (FN + TP)**, the share of actual attacks incorrectly called benign. Metrics include per-class precision/recall/F1/support, ROC-AUC, average precision (reported as PR-AUC), Brier score, and log loss. Fit, calibration and test-batch inference timing are recorded.

Each run writes latest artifacts at `artifacts/` and immutable, versioned artifacts under `artifacts/runs/<run_id>/`: model and preprocessing joblibs, metrics, validation-only model comparison, split and preprocessing reports, version/config metadata, plus ROC/PR and confusion-matrix PNGs. Metadata records the dataset identifier/version, selected features, estimator parameters, Python/library versions, split sizes, seed and selection configuration. The root model path remains compatible with the API. The `reports/generated/synthetic_training_demo/` folder in this workspace is an executed pipeline demonstration on synthetic balanced data only; its metrics are not results for the real NIDS dataset.

### Executed synthetic demonstration

The synthetic smoke experiment used 420 generated rows, a 60/20/20 split, seed 2026, 40-tree Random Forest plus Logistic Regression, and selected Logistic Regression by validation criteria. Its held-out synthetic test report is saved in `reports/generated/synthetic_training_demo/metrics.json`; plots and full versioned run metadata are alongside it. The resulting synthetic metrics are provided solely to confirm the pipeline emits its outputs; they do not support any performance claim about NF-CSE-CIC-IDS2018-v2. Full dataset results remain pending until the real files are obtained and a run is executed.

## Calibration and confidence-aware alert decisions

Calibration uses a dedicated subset carved from the original validation split. The remaining policy-validation rows select the attack-alert threshold and operating point. The final test split is not used for calibration or threshold selection. `sigmoid` is the default Platt scaling option; `isotonic` is supported only when the separate calibration subset meets `min_isotonic_samples` (default 1,000), because a flexible stepwise mapping can overfit smaller samples. Calibration method, subset sizes, thresholds, seed, and library versions are saved in run metadata. The adapter uses scikit-learn's `FrozenEstimator` API (the project dependency range is 1.6 through 1.7) and keeps a prefit fallback for older environments. See [scikit-learn probability calibration documentation](https://scikit-learn.org/stable/modules/calibration.html).

The test report compares the raw classifier at 0.50, calibrated probabilities at 0.50, and the confidence-aware policy. Raw and calibrated probabilities are compared with Brier score, log loss, and a reliability diagram. Expected calibration error (ECE) uses fixed-width bins over [0,1]: sum across occupied bins of `(bin count / total count) * abs(observed attack fraction - mean predicted P(attack))`. Reliability and calibration metrics describe empirical probability quality; calibration does not itself reduce false positives.

At inference, the alert engine receives calibrated `P(attack)`. Its `calibrated_confidence` is the calibrated probability for the argmax class (`P(attack)` when it is at least 0.5, otherwise `1-P(attack)`). This is a model probability that depends on calibration quality and representative data, not a universal certainty estimate or an out-of-distribution detector. The policy alerts at/above the validation-selected attack threshold; it suppresses a benign prediction only when it falls below the configured review cutoff and meets `minimum_confidence`; remaining cases go to analyst review. Every decision includes predicted class, attack probability, calibrated confidence, thresholds, policy decision, and reason. Test-set per-flow decisions are written to `policy_decisions.csv`.

For the policy, the primary FPR, false-positive count, precision, attack recall, and FNR evaluate **high-priority alerts**. A review is surfaced for analyst attention but is not treated as an automatic attack verdict. The report separately gives attention-level FPR/recall (high priority plus review), alert volume, reviewed benign/attack counts, and benign/attack suppression counts. FNR therefore means attacks not promoted to high priority; `suppressed_attack_count` distinguishes attacks actually suppressed as benign from attacks sent for review. This makes the trade-off between high-priority false alerts, analyst workload, and missed attacks explicit. Operating settings are configurable under `training` in `config.yaml`: `calibration_method`, `calibration_fraction`, `min_isotonic_samples`, `target_recall`, `review_threshold`, `minimum_confidence`, and `ece_bins`.

The API response returns the same policy fields as the offline decision records. The dashboard displays calibration comparisons, policy metrics, reliability and trade-off plots, and supports downloading the per-flow decision CSV. A confidence score alone must not be treated as proof that a flow is safe; model drift, distribution changes, and calibration data size remain limitations.

### Executed calibration and alert-policy demonstration

An executed synthetic-only report is available at `reports/generated/calibration_policy_demo/metrics.json`, with per-flow decisions, validation comparison, metadata, and four plots in the same folder. On this generated 420-row fixture, test-set raw versus calibrated Brier/log-loss/ECE were 0.1361/0.4093/0.1587 and 0.1349/0.4092/0.1352. The 0.50 raw classifier had 9 false positives, FPR 0.2143, attack recall 0.7143, and 12 false negatives. The 0.50 calibrated classifier had 10 false positives, FPR 0.2381, attack recall 0.7381, and 11 false negatives. The confidence-aware policy had 13 high-priority false positives, FPR 0.3095, high-priority attack recall 0.9286, 3 attacks not promoted to high priority, one attack suppressed as benign, and 61 total high-priority-plus-review cases. This run shows a recall/workload trade-off and does not show false-positive reduction. These are synthetic demonstration values, not results on the real NIDS dataset.

## SHAP explainability

Training stores a bounded training-only background sample in the selected model bundle and attempts bounded SHAP artifact generation using the same selected model pipeline. If SHAP is unavailable or the model type has no supported adapter, training and prediction still finish and `shap_manifest.json` records the reason. After installing the declared dependencies, regenerate explanations from the existing saved model without retraining:

```powershell
python -m nids explain --path data/raw/NF-CSE-CIC-IDS2018-v2.csv --max-rows 1000
```

The command samples at most the configured number of source rows, runs the saved preprocessing and calibrated model to identify benign, attack-alert, and uncertain examples, and writes global mean-absolute-SHAP importance plus representative local explanations and plots under `artifacts/explanations/latest/`. `explanations.background_sample_size`, `explanations.explanation_sample_size`, `explanations.top_k`, and `explanations.max_dense_mb` bound work and memory. The API returns a local contribution list with each scored flow; the dashboard displays signed contributions and the global importance artifact when available.

For Random Forest, `TreeExplainer` attributes the base classifier's attack probability. For Logistic Regression, `LinearExplainer` reports additive contributions in raw log-odds space. Both transform inputs using the fitted training preprocessor and preserve transformed names such as `numeric__duration` or `categorical__protocol_tcp`, alongside their source feature names. Positive SHAP values move the explained base-model output toward attack and negative values move it away. The calibrated probability policy still decides the alert; the explanation notes that calibration follows the explained base output. SHAP is an attribution relative to a chosen model and training background, not causal proof, and correlated flow features can share or shift attribution. See the official [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) and [LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html) references.

## Final audit run (synthetic data only)

The final offline run is saved under `reports/generated/synthetic_audit_final/`. It used 360 generated rows, seed 42, a stratified flow-level split, Random Forest, validation-only calibration/threshold selection, and untouched synthetic test rows. The model emitted evaluation JSON/CSV, plots, a versioned bundle, and bounded SHAP artifacts. These numbers are only a software demonstration and say nothing about performance on NF-CSE-CIC-IDS2018-v2.

On that small synthetic test split, the calibrated 0.50 classifier had 1 high-priority false positive and 0 false negatives (attack recall 100%). The confidence-aware policy had 0 high-priority false positives and 7 attacks routed to review rather than high priority (attack recall 85.4%); none of those 7 was suppressed as benign. Two benign rows were also routed to review, so total analyst-attention false positives were 2 and attention-level attack recall was 100%. Calibration diagnostics on this tiny generated split were worse after calibration (Brier 0.0095 to 0.0125; log loss 0.0484 to 0.0795; ECE 0.0416 to 0.0689), illustrating that calibration does not guarantee improved finite-sample results. The run is in `reports/generated/synthetic_audit_final/artifacts/metrics.json`; reliability, confusion, ROC/PR, policy-tradeoff, and SHAP plots are alongside it.

The available focused suite passed **52 tests**. HTTP `TestClient` tests could not be completed in the pre-existing environment because installed FastAPI/Starlette/httpx versions are outside the compatible bounds in `pyproject.toml`; dependency installation could not reach the package index. After installing the declared dependencies in a connected Python 3.11 environment, run the full `python -m pytest -q` command above, including API transport tests.
