# NIDS Project Viva Guide

## Project in one minute

This project builds a binary network intrusion detector that classifies a flow as benign or attack. It compares Random Forest with Logistic Regression, calibrates model probabilities on data separate from model fitting, and selects an alert operating threshold on validation data. Its confidence-aware policy produces attack alerts, benign suppressions, or analyst-review decisions. SHAP can describe which input features moved a supported model output toward or away from attack. A FastAPI service scores supplied flow records, and a Streamlit dashboard presents the prediction and saved evaluation report.

The research question is whether calibrated, confidence-aware handling can reduce unnecessary high-priority alerts while retaining attack detection. This is an empirical question. It cannot be answered affirmatively until results on the real synopsis dataset support the claim.

## Dataset and labels

The synopsis specifies **NF-CSE-CIC-IDS2018-v2**, the 43-feature NetFlow V2 data. The supplied official UQ archive was checksum-verified. Its CSV has 18,893,708 records and 45 columns: 43 model-input candidates plus numeric `Label` and detailed `Attack`. A full streaming scan found 16,635,567 benign (`Label=0`) and 2,258,141 attack (`Label=1`) records. `Attack` is configured as the target so the source's detailed attack categories can be retained. The separate numeric `Label` is excluded from the model features to prevent leakage.

- `Benign` → binary class 0
- `BruteForce`, `Bot`, `DoS`, `DDoS`, `Infiltration`, `Web Attacks` → binary class 1

The actual `Attack` values are finer-grained than UQ's seven documented parent classes. They are grouped as follows: SSH-Bruteforce, FTP-BruteForce, Brute Force -Web and Brute Force -XSS → BruteForce; DoS attacks-Hulk, SlowHTTPTest, GoldenEye and Slowloris → DoS; DDoS attacks-LOIC-HTTP, DDOS attack-HOIC and DDOS attack-LOIC-UDP → DDoS; Infilteration → Infiltration; SQL Injection → Web Attacks. Benign and Bot remain their own classes. The original text is retained in `original_label`, the parent class in `normalized_attack_label`, and `binary_target` is 0 for benign and 1 for attacks. Unknown labels stop preprocessing. The source and destination IPv4 fields are excluded as endpoint identifiers to prevent endpoint memorization and high-cardinality categorical expansion; flow statistics and directional packet/byte fields remain available. Development-demo labels are synthetic and must not be described as dataset observations.

## Method and data boundaries

1. **Ingestion:** discover CSV/CSV.GZ files, inspect schemas and label candidates, reject malformed records, and report missing columns and values. Small samples can be generated without downloading all files.
2. **Preprocessing:** normalize column names; analyze duplicates; parse numeric-like values; count malformed/infinite values; and remove only target/label metadata, clear identifiers, post-event fields, all-null columns, and explicit exclusions. Categorical flow fields are retained for encoding.
3. **Leakage control:** split cleaned flows into stratified train, validation, and test partitions. The fitted imputer, scaler, and one-hot encoder are inside the estimator pipeline and fit only on training rows. The validation partition is divided into calibration and policy-selection subsets. The test partition is held untouched until final evaluation.
4. **Models:** Random Forest is the primary baseline. Logistic Regression is the comparison model. Class weights address imbalance; validation and test distributions are not resampled.
5. **Calibration:** fit sigmoid/Platt calibration on the separate calibration subset. Isotonic is allowed only with the configured minimum sample count. Calibration improves probability reliability if supported by diagnostics; it does not guarantee fewer false positives.
6. **Alert policy:** select an attack threshold on the policy-validation subset for the configured attack-recall target. The confidence policy separates high-priority alert, confident benign suppression, and review. The final test set is not used to choose thresholds.
7. **Evaluation:** compare the uncalibrated 0.50 classifier, calibrated 0.50 classifier, and confidence-aware policy on the same held-out test rows.
8. **Serving and explanation:** save a single model bundle containing the calibrated model, fitted preprocessing pipeline, raw feature schema/types, policy thresholds, and training-only SHAP background. Both `nids predict` and FastAPI call the shared inference function. SHAP describes model contributions; it is not causal evidence.

## Algorithms and decisions

### Why Random Forest and Logistic Regression?

Random Forest is the primary nonlinear baseline and handles mixed flow patterns without numeric scaling. Logistic Regression is a simpler linear comparison and benefits from standardized numeric fields. The model selector uses validation results only. XGBoost was not added because the synopsis lists it as one possible method, not a required choice; the current two-model setup is simpler to reproduce on Windows and inspect in a viva.

### What does calibration do?

Calibration maps scores from the fitted classifier to probabilities intended to match observed class frequencies. For example, among many predictions near 0.8, about 80% should be attacks under representative conditions. Brier score, log loss, reliability bins, and fixed-width expected calibration error (ECE) measure that behavior. A lower calibration error does not imply fewer alerts; the policy and threshold determine alert counts.

### What does the confidence score mean?

The API defines it as the calibrated probability assigned to the model's argmax class: `P(attack)` when attack probability is at least 0.5, otherwise `1 - P(attack)`. It is not a universal certainty measure, an out-of-distribution detector, or a guarantee of safety. Because the policy threshold may differ from 0.5, the argmax `predicted_class` and operational `policy_decision` can differ; the alert decision is the action an analyst should follow.

## Metrics to explain

Let `TN` be benign correctly classified, `FP` benign classified as high-priority attack, `FN` attack not classified as high-priority, and `TP` attack classified as high-priority.

- **Precision:** `TP / (TP + FP)`; fraction of high-priority attack calls that are actual attacks.
- **Attack recall:** `TP / (TP + FN)`; fraction of actual attacks promoted to high-priority alerts.
- **F1:** harmonic mean of precision and recall.
- **False-positive rate:** `FP / (FP + TN)`; false high-priority alerts divided by all actual benign flows.
- **False-negative rate:** `FN / (FN + TP)`; attacks not promoted to high priority divided by all actual attacks. Review and benign-suppression counts are reported separately so an uncertain review is not mistaken for a confident benign result.
- **Alert volume:** high-priority alerts plus review cases when referring to analyst attention. It is different from high-priority alert count.
- **Brier score and log loss:** proper scoring rules for probability quality; lower is better on the evaluated sample.
- **ECE:** fixed-width probability bins; sum `bin_size / N * abs(observed attack rate - mean predicted attack probability)` over nonempty bins.

The saved report compares baseline, calibrated classifier, and policy on identical final test labels. A policy may lower FPR while increasing missed attacks or reviews. It may also increase FPR to meet a high recall target. Use the generated report values and discuss the complete trade-off; do not present synthetic-demo scores as real-data findings.

## Explaining SHAP

SHAP contributions allocate the supported base model's output relative to a bounded training-only background sample. Positive signed values move the explained output toward attack; negative values move it away. Feature correlations, the chosen background, and the model affect attributions. The explanations do not prove that a feature caused the traffic to be malicious. The calibrated probability and policy make the operational decision; SHAP is an explanation aid.

## Architecture for a viva diagram

```text
CSV files / synthetic demo
        ↓
ingest + schema validation
        ↓
label/feature separation + leakage exclusions
        ↓
train / validation / untouched test
        ├── train: fit preprocessing + RF/LR
        ├── validation calibration subset: fit calibrator
        └── validation policy subset: choose threshold
        ↓
final test metrics + plots + versioned artifact bundle
        ↓
shared inference contract ← CLI prediction / FastAPI
        ├── calibrated probability → confidence policy
        └── same saved feature pipeline → optional SHAP
        ↓
Streamlit analyst dashboard
```

## Limitations and future scope

- The full primary dataset has not been trained/evaluated end to end. A 100,000-row class-preserving development sample was evaluated; see the saved real-data report and trade-offs in the README. Do not generalize that sample's metrics to all 18.9 million flows.
- Current splits are stratified at flow level. A temporal, capture-day, or source-group holdout should be used if verified grouping metadata is present; otherwise related flows may cross partitions.
- Training materializes the selected modeling data in memory. Start with a bounded sample and monitor RAM before increasing it.
- Model calibration and thresholds can drift when traffic changes. A benchmark result does not establish performance on live networks.
- The API accepts supplied flow features only. No packet capture or live telemetry collector exists.
- Authentication, persistent alert storage, SIEM integration, online learning, and deployment hardening are outside this prototype.

The synopsis' future scope includes real-time flow collection, additional datasets and cross-dataset testing, adaptive thresholds, unseen/zero-day attack research, SIEM integration, other explainability methods, cloud/edge deployment, and online learning. Treat these as future work, not implemented capabilities.

## Common viva questions

**Why use a separate calibration set?** To fit the probability mapping on rows the classifier did not train on, avoiding optimistic probabilities from training data.

**Why is the test set untouched?** It estimates final generalization after model and operating-point choices. Tuning on it would make the reported test result optimistic.

**Does calibration reduce false positives?** Not by itself. Calibration changes probability reliability. The threshold and policy change which flows become alerts, and that trade-off must be measured.

**How do you prevent target leakage?** Target and label-like metadata, identifiers, and post-event fields are excluded from the feature matrix; the saved API schema accepts only the model's selected features.

**What if the user gives an extra field or wrong type?** The shared schema validator rejects unexpected, missing, malformed, non-finite, and incompatible fields before scoring.

**Can an attack alert have a benign argmax class?** Yes, if the validation-selected alert cutoff is below 0.5. The dashboard shows the policy decision separately from the ordinary 0.50 argmax class.

**What does a local explanation tell the analyst?** Which transformed input contributions move the base model output toward or away from attack for that row and background. It does not establish causation.
