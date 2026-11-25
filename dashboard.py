"""Streamlit analyst UI. All prediction work is delegated to the FastAPI service."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from nids.dashboard_data import (
    DashboardInputError,
    MAX_BATCH_ROWS,
    alert_queue_category,
    get_json,
    parse_flow_text,
    parse_uploaded_records,
    predict_records,
)

st.set_page_config(page_title="Explainable NIDS | Analyst Console", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1500px;}
  [data-testid="stMetric"] {background: #101c2a; border: 1px solid #23354a; padding: 14px; border-radius: 10px;}
  section[data-testid="stSidebar"] {background: #0b1420;}
  .nids-caption {color:#9fb2c7; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

API_URL = st.sidebar.text_input("FastAPI service", os.getenv("NIDS_API_URL", "http://127.0.0.1:8000"))
ARTIFACT_DIR = Path(os.getenv("NIDS_ARTIFACTS_DIR", "artifacts"))
st.sidebar.markdown("**Research prototype**  \nScores supplied flow features; it does not capture live network traffic.")
page = st.sidebar.radio("Workspace", ["Overview", "Analyze flows", "Alert review", "Model evaluation", "Dataset & model"])


@st.cache_data(ttl=5, show_spinner=False)
def fetch_health(api_url: str):
    return get_json(api_url, "/health")


@st.cache_data(ttl=30, show_spinner=False)
def fetch_model(api_url: str):
    return get_json(api_url, "/model/info")


@st.cache_data(ttl=30, show_spinner=False)
def fetch_metrics(api_url: str):
    return get_json(api_url, "/metrics")


def backend_state():
    try:
        health = fetch_health(API_URL)
        model = fetch_model(API_URL) if health.get("artifacts", {}).get("model") else None
        metrics = fetch_metrics(API_URL) if health.get("artifacts", {}).get("evaluation_metrics") else None
        return health, model, metrics, None
    except requests.RequestException as exc:
        return None, None, None, f"Could not reach the FastAPI service at {API_URL}: {exc}"
    except (ValueError, KeyError) as exc:
        return None, None, None, f"The backend returned an unreadable response: {exc}"


health, model_info, metrics_response, backend_error = backend_state()
if backend_error:
    st.error(backend_error)
elif health and health.get("status") != "ready":
    errors = "; ".join(health.get("errors", {}).values()) or "Train the model and make its artifacts available to the API."
    st.warning(f"API is running, but model artifacts are not ready. {errors}")
elif health:
    st.sidebar.success("Model ready")
else:
    st.sidebar.info("Waiting for API")

if "nids_results" not in st.session_state:
    st.session_state.nids_results = []


def metric_value(container, title, metrics, key, fmt=".3f"):
    value = metrics.get(key) if metrics else None
    container.metric(title, format(value, fmt) if isinstance(value, (int, float)) else "Not available")


def render_explanation(result: dict, key: str):
    explanation = result.get("explanation")
    if not explanation:
        st.info("No explanation was returned for this prediction.")
        return
    if explanation.get("error"):
        st.warning("SHAP explanation unavailable: " + explanation["error"])
        return
    contributions = explanation.get("contributions", [])
    if not contributions:
        st.info("The explanation contains no feature contributions.")
        return
    frame = pd.DataFrame(contributions)
    feature_col = "feature" if "feature" in frame else "transformed_feature"
    value_col = "shap_value" if "shap_value" in frame else None
    if value_col:
        frame["direction"] = frame[value_col].map(lambda value: "Toward attack" if value > 0 else "Away from attack")
        st.bar_chart(frame.set_index(feature_col)[value_col], color="#ff6b6b", horizontal=True)
        st.dataframe(frame[[feature_col, value_col, "direction"]], hide_index=True, use_container_width=True)
    else:
        st.dataframe(frame, hide_index=True, use_container_width=True)
    st.caption(explanation.get("interpretation_note", "SHAP values describe feature contribution for this model; they do not establish causation."))


def render_prediction(result: dict, row_number: int | None = None):
    label = f"Flow {row_number} · " if row_number else ""
    category = alert_queue_category(result)
    if category == "Uncertain / review":
        st.warning(f"{label}Needs analyst review · {result.get('alert_reason', 'Uncertain prediction')}")
    elif category == "Attack alert":
        model_class = result.get("attack_label", result.get("predicted_class", "Unknown"))
        st.error(f"{label}Policy attack alert · model class: {model_class} · {result.get('alert_reason', 'Policy alert')}")
    else:
        st.success(f"{label}Benign prediction · {result.get('alert_reason', 'No alert')}")
    cols = st.columns(4)
    cols[0].metric("Model class (0.50 argmax)", result.get("attack_label", result.get("predicted_class", "Unknown")))
    prob = result.get("calibrated_attack_probability")
    cols[1].metric("Calibrated P(attack)", f"{prob:.3f}" if isinstance(prob, (float, int)) else "Unavailable")
    conf = result.get("confidence_score")
    cols[2].metric("Confidence score", f"{conf:.3f}" if isinstance(conf, (float, int)) else "Unavailable")
    cols[3].metric("Policy decision", str(result.get("policy_decision", "Unavailable")).replace("_", " ").title())
    st.caption(result.get("confidence_definition", "Confidence is model probability for the predicted class; it is not an out-of-distribution score."))
    st.caption(f"Model {result.get('model_name', 'unknown')} · {result.get('model_version', 'unknown')}")
    if result.get("explanation"):
        with st.expander("Inspect SHAP feature contributions", expanded=True):
            render_explanation(result, f"exp-{row_number or 'one'}")


if page == "Overview":
    st.title("🛡️ Explainable NIDS")
    st.caption("Analyst console · Confidence-aware intrusion detection")
    if health:
        ready = health.get("status") == "ready"
        a, b, c = st.columns(3)
        a.metric("Service", "Online" if health else "Unavailable")
        b.metric("Model artifacts", "Ready" if health.get("artifacts", {}).get("model") else "Missing")
        c.metric("Saved evaluation", "Available" if health.get("artifacts", {}).get("evaluation_metrics") else "Unavailable")
    if metrics_response:
        saved = metrics_response.get("metrics", {})
        st.subheader("Held-out evaluation · saved report")
        st.caption(f"Historical test-set values from {model_info.get('evaluation_run', 'saved run') if model_info else 'saved run'}. Current dashboard predictions are not included in these metrics.")
        cols = st.columns(5)
        for col, (title, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Attack recall", "attack_recall"), ("F1", "f1"), ("False-positive rate", "false_positive_rate")]):
            metric_value(col, title, saved, key)
        st.caption("False-positive rate = benign flows incorrectly classified as attack / all actual benign flows.")
        if model_info:
            st.write(f"**Model:** {model_info.get('model_name')} · **Dataset:** {model_info.get('dataset_version')} · **Calibration:** {model_info.get('calibration_method', 'not recorded')}")
    else:
        st.info("Saved evaluation metrics are unavailable. Train a model to populate the report; no metrics are substituted.")
    st.subheader("Current session activity")
    st.caption("These are only predictions submitted during this dashboard session, not network-wide live counts.")
    results = st.session_state.nids_results
    alerts = sum(alert_queue_category(item) == "Attack alert" for item in results)
    reviews = sum(alert_queue_category(item) == "Uncertain / review" for item in results)
    benign = sum(alert_queue_category(item) == "Benign prediction" for item in results)
    a, b, c, d = st.columns(4)
    a.metric("Submitted flows", len(results)); b.metric("Attack predictions", alerts); c.metric("Uncertain / review", reviews); d.metric("Benign predictions", benign)

elif page == "Analyze flows":
    st.title("Analyze network flows")
    if not model_info:
        st.info("Start the API and ensure the trained model artifacts are ready before submitting flows.")
    else:
        features = model_info.get("features", [])
        st.caption(f"Expected schema: {len(features)} features in the model. Maximum {MAX_BATCH_ROWS} records per upload, 2 MB per file.")
        with st.expander("Model feature schema"):
            st.dataframe(pd.DataFrame(features), hide_index=True, use_container_width=True)
        input_mode = st.radio("Input method", ["Enter one flow as JSON", "Upload CSV or JSON"], horizontal=True)
        rows = None
        if input_mode == "Enter one flow as JSON":
            template = {feature["name"]: 0 if feature.get("type") == "numeric" else "" for feature in features}
            raw = st.text_area("Feature values", json.dumps(template, indent=2), height=300,
                               help="Paste one JSON object with every trained feature name.")
            if st.button("Analyze flow", type="primary"):
                try:
                    rows = [parse_flow_text(raw, features)]
                except DashboardInputError as exc:
                    st.error(str(exc))
        else:
            uploaded = st.file_uploader("Select CSV or JSON flow records", type=["csv", "json"], accept_multiple_files=False)
            if uploaded:
                st.caption(f"Selected: {uploaded.name} · {uploaded.size / 1024:.1f} KB")
            if st.button("Validate and analyze upload", type="primary") and uploaded:
                try:
                    rows = parse_uploaded_records(uploaded.name, uploaded.getvalue(), features)
                except DashboardInputError as exc:
                    st.error(str(exc))
        if rows:
            explain_rows = st.checkbox("Request local SHAP explanation for each flow", value=True)
            try:
                with st.spinner(f"Sending {len(rows)} flow(s) to FastAPI…"):
                    results = predict_records(API_URL, rows, explain=explain_rows)
                st.session_state.nids_results.extend(results)
                st.success(f"FastAPI analyzed {len(results)} flow(s).")
                for index, result in enumerate(results, start=1):
                    with st.container(border=True):
                        render_prediction(result, index if len(results) > 1 else None)
            except DashboardInputError as exc:
                st.error(str(exc))
            except requests.HTTPError as exc:
                response = exc.response
                detail = "The backend rejected or could not process the request."
                if response is not None:
                    try:
                        payload = response.json().get("detail", {})
                        detail = payload.get("message", detail) if isinstance(payload, dict) else str(payload)
                        reason = payload.get("reason") if isinstance(payload, dict) else None
                        if reason:
                            detail += " " + str(reason)
                    except (ValueError, AttributeError):
                        pass
                    st.error(f"FastAPI returned HTTP {response.status_code}: {detail}")
                else:
                    st.error(f"The API request failed: {exc}")
            except requests.RequestException as exc:
                st.error(f"The API request failed. Check that FastAPI is running: {exc}")
            except (ValueError, KeyError) as exc:
                st.error(f"Could not interpret the API response: {exc}")

elif page == "Alert review":
    st.title("Alert review queue")
    st.caption("Review contains only records submitted in this dashboard session. No persistent alert store or live traffic feed is configured.")
    results = st.session_state.nids_results
    if not results:
        st.info("No predictions in this session yet. Use Analyze flows to submit flow features.")
    else:
        rows = []
        for number, result in enumerate(results, start=1):
            decision = str(result.get("policy_decision", ""))
            group = alert_queue_category(result)
            rows.append({"Session flow": number, "Queue": group, "Attack probability": result.get("calibrated_attack_probability"), "Confidence": result.get("confidence_score"), "Decision": decision, "Reason": result.get("alert_reason", "")})
        category = st.selectbox("Show", ["All", "Attack alert", "Benign prediction", "Uncertain / review"])
        frame = pd.DataFrame(rows)
        if category != "All":
            frame = frame[frame["Queue"] == category]
        st.dataframe(frame, hide_index=True, use_container_width=True)
        selected = st.selectbox("Inspect a flow", [row["Session flow"] for row in rows])
        render_prediction(results[selected - 1], selected)
        if st.button("Clear session predictions"):
            st.session_state.nids_results = []
            st.rerun()

elif page == "Model evaluation":
    st.title("Model evaluation")
    if not metrics_response:
        st.warning("No saved evaluation report is available from the backend. Metrics and plots are not generated from current session predictions.")
    else:
        metrics = metrics_response["metrics"]
        st.caption(f"Source: saved held-out test report · Run {model_info.get('evaluation_run', 'unknown') if model_info else 'unknown'} · Not live performance.")
        cols = st.columns(5)
        for col, (title, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Attack recall", "attack_recall"), ("F1-score", "f1"), ("False-positive rate", "false_positive_rate")]):
            metric_value(col, title, metrics, key)
        st.caption("FPR = FP / (FP + TN): benign flows incorrectly labeled as attacks divided by all benign flows in the held-out test set.")
        st.subheader("Confusion matrix")
        matrix = metrics.get("confusion_matrix", {}).get("rows_actual_columns_predicted")
        if matrix and len(matrix) == 2 and all(len(row) == 2 for row in matrix):
            st.dataframe(pd.DataFrame(matrix, index=["Actual benign", "Actual attack"], columns=["Predicted benign", "Predicted attack"]), use_container_width=True)
        else:
            st.info("Confusion matrix is not included in this report.")
        st.subheader("Baseline, calibrated classifier, and confidence policy")
        comparison = metrics.get("decision_comparison", {})
        flat = []
        for name, item in comparison.items():
            if isinstance(item, dict):
                flat.append({"Operating point": name.replace("_", " "), **{key: item.get(key) for key in ("threshold", "false_positive_rate", "false_positive_count", "precision", "attack_recall", "false_negative_rate", "alert_volume")}})
        if flat:
            st.dataframe(pd.DataFrame(flat), hide_index=True, use_container_width=True)
        else:
            st.info("Decision comparison is not included in the saved report.")
        st.subheader("Probability calibration")
        st.json(metrics.get("calibration_comparison", {}))
        st.caption("Calibration diagnostics compare probability reliability. Calibration itself does not guarantee fewer false positives.")
        for plot_name, relative in metrics.get("plots", {}).items():
            plot_path = ARTIFACT_DIR / relative
            if plot_path.is_file() and plot_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                st.image(str(plot_path), caption=plot_name.replace("_", " ").title(), use_container_width=True)
        explanation_status = metrics.get("explanations", {})
        if explanation_status.get("status") == "complete":
            st.subheader("Global SHAP feature importance")
            global_plot = ARTIFACT_DIR / "shap_global_importance.png"
            global_json = ARTIFACT_DIR / "shap_global_importance.json"
            if global_plot.is_file():
                st.image(str(global_plot), caption="Mean absolute SHAP contribution on a bounded held-out sample.", use_container_width=True)
            if global_json.is_file():
                with global_json.open(encoding="utf-8") as stream:
                    global_data = json.load(stream)
                importance = global_data.get("feature_importance", [])
                if importance:
                    st.dataframe(pd.DataFrame(importance[:30]), hide_index=True, use_container_width=True)
                if global_data.get("interpretation_note"):
                    st.caption(global_data["interpretation_note"])
        elif explanation_status:
            st.info(f"Global SHAP report: {explanation_status.get('status')}. {explanation_status.get('reason', '')}")
        decisions_path = ARTIFACT_DIR / metrics.get("policy_decisions_file", "")
        if decisions_path.is_file():
            st.download_button("Download held-out per-flow policy decisions", decisions_path.read_bytes(), "policy_decisions.csv", "text/csv")

elif page == "Dataset & model":
    st.title("Dataset and model information")
    if not model_info:
        st.info("Model information is unavailable until the backend loads a trained model.")
    else:
        left, right = st.columns(2)
        left.subheader("Model")
        left.write({"model": model_info.get("model_name"), "version": model_info.get("model_version"), "evaluation run": model_info.get("evaluation_run"), "calibration": model_info.get("calibration_method"), "feature count": model_info.get("feature_count")})
        right.subheader("Dataset run metadata")
        right.write({"dataset version": model_info.get("dataset_version"), **model_info.get("dataset_metadata", {})})
        st.subheader("Training configuration")
        st.json(model_info.get("training_configuration", {}))
        st.subheader("Feature schema and order")
        st.dataframe(pd.DataFrame(model_info.get("features", [])), hide_index=True, use_container_width=True)
        if metrics_response:
            st.caption("Evaluation metrics shown here are the saved held-out test results returned by the backend, not current session outcomes.")
        else:
            st.warning("No saved evaluation report is available.")
