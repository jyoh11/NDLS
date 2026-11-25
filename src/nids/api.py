"""FastAPI scoring service for supplied flow features; no packet capture occurs here."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
import uuid

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .explain import explain
from .inference import PredictionError, score_bundle, validate_features
from .logging_config import configure_logging
from .policy import confidence_alert

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:  # Environment variables still work when python-dotenv is absent.
    pass

configure_logging()
LOG = logging.getLogger("nids.api")

ARTIFACTS_DIR = Path(os.getenv("NIDS_ARTIFACTS_DIR", "artifacts"))
BUNDLE_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"


class FlowRequest(BaseModel):
    """A raw flow row matching the trained model's input columns."""

    model_config = ConfigDict(extra="forbid")
    features: dict[str, object] = Field(..., description="Raw named flow features in the trained schema")
    include_explanation: bool = Field(default=False, description="Compute a bounded local SHAP explanation")


class FeatureValidationError(BaseModel):
    feature: str
    code: str
    message: str


class PredictionResponse(BaseModel):
    predicted_class: str
    attack_label: str
    prediction: str
    calibrated_attack_probability: float
    confidence_score: float
    confidence_definition: str
    policy_decision: str
    alert_status: str
    alert_reason: str
    threshold: float
    review_threshold: float
    minimum_confidence: float
    model_name: str
    model_version: str
    feature_validation_errors: list[FeatureValidationError] = Field(default_factory=list)
    explanation: dict | None = None


class ArtifactRuntime:
    """Loads trained artifacts once and exposes readiness without leaking paths."""

    def __init__(self, bundle_path: Path, metrics_path: Path, metadata_path: Path):
        self.bundle_path = Path(bundle_path)
        self.metrics_path = Path(metrics_path)
        self.metadata_path = Path(metadata_path)
        self.bundle: dict | None = None
        self.metrics: dict | None = None
        self.metadata: dict = {}
        self.errors: dict[str, str] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load each artifact at most once for this process/runtime instance."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                candidate = joblib.load(self.bundle_path)
                if not isinstance(candidate, dict):
                    raise ValueError("Model bundle must be a mapping")
                required = {"model", "features", "threshold", "review_threshold", "model_name"}
                missing = sorted(required - set(candidate))
                if missing:
                    raise ValueError(f"Model bundle is missing required fields: {missing}")
                if not candidate["features"] or len(set(candidate["features"])) != len(candidate["features"]):
                    raise ValueError("Model bundle feature schema is empty or contains duplicates")
                if not hasattr(candidate["model"], "predict_proba"):
                    raise ValueError("Saved calibrated model does not implement predict_proba")
                fitted_calibrators = getattr(candidate["model"], "calibrated_classifiers_", None)
                is_calibrated = bool(fitted_calibrators)
                if not is_calibrated:
                    raise ValueError("Saved model bundle does not contain a probability calibrator")
                calibration_method = candidate.get("calibration_method")
                if calibration_method not in {"sigmoid", "isotonic"}:
                    raise ValueError("Saved calibration method is missing or unsupported")
                fitted_methods = {getattr(item, "method", None) for item in fitted_calibrators}
                fitted_methods.discard(None)
                if fitted_methods and fitted_methods != {calibration_method}:
                    raise ValueError("Saved calibration method does not match the fitted calibrator")
                confidence_alert(0.5, candidate["threshold"], candidate["review_threshold"],
                                 candidate.get("minimum_confidence", 0.8))
                fitted_pipeline = candidate.get("uncalibrated_model")
                if fitted_pipeline is not None and hasattr(fitted_pipeline, "feature_names_in_"):
                    if list(fitted_pipeline.feature_names_in_) != list(candidate["features"]):
                        raise ValueError("Saved fitted preprocessing pipeline does not match the model feature schema")
                self.bundle = candidate
            except Exception as exc:
                self.errors["model"] = _safe_artifact_error(exc, "model")
                LOG.error("Could not load trained model bundle (%s)", type(exc).__name__)
            try:
                if self.metrics_path.exists():
                    data = json.loads(self.metrics_path.read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        raise ValueError("Metrics artifact must contain a JSON object")
                    self.metrics = data
                else:
                    self.errors["metrics"] = "Saved evaluation report is missing"
            except Exception as exc:
                self.errors["metrics"] = _safe_artifact_error(exc, "metrics")
                LOG.error("Could not load saved evaluation metrics (%s)", type(exc).__name__)
            try:
                if self.metadata_path.exists():
                    data = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        self.metadata = data
                else:
                    self.errors["metadata"] = "Saved model metadata is missing"
            except Exception as exc:
                self.errors["metadata"] = _safe_artifact_error(exc, "metadata")
                LOG.error("Could not load saved model metadata (%s)", type(exc).__name__)
            if self.bundle is not None:
                bundle_features = list(self.bundle["features"])
                dtype_features = self.bundle.get("feature_dtypes")
                if dtype_features and set(dtype_features) != set(bundle_features):
                    self.errors["model"] = "Saved feature type metadata does not match the model feature schema."
                    self.bundle = None
                selected_features = self.bundle.get("selected_features") if self.bundle else None
                if selected_features and list(selected_features) != bundle_features:
                    self.errors["model"] = "Saved selected features do not match the model input schema."
                    self.bundle = None
            if self.bundle is not None:
                bundle_run = self.bundle.get("run_id")
                metadata_run = self.metadata.get("run_id")
                metrics_run = (self.metrics or {}).get("run_id")
                if metadata_run and bundle_run and metadata_run != bundle_run:
                    self.errors["model"] = "Model bundle and metadata belong to different training runs."
                    self.bundle = None
                elif metrics_run and bundle_run and metrics_run != bundle_run:
                    self.errors["model"] = "Model bundle and evaluation report belong to different training runs."
                    self.bundle = None
                elif (self.metadata.get("selected_features") and
                      list(self.metadata["selected_features"]) != list(self.bundle["features"])):
                    self.errors["model"] = "Model bundle and metadata feature schemas do not match."
                    self.bundle = None
                elif (isinstance(self.metadata.get("training_configuration"), dict) and
                      self.metadata["training_configuration"].get("calibration_method") and
                      self.metadata["training_configuration"]["calibration_method"] != self.bundle.get("calibration_method")):
                    self.errors["model"] = "Model bundle and training metadata calibration settings do not match."
                    self.bundle = None
                operating_point = (self.metrics or {}).get("operating_point_selection", {})
                if self.bundle is not None and isinstance(operating_point, dict):
                    report_threshold = operating_point.get("attack_alert_threshold")
                    if report_threshold is not None and not math.isclose(
                            float(report_threshold), float(self.bundle["threshold"]), rel_tol=0.0, abs_tol=1e-12):
                        self.errors["model"] = "Model bundle and evaluation report thresholds do not match."
                        self.bundle = None
            self._loaded = True

    @property
    def model_ready(self) -> bool:
        self.load()
        return self.bundle is not None

    def readiness(self) -> dict:
        self.load()
        return {
            "status": "ready" if self.bundle is not None else "not_ready",
            "artifacts": {
                "model": self.bundle is not None,
                "evaluation_metrics": self.metrics is not None,
                "model_metadata": bool(self.metadata),
            },
            "errors": dict(self.errors),
        }


def _safe_artifact_error(exc: Exception, kind: str) -> str:
    if isinstance(exc, FileNotFoundError):
        return f"Saved {kind} artifact file was not found."
    if isinstance(exc, PermissionError):
        return f"Saved {kind} artifact could not be accessed."
    if isinstance(exc, json.JSONDecodeError):
        return f"Saved {kind} artifact is not valid JSON."
    if isinstance(exc, (ValueError, TypeError)):
        return str(exc)[:240]
    return f"Saved {kind} artifact could not be loaded ({type(exc).__name__})."

def _cors_origins() -> list[str]:
    configured = os.getenv("NIDS_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(application: FastAPI):
    runtime = ArtifactRuntime(BUNDLE_PATH, METRICS_PATH, METADATA_PATH)
    runtime.load()
    application.state.nids_runtime = runtime
    yield


app = FastAPI(
    title="Explainable NIDS API",
    version="0.2.0",
    description=(
        "Scores caller-supplied network-flow features using saved model artifacts. "
        "This service does not capture packets or monitor a live network."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)


def _runtime(request: Request) -> ArtifactRuntime:
    runtime = getattr(request.app.state, "nids_runtime", None)
    if runtime is None:
        # Supports ASGI transports that do not execute lifespan; still load once.
        runtime = ArtifactRuntime(BUNDLE_PATH, METRICS_PATH, METADATA_PATH)
        runtime.load()
        request.app.state.nids_runtime = runtime
    return runtime


def _require_bundle(runtime: ArtifactRuntime) -> dict:
    runtime.load()
    if runtime.bundle is None:
        reason = runtime.errors.get("model", "Trained model artifact is unavailable")
        raise HTTPException(status_code=503, detail={
            "code": "model_not_ready",
            "message": "The trained calibrated model is not ready. Train the model and verify the artifact bundle.",
            "reason": reason,
        })
    return runtime.bundle


_validate_features = validate_features


def _raise_feature_errors(errors: list[dict]) -> None:
    if errors:
        raise HTTPException(status_code=422, detail={
            "code": "feature_validation_error",
            "message": "Feature values do not match the saved model schema.",
            "feature_validation_errors": errors,
        })


def _predict(bundle: dict, ordered_features: dict[str, object], include_explanation: bool) -> dict:
    try:
        return score_bundle(bundle, ordered_features, include_explanation, explanation_fn=explain)
    except PredictionError as exc:
        if exc.code == "prediction_failed":
            LOG.exception("Model scoring failed")
            raise HTTPException(status_code=500, detail={
                "code": exc.code, "message": str(exc)
            }) from exc
        LOG.error("Saved alert policy configuration is invalid")
        raise HTTPException(status_code=503, detail={
            "code": exc.code, "message": str(exc)
        }) from exc


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        LOG.exception("Unhandled API error request_id=%s path=%s", request_id, request.url.path)
        raise
    response.headers["X-Request-ID"] = request_id
    LOG.info("request_id=%s method=%s path=%s status=%s elapsed_ms=%.1f",
             request_id, request.method, request.url.path, response.status_code,
             (time.perf_counter() - started) * 1000)
    return response


@app.exception_handler(RequestValidationError)
async def request_validation_handler(_request: Request, exc: RequestValidationError):
    # Do not echo the submitted packet/flow values in error responses or logs.
    errors = [{"location": list(item.get("loc", ())), "code": item.get("type", "invalid"),
               "message": item.get("msg", "Invalid request.")} for item in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": {
        "code": "request_validation_error", "message": "Request body failed validation.", "errors": errors
    }})


@app.get("/health", tags=["system"])
def health(request: Request) -> dict:
    """Report artifact readiness; this endpoint does not claim live traffic capture."""
    return _runtime(request).readiness()


@app.get("/model/info", tags=["system"])
def model_info(request: Request) -> dict:
    runtime = _runtime(request)
    bundle = _require_bundle(runtime)
    return {
        "model_name": bundle.get("model_name"),
        "model_version": bundle.get("run_id", bundle.get("artifact_version", "unknown")),
        "dataset_version": bundle.get("dataset_version", runtime.metadata.get("dataset_version", "unknown")),
        "calibration_method": bundle.get("calibration_method", runtime.metadata.get("training_configuration", {}).get("calibration_method")),
        "evaluation_run": runtime.metadata.get("run_id", bundle.get("run_id", "unknown")),
        "dataset_metadata": {
            key: runtime.metadata.get(key)
            for key in ("dataset_rows_before_cleaning", "dataset_rows_after_cleaning", "collection_structure", "created_utc")
            if key in runtime.metadata
        },
        "training_configuration": runtime.metadata.get("training_configuration", {}),
        "features": [{"name": name, "type": bundle.get("feature_dtypes", {}).get(name, "unknown"), "position": index}
                     for index, name in enumerate(bundle["features"])],
        "feature_count": len(bundle["features"]),
        "configuration": _safe_configuration(bundle),
    }


@app.get("/schema", tags=["system"], deprecated=True)
def schema_compat(request: Request) -> dict:
    """Compatibility alias for older dashboard builds; use /model/info."""
    bundle = _require_bundle(_runtime(request))
    return {"features": list(bundle["features"])}


def _safe_configuration(bundle: dict) -> dict:
    return {
        "attack_threshold": bundle.get("threshold"),
        "review_threshold": bundle.get("review_threshold"),
        "minimum_confidence": bundle.get("minimum_confidence", 0.8),
        "calibration_method": bundle.get("calibration_method"),
        "explanations": {
            "top_k": bundle.get("explanation_config", {}).get("top_k", 8),
            "max_dense_mb": bundle.get("explanation_config", {}).get("max_dense_mb", 512.0),
        },
    }


@app.get("/config", tags=["system"])
def safe_config(request: Request) -> dict:
    """Expose policy and explanation settings only; secrets and filesystem paths are omitted."""
    bundle = _require_bundle(_runtime(request))
    return _safe_configuration(bundle)


@app.get("/metrics", tags=["evaluation"])
def metrics(request: Request) -> dict:
    """Return the saved held-out evaluation report; this is not a live metric."""
    runtime = _runtime(request)
    if runtime.metrics is None:
        raise HTTPException(status_code=404, detail={
            "code": "metrics_unavailable", "message": "No saved evaluation report is available."
        })
    return {"source": "saved_held_out_evaluation_report", "metrics": runtime.metrics}


def _score_request(request: Request, body: FlowRequest, include_explanation: bool) -> dict:
    bundle = _require_bundle(_runtime(request))
    ordered, errors = _validate_features(body.features, bundle)
    _raise_feature_errors(errors)
    return _predict(bundle, ordered, include_explanation)


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(body: FlowRequest, request: Request) -> dict:
    """Score one supplied flow feature row with the persisted calibrated model."""
    return _score_request(request, body, body.include_explanation)


@app.post("/explain", response_model=PredictionResponse, tags=["explainability"])
def explain_flow(body: FlowRequest, request: Request) -> dict:
    """Score and explain one supplied row; explanation failure never blocks scoring."""
    result = _score_request(request, body, include_explanation=True)
    return result
