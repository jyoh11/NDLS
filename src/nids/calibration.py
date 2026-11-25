"""Probability calibration and calibration-quality diagnostics.

Calibration is fitted on a dedicated, held-out calibration subset. The final
test split is used only to compare raw and calibrated probabilities.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold


def calibrate_fitted_estimator(estimator, X_calibration, y_calibration,
                               method: str = "sigmoid", min_isotonic_samples: int = 1000):
    """Calibrate a train-fitted classifier on rows disjoint from model fitting.

    Supported calibration maps are Platt's sigmoid and isotonic regression.
    Isotonic needs a reasonably large independent calibration set; this helper
    rejects small sets instead of returning a high-variance calibration map.
    """
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("Calibration method must be 'sigmoid' or 'isotonic'")
    y = np.asarray(y_calibration)
    if len(y) != len(X_calibration) or len(y) == 0:
        raise ValueError("Calibration inputs must contain the same non-zero number of rows")
    if len(np.unique(y)) < 2:
        raise ValueError("Calibration set must contain benign and attack examples")
    class_counts = np.unique(y, return_counts=True)[1]
    if class_counts.min() < 2:
        raise ValueError("Calibration set needs at least two rows from each class")
    if method == "isotonic" and len(y) < min_isotonic_samples:
        raise ValueError(
            f"Isotonic calibration requires at least {min_isotonic_samples} independent calibration rows; "
            "use sigmoid calibration or increase the calibration set."
        )

    # FrozenEstimator is the supported API in newer sklearn. The prefit option
    # is retained for supported earlier 1.x versions.
    try:
        from sklearn.frozen import FrozenEstimator
    except ImportError:  # scikit-learn < 1.6
        calibrator = CalibratedClassifierCV(estimator=estimator, method=method, cv="prefit")
    else:
        # FrozenEstimator is not refit. CalibratedClassifierCV still obtains
        # out-of-fold scores from it; bound folds to the smallest class so
        # small but valid development fixtures do not fail at sklearn's default 5.
        folds = min(5, int(class_counts.min()))
        cv = StratifiedKFold(n_splits=folds, shuffle=False)
        calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(estimator), method=method, cv=cv)
    return calibrator.fit(X_calibration, y)


def positive_probability(estimator, X, positive_label: int = 1) -> np.ndarray:
    """Get probability for the requested positive class independent of order."""
    classes = np.asarray(estimator.classes_)
    matches = np.flatnonzero(classes == positive_label)
    if len(matches) != 1:
        raise ValueError(f"Estimator classes {classes.tolist()} do not contain positive label {positive_label!r}")
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)[:, int(matches[0])]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Estimator returned missing or invalid probabilities")
    return probabilities


def calibration_diagnostics(y_true, probabilities, n_bins: int = 10) -> dict:
    """Return Brier, log loss, fixed-width ECE, and reliability-curve bins.

    ECE definition: partition [0, 1] into ``n_bins`` equal-width bins and sum
    (bin population / total population) * abs(mean outcome - mean probability).
    Empty bins are omitted from the reported bin list.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("Labels and probabilities must be equal-length, non-empty one-dimensional arrays")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Calibration diagnostics require binary labels encoded as 0/1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    bins = []
    ece = 0.0
    for index in range(n_bins):
        mask = assignments == index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_probability = float(p[mask].mean())
        observed_rate = float(y[mask].mean())
        ece += (count / len(y)) * abs(observed_rate - mean_probability)
        bins.append({
            "lower": float(edges[index]), "upper": float(edges[index + 1]),
            "count": count, "mean_probability": mean_probability,
            "observed_positive_rate": observed_rate,
        })
    return {
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "expected_calibration_error": float(ece),
        "ece_definition": f"Fixed-width bins={n_bins}; sum(bin_count/N * abs(observed positive rate - mean probability)).",
        "n_bins": n_bins,
        "reliability_bins": bins,
    }
