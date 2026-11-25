"""Offline probability calibration diagnostics and compatibility tests."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from nids.calibration import (
    calibration_diagnostics,
    calibrate_fitted_estimator,
    positive_probability,
)


def test_sigmoid_calibration_uses_held_out_rows_and_returns_probabilities():
    train_x = pd.DataFrame({"x": [-3, -2, -1, 1, 2, 3]})
    train_y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression(random_state=3).fit(train_x, train_y)
    calibration_x = pd.DataFrame({"x": [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]})
    calibration_y = np.array([0, 0, 0, 1, 1, 1])
    calibrated = calibrate_fitted_estimator(model, calibration_x, calibration_y, "sigmoid")
    probability = positive_probability(calibrated, calibration_x)
    assert probability.shape == (6,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()


def test_isotonic_refuses_too_small_independent_calibration_set():
    x = pd.DataFrame({"x": [-2, -1, 1, 2]})
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression().fit(x, y)
    with pytest.raises(ValueError, match="at least 1000"):
        calibrate_fitted_estimator(model, x, y, "isotonic")


def test_isotonic_calibration_supported_when_sample_is_large_enough():
    train_x = pd.DataFrame({"x": np.r_[-np.arange(1, 21), np.arange(1, 21)]})
    train_y = np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)]
    model = LogisticRegression(random_state=1).fit(train_x, train_y)
    calibration_x = pd.DataFrame({"x": np.linspace(-5, 5, 1000)})
    calibration_y = (calibration_x["x"].to_numpy() > 0).astype(int)
    calibrated = calibrate_fitted_estimator(model, calibration_x, calibration_y, "isotonic")
    probability = positive_probability(calibrated, calibration_x.iloc[[0, -1]])
    assert probability.shape == (2,)
    assert probability[0] < probability[1]


def test_fixed_width_ece_definition_and_validation():
    report = calibration_diagnostics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], n_bins=2)
    assert report["brier_score"] >= 0
    assert report["log_loss"] > 0
    assert report["expected_calibration_error"] == pytest.approx(0.25)
    assert len(report["reliability_bins"]) == 2
    assert "Fixed-width bins=2" in report["ece_definition"]


@pytest.mark.parametrize("probabilities", [[float("nan")], [float("inf")], [-0.1], [1.1]])
def test_calibration_diagnostics_reject_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        calibration_diagnostics([0], probabilities)
