"""Frozen R5-3A scaler and LogisticRegression factories."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from r5_3a_features import validate_design_matrix


def make_scaler() -> StandardScaler:
    return StandardScaler(copy=True, with_mean=True, with_std=True)


def make_classifier() -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
    )


def fit_calibration_model(
    calibration_features: np.ndarray, calibration_labels: np.ndarray
) -> tuple[StandardScaler, LogisticRegression, np.ndarray]:
    features = np.asarray(calibration_features, dtype=np.float64)
    labels = np.asarray(calibration_labels, dtype=np.int64)
    validate_design_matrix(features)
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("calibration labels must align with calibration features")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("calibration rows must contain both binary classes")
    scaler = make_scaler()
    transformed = scaler.fit_transform(features)
    model = make_classifier()
    model.fit(transformed, labels)
    return scaler, model, transformed


def addition_probability(model: Any, scaled_features: np.ndarray) -> np.ndarray:
    """Use predict_proba only and select the column whose class identity is 1."""
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == 1)
    if matches.size != 1:
        raise ValueError("classifier classes_ must contain Addition class 1 exactly once")
    probabilities = np.asarray(model.predict_proba(scaled_features), dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != classes.size:
        raise ValueError("predict_proba output shape does not match classes_")
    result = probabilities[:, int(matches[0])]
    if not np.all(np.isfinite(result)):
        raise ValueError("Addition probabilities must be finite")
    return result
