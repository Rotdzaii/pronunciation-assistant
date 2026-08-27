"""Reusable frozen R5-3A speaker-LOSO control flow."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from r5_3a_classifier import addition_probability, fit_calibration_model
from r5_3a_features import FEATURE_ORDER, validate_design_matrix
from r5_3a_threshold import apply_threshold, select_threshold


def _matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    result = np.asarray([[row[name] for name in FEATURE_ORDER] for row in rows], dtype=np.float64)
    validate_design_matrix(result)
    return result


def event_from_decision(row: Mapping[str, Any], predicted_addition: bool) -> dict[str, Any] | None:
    if not predicted_addition:
        return None
    return {
        "phone": row["best_insert_phone"],
        "boundary": row["best_insert_boundary"],
    }


def run_fold(
    rows: Sequence[Mapping[str, Any]], heldout_speaker: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibration = [row for row in rows if row["speaker"] != heldout_speaker]
    heldout = [row for row in rows if row["speaker"] == heldout_speaker]
    if not calibration or not heldout:
        raise ValueError("fold must contain calibration and held-out rows")
    calibration_x = _matrix(calibration)
    heldout_x = _matrix(heldout)
    calibration_y = np.asarray([row["label"] for row in calibration], dtype=np.int64)
    calibration_correct = np.asarray(
        [row["correct_only_negative"] for row in calibration], dtype=bool
    )
    scaler, model, calibration_scaled = fit_calibration_model(calibration_x, calibration_y)
    calibration_probabilities = addition_probability(model, calibration_scaled)
    theta, evaluations = select_threshold(
        calibration_probabilities, calibration_y, calibration_correct
    )
    heldout_scaled = scaler.transform(heldout_x)
    heldout_probabilities = addition_probability(model, heldout_scaled)
    heldout_predictions = apply_threshold(heldout_probabilities, theta)
    outputs: list[dict[str, Any]] = []
    for row, probability, prediction in zip(heldout, heldout_probabilities, heldout_predictions):
        event = event_from_decision(row, bool(prediction))
        outputs.append(
            {
                "row_id": row["row_id"],
                "speaker": row["speaker"],
                "probability": float(probability),
                "threshold": float(theta),
                "predicted_addition": bool(prediction),
                "predicted_event": event,
            }
        )
    audit = {
        "heldout_speaker": heldout_speaker,
        "calibration_row_ids": [row["row_id"] for row in calibration],
        "heldout_row_ids": [row["row_id"] for row in heldout],
        "scaler_fit_row_ids": [row["row_id"] for row in calibration],
        "model_fit_row_ids": [row["row_id"] for row in calibration],
        "threshold_selection_row_ids": [row["row_id"] for row in calibration],
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "scaler_var": scaler.var_.astype(float).tolist(),
        "calibration_feature_matrix": calibration_x.astype(float).tolist(),
        "heldout_feature_matrix": heldout_x.astype(float).tolist(),
        "heldout_transformed": heldout_scaled.astype(float).tolist(),
        "same_scaler_used_for_heldout_transform": True,
        "threshold": float(theta),
        "threshold_candidate_count": len(evaluations),
        "model_classes": model.classes_.astype(int).tolist(),
        "model_coefficients": model.coef_.astype(float).tolist(),
        "model_intercept": model.intercept_.astype(float).tolist(),
    }
    return outputs, audit


def run_loso(
    rows: Sequence[Mapping[str, Any]], speaker_order: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(speaker_order) != len(set(speaker_order)):
        raise ValueError("speaker order contains duplicates")
    observed = {str(row["speaker"]) for row in rows}
    if observed != set(speaker_order):
        raise ValueError("speaker order must exactly cover observed speakers")
    outputs: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for speaker in speaker_order:
        fold_outputs, fold_audit = run_fold(rows, speaker)
        outputs.extend(fold_outputs)
        audits.append(fold_audit)
    row_ids = [str(row["row_id"]) for row in outputs]
    expected = [str(row["row_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != set(expected):
        raise RuntimeError("OOF outputs must contain every row exactly once")
    return outputs, audits
