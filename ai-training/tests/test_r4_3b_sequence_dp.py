from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import r4_3b_sequence_dp as dp  # noqa: E402


def evidence(phones: list[str]) -> np.ndarray:
    logits = np.full((len(phones), len(dp.PHONE_VOCAB)), -20.0, dtype=np.float64)
    for index, phone in enumerate(phones):
        logits[index, dp.PHONE_TO_ID[phone]] = 20.0
    maximum = logits.max(axis=1, keepdims=True)
    shifted = logits - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def operations(result: dict[str, object]) -> list[str]:
    assert result["status"] == "OK"
    return [step.operation for step in result["best"].steps]


def test_probe_grid_matches_r4_3a() -> None:
    assert dp.probe_centers(1.0, 1.0) == [1.0]
    assert dp.probe_centers(1.0, 1.02) == [1.0, 1.02]
    assert np.allclose(dp.probe_centers(1.0, 1.08), [1.0, 1.04, 1.08])
    assert np.allclose(dp.probe_centers(1.0, 1.09), [1.0, 1.04, 1.08, 1.09])


def test_medial_deletion_is_representable() -> None:
    result = dp.align(["K", "IH", "T", "CH", "AH", "N"], evidence(["K", "IH", "CH", "AH", "N"]))
    assert operations(result) == ["MATCH", "MATCH", "DELETE_EXPECTED", "MATCH", "MATCH", "MATCH"]


def test_substitution_is_representable() -> None:
    result = dp.align(["K", "IH", "T"], evidence(["K", "IH", "D"]))
    assert operations(result) == ["MATCH", "MATCH", "SUBSTITUTION"]
    assert result["best"].steps[-1].observed_phone_index == dp.PHONE_TO_ID["D"]


def test_all_correct() -> None:
    result = dp.align(["K", "IH", "T"], evidence(["K", "IH", "T"]))
    assert operations(result) == ["MATCH", "MATCH", "MATCH"]


def test_initial_deletion() -> None:
    result = dp.align(["HH", "K", "IH"], evidence(["K", "IH"]))
    assert operations(result) == ["DELETE_EXPECTED", "MATCH", "MATCH"]


def test_final_deletion() -> None:
    result = dp.align(["K", "IH", "T"], evidence(["K", "IH"]))
    assert operations(result) == ["MATCH", "MATCH", "DELETE_EXPECTED"]


def test_two_consecutive_deletions() -> None:
    result = dp.align(["K", "T", "R", "IH"], evidence(["K", "IH"]))
    assert operations(result) == ["MATCH", "DELETE_EXPECTED", "DELETE_EXPECTED", "MATCH"]
