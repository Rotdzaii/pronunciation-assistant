from __future__ import annotations

import os
from typing import Any

from app.contracts.scoring_contract import build_failed_scoring_result
from app.scoring.heuristic_gop_scorer import score_heuristic_gop


def score_pronunciation_segments(
    alignment_result: dict[str, Any],
    segment_predictions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    mode = os.getenv("SCORING_MODE", "heuristic_gop").strip().lower()
    if mode == "heuristic_gop":
        return score_heuristic_gop(alignment_result, segment_predictions)
    if mode == "none":
        return build_failed_scoring_result(
            scoring_method="none",
            error="Pronunciation segmental scoring disabled by SCORING_MODE=none.",
        )
    return build_failed_scoring_result(
        scoring_method=mode,
        error=f"Unsupported SCORING_MODE={mode!r}. Use heuristic_gop or none.",
    )
