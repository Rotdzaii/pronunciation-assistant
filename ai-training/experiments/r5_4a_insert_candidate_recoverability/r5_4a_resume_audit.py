from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MAIN_ROOT = Path(r"C:\Users\Admin\Documents\KLTN\pronunciation-assistant")
RESEARCH_ROOT = Path(r"C:\Users\Admin\Documents\KLTN\pronunciation-assistant-research")
R5_4A_DIR = RESEARCH_ROOT / "ai-training/experiments/r5_4a_insert_candidate_recoverability"
R5_4B_DIR = RESEARCH_ROOT / "ai-training/experiments/r5_4b_insert_candidate_materialization"
R5_4B_RESULT = R5_4B_DIR / "materialization_result"
R5_3A_DIR = RESEARCH_ROOT / "ai-training/experiments/r5_3a_evidence_separated_relation_scoring"
R5_2B_DIR = RESEARCH_ROOT / "ai-training/experiments/r5_2b_relation_competition"
R5_1A_DIR = RESEARCH_ROOT / "ai-training/experiments/r5_1a_alignability_safe_addition_scoring"

TRAIN = ("BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA")
TRUTH_SOURCE = R5_2B_DIR / "r5_2b_tc1_pa1_env_train_scores.jsonl"
V4_PATH = RESEARCH_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"

DIRECT_IDENTITIES = {
    "r5_4a_contract": (
        R5_4A_DIR / "R5_4A_AUDIT_CONTRACT.md",
        "111DDB77EC09B177505AEEF7B260476D8C718F745AD6170E6969926841338A27",
    ),
    "r5_4a_manifest": (
        R5_4A_DIR / "R5_4A_MANIFEST.json",
        "012322265AACD90009001AFCBF675C9228ADC55417B0A31029CD2A3EF32393B8",
    ),
    "r5_4b_contract_manifest": (
        R5_4B_DIR / "R5_4B_CONTRACT_MANIFEST.json",
        "5087DB63E37C52A407720A9A7A11D542A4DD651510469CE5960CE8B3B2421297",
    ),
    "r5_4b_execution_manifest": (
        R5_4B_RESULT / "R5_4B_EXECUTION_MANIFEST.json",
        "FD1E4E66168654EC54778A489E0B443BD9C691DD1392233C4284CDF6CDF07B11",
    ),
    "r5_3a_closure_manifest": (
        R5_3A_DIR / "R5_3A_CLOSURE_MANIFEST.json",
        "9F57A293E0F4CA6E35E06761CFF54ABF70D967FE6ECF4F42ABFFF7C35397768C",
    ),
    "r5_2b_execution_manifest": (
        R5_2B_DIR / "R5_2B_TC1_PA1_ENV_EXECUTION_MANIFEST.json",
        "37F3C86FF11526B8AB54D173937A0F488125A1D0B610032CC8A5D47B11602387",
    ),
    "r5_1a_execution_manifest": (
        R5_1A_DIR / "R5_1A_EXECUTION_MANIFEST.json",
        "C9343A75CE26C2BEECA388EBA855E91AE0992D22C0C5D785308D0A98B89A3CD6",
    ),
    "v4": (V4_PATH, "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D"),
    "truth_source": (TRUTH_SOURCE, "EAC81ABDDF56C77E4A71AD96AE5985D7C00DD60559A27C3A3BAAB16859EEFF3C"),
}

MANIFEST_NAMES = (
    "r5_4a_manifest",
    "r5_4b_contract_manifest",
    "r5_4b_execution_manifest",
    "r5_3a_closure_manifest",
    "r5_2b_execution_manifest",
    "r5_1a_execution_manifest",
)

OUTPUT_NAMES = (
    "r5_4a_resume_audit.py",
    "r5_4a_resume_source_identity.json",
    "r5_4a_truth_mapping_audit.json",
    "r5_4a_single_addition_recoverability.json",
    "r5_4a_rank_distribution.json",
    "r5_4a_incremental_topk.json",
    "r5_4a_score_gap_diagnostics.json",
    "r5_4a_speaker_diagnostics.json",
    "r5_4a_phone_diagnostics.json",
    "r5_4a_position_diagnostics.json",
    "r5_4a_multiple_addition_diagnostics.json",
    "r5_4a_interpretation.json",
    "r5_4a_resume_protocol_audit.json",
    "R5_4A_RESUMED_RECOVERABILITY_REPORT.md",
    "R5_4A_RESUME_MANIFEST.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def resolve_manifest_entry(manifest_path: Path, entry: dict[str, Any]) -> Path:
    repository = entry.get("repository")
    relative = Path(entry["relative_path"])
    if repository in {"main", "main_repository"}:
        return MAIN_ROOT / relative
    if repository in {"research", "research_worktree"}:
        return RESEARCH_ROOT / relative
    return manifest_path.parent / relative


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["artifacts"]:
        artifact = resolve_manifest_entry(manifest_path, entry)
        if not artifact.is_file():
            failures.append({"relative_path": entry["relative_path"], "reason": "missing"})
            continue
        actual_size = artifact.stat().st_size
        actual_sha = sha256(artifact)
        if actual_size != int(entry["byte_size"]) or actual_sha != entry["sha256"]:
            failures.append(
                {
                    "relative_path": entry["relative_path"],
                    "reason": "identity_mismatch",
                    "expected_byte_size": int(entry["byte_size"]),
                    "actual_byte_size": actual_size,
                    "expected_sha256": entry["sha256"],
                    "actual_sha256": actual_sha,
                }
            )
    return {
        "path": str(manifest_path),
        "artifact_count": len(manifest["artifacts"]),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }


def verify_identities() -> dict[str, Any]:
    actual = {}
    failures = {}
    for name, (path, expected_sha) in DIRECT_IDENTITIES.items():
        if not path.is_file():
            failures[name] = {"reason": "missing", "path": str(path)}
            continue
        actual_sha = sha256(path)
        actual[name] = {"path": str(path), "byte_size": path.stat().st_size, "sha256": actual_sha}
        if actual_sha != expected_sha:
            failures[name] = {"expected_sha256": expected_sha, "actual_sha256": actual_sha}
    manifest_audits = {
        name: verify_manifest(DIRECT_IDENTITIES[name][0])
        for name in MANIFEST_NAMES
        if DIRECT_IDENTITIES[name][0].is_file()
    }
    for name, audit in manifest_audits.items():
        if audit["status"] != "PASS":
            failures[f"{name}_entries"] = audit["failures"]
    if failures:
        raise RuntimeError(f"R5_4A_RESUME_BLOCKED_IDENTITY: {failures}")
    r5_4b = json.loads(DIRECT_IDENTITIES["r5_4b_execution_manifest"][0].read_text(encoding="utf-8"))
    if r5_4b["status"] != "R5_4B_INSERT_CANDIDATE_MATERIALIZATION_PASS" or r5_4b["hash_audit"] != "HASH_AUDIT_PASS":
        raise RuntimeError("R5_4A_RESUME_BLOCKED_IDENTITY: R5-4B status/hash audit")
    return {"status": "PASS", "identities": actual, "manifest_entry_audits": manifest_audits}


def load_frozen_truth() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    positive: dict[str, dict[str, Any]] = {}
    total_rows = 0
    with TRUTH_SOURCE.open("r", encoding="utf-8") as handle:
        for line in handle:
            total_rows += 1
            row = json.loads(line)
            if not row["addition_label"]:
                continue
            identity = row["source_identity"]
            if identity in positive:
                raise RuntimeError("R5_4A_RESUME_BLOCKED_TRUTH_PROVENANCE: duplicate positive identity")
            positive[identity] = {
                "source_identity": identity,
                "speaker": row["speaker"],
                "expected_sequence_length": len(row["expected_sequence_ids"]),
                "events": [
                    {
                        "phone": event["phone"],
                        "phone_index": int(event["phone_index"]),
                        "boundary": int(event["boundary"]),
                        "position": event["position"],
                    }
                    for event in row["ground_truth_events"]
                ],
            }
    event_counts = Counter(len(row["events"]) for row in positive.values())
    return positive, {
        "source": str(TRUTH_SOURCE),
        "source_sha256": sha256(TRUTH_SOURCE),
        "runtime_rows": total_rows,
        "addition_positive_words": len(positive),
        "truth_events": sum(len(row["events"]) for row in positive.values()),
        "event_count_distribution": {str(key): value for key, value in sorted(event_counts.items())},
        "manual_relabeling": False,
        "fuzzy_phone_matching": False,
        "audio_or_textgrid_read": False,
    }


def candidate_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    score_hex = row["authoritative_insert_target_score_float64_hex"]
    if score_hex is None:
        return (1, 0.0, int(row["boundary_index"]), int(row["inserted_phone_index"]))
    score = float.fromhex(score_hex)
    return (0, -score, int(row["boundary_index"]), int(row["inserted_phone_index"]))


def read_positive_candidate_families(
    truth: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    shard_records = {}
    total_rows = 0
    for speaker in TRAIN:
        path = R5_4B_RESULT / f"r5_4b_insert_candidates_{speaker}.jsonl"
        rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                rows += 1
                row = json.loads(line)
                identity = row["source_identity"]
                if identity in truth:
                    families[identity].append(row)
        total_rows += rows
        shard_records[speaker] = {
            "path": str(path),
            "candidate_rows": rows,
            "byte_size": path.stat().st_size,
            "sha256": sha256(path),
        }
    missing_families = sorted(set(truth) - set(families))
    if total_rows != 2_977_040 or missing_families:
        raise RuntimeError(
            f"R5_4A_RESUME_BLOCKED_IDENTITY: candidates={total_rows}, missing positive families={missing_families[:10]}"
        )
    for identity, rows in families.items():
        expected = 40 * (truth[identity]["expected_sequence_length"] + 1)
        if len(rows) != expected:
            raise RuntimeError(f"R5_4A_RESUME_BLOCKED_IDENTITY: {identity} candidate count {len(rows)} != {expected}")
        indexes = [int(row["candidate_index"]) for row in rows]
        if indexes != list(range(expected)):
            raise RuntimeError(f"R5_4A_RESUME_BLOCKED_IDENTITY: {identity} candidate ordering")
        rows.sort(key=candidate_sort_key)
        for rank, row in enumerate(rows, start=1):
            row["_rank"] = rank
    return dict(families), {
        "status": "PASS",
        "shards": len(shard_records),
        "candidate_rows": total_rows,
        "positive_candidate_families_loaded": len(families),
        "missing_positive_families": len(missing_families),
        "shard_records": shard_records,
        "new_candidate_scores_created": False,
    }


def finite_score(row: dict[str, Any]) -> float | None:
    value = row["authoritative_insert_target_score_float64_hex"]
    return None if value is None else float.fromhex(value)


def map_events(
    truth: dict[str, dict[str, Any]],
    families: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    singles = []
    multiples = []
    unmappable = []
    for identity, truth_row in truth.items():
        lookup: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for candidate in families[identity]:
            lookup[(int(candidate["inserted_phone_index"]), int(candidate["boundary_index"]))].append(candidate)
        mapped_events = []
        for event in truth_row["events"]:
            matches = lookup[(event["phone_index"], event["boundary"])]
            if len(matches) != 1:
                unmappable.append(
                    {
                        "source_identity": identity,
                        "event": event,
                        "candidate_match_count": len(matches),
                    }
                )
                mapped_events.append({**event, "mappable": False})
                continue
            candidate = matches[0]
            truth_score = finite_score(candidate)
            best = families[identity][0]
            best_score = finite_score(best)
            gap = None if truth_score is None or best_score is None else best_score - truth_score
            mapped_events.append(
                {
                    **event,
                    "mappable": True,
                    "candidate_index": int(candidate["candidate_index"]),
                    "alignable": bool(candidate["alignable"]),
                    "truth_rank": int(candidate["_rank"]),
                    "truth_score": truth_score,
                    "truth_score_float64_hex": None if truth_score is None else truth_score.hex(),
                    "best_insert_score": best_score,
                    "best_insert_score_float64_hex": None if best_score is None else best_score.hex(),
                    "score_gap": gap,
                    "score_gap_float64_hex": None if gap is None else gap.hex(),
                }
            )
        result = {
            "source_identity": identity,
            "speaker": truth_row["speaker"],
            "expected_sequence_length": truth_row["expected_sequence_length"],
            "truth_event_count": len(truth_row["events"]),
            "mapped_events": mapped_events,
        }
        if len(truth_row["events"]) == 1:
            singles.append(result)
        else:
            multiples.append(result)
    single_mapped = sum(row["mapped_events"][0]["mappable"] for row in singles)
    multiple_events = sum(len(row["mapped_events"]) for row in multiples)
    multiple_mapped = sum(event["mappable"] for row in multiples for event in row["mapped_events"])
    audit = {
        "status": "PASS" if not unmappable else "PARTIAL_MAPPING",
        "addition_positive_words": len(truth),
        "total_truth_events": sum(len(row["events"]) for row in truth.values()),
        "single_addition_words": len(singles),
        "single_addition_exactly_mappable": single_mapped,
        "single_addition_unmappable": len(singles) - single_mapped,
        "single_addition_mapping_coverage": single_mapped / len(singles) if singles else 0.0,
        "multiple_addition_words": len(multiples),
        "multiple_addition_events": multiple_events,
        "multiple_addition_mapped_events": multiple_mapped,
        "multiple_addition_event_mapping_coverage": multiple_mapped / multiple_events if multiple_events else 0.0,
        "unmappable_event_details": unmappable,
        "mapping_identity": "exact (canonical phone index, expected-sequence boundary)",
        "manual_fixes": False,
    }
    return singles, multiples, audit


def quantiles(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {key: None for key in ("count", "mean", "median", "q25", "q75", "q90", "maximum")}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
        "quantile_method": "numpy default linear",
    }


def topk_records(ranks: list[int]) -> dict[str, Any]:
    total = len(ranks)
    return {
        f"top_{k}": {
            "recovered": sum(rank <= k for rank in ranks),
            "denominator": total,
            "recall": sum(rank <= k for rank in ranks) / total if total else 0.0,
        }
        for k in (1, 3, 5, 10)
    }


def group_diagnostic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [int(row["truth_rank"]) for row in rows]
    result = {
        "N": len(rows),
        **topk_records(ranks),
        "median_rank": float(np.median(ranks)) if ranks else None,
        "mean_rank": float(np.mean(ranks)) if ranks else None,
    }
    return result


def analyze_single(singles: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    mapped_rows = []
    for row in singles:
        event = row["mapped_events"][0]
        if not event["mappable"]:
            continue
        mapped_rows.append(
            {
                "source_identity": row["source_identity"],
                "speaker": row["speaker"],
                "phone": event["phone"],
                "phone_index": event["phone_index"],
                "boundary": event["boundary"],
                "position": event["position"],
                "truth_rank": event["truth_rank"],
                "truth_candidate_score": event["truth_score"],
                "truth_candidate_score_float64_hex": event["truth_score_float64_hex"],
                "best_insert_score": event["best_insert_score"],
                "best_insert_score_float64_hex": event["best_insert_score_float64_hex"],
                "score_gap": event["score_gap"],
                "score_gap_float64_hex": event["score_gap_float64_hex"],
            }
        )
    ranks = [row["truth_rank"] for row in mapped_rows]
    if any(row["score_gap"] is None for row in mapped_rows):
        raise RuntimeError("R5_4A_RESUME_BLOCKED_TRUTH_PROVENANCE: nonfinite truth candidate")
    gaps = [row["score_gap"] for row in mapped_rows]
    rank_one_gap_violations = sum(row["truth_rank"] == 1 and row["score_gap"] != 0.0 for row in mapped_rows)
    topk = topk_records(ranks)
    recoverability = {
        "population": "exactly mappable single-addition TRAIN words",
        "rows": len(mapped_rows),
        **topk,
        "rank_distribution": quantiles(ranks),
        "mean_reciprocal_rank": float(np.mean([1.0 / rank for rank in ranks])) if ranks else 0.0,
        "per_word": mapped_rows,
    }
    rank_distribution = {
        "population": recoverability["population"],
        **recoverability["rank_distribution"],
        "rank_frequency": {str(rank): count for rank, count in sorted(Counter(ranks).items())},
        "cumulative_at_frozen_k": {
            str(k): sum(rank <= k for rank in ranks) for k in (1, 3, 5, 10)
        },
    }
    gap_diagnostics = {
        "definition": "BEST_INSERT_SCORE - TRUTH_INSERT_SCORE",
        "distribution": quantiles(gaps),
        "rank_1_words": sum(rank == 1 for rank in ranks),
        "rank_1_zero_gap": sum(row["truth_rank"] == 1 and row["score_gap"] == 0.0 for row in mapped_rows),
        "rank_1_nonzero_gap_violations": rank_one_gap_violations,
        "finite_gap_count": len(gaps),
    }
    return recoverability, rank_distribution, gap_diagnostics


def analyze_groups(
    per_word: list[dict[str, Any]], single_totals_by_speaker: Counter[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    speakers = {}
    for speaker in TRAIN:
        rows = [row for row in per_word if row["speaker"] == speaker]
        speakers[speaker] = group_diagnostic(rows)
        speakers[speaker]["single_addition_words"] = int(single_totals_by_speaker[speaker])
        speakers[speaker]["mappable_words"] = len(rows)
        speakers[speaker]["mapping_coverage"] = (
            len(rows) / single_totals_by_speaker[speaker] if single_totals_by_speaker[speaker] else 0.0
        )
    positions = {}
    for position in ("BEFORE_FIRST", "BETWEEN", "AFTER_FINAL"):
        positions[position] = group_diagnostic([row for row in per_word if row["position"] == position])
    phones = {}
    for phone in sorted({row["phone"] for row in per_word}):
        rows = [row for row in per_word if row["phone"] == phone]
        phones[phone] = group_diagnostic(rows)
        phones[phone]["interpretation_support"] = "sufficient_for_description" if len(rows) >= 5 else "rare_preserve_count_only"
    return speakers, positions, phones


def analyze_multiples(multiples: list[dict[str, Any]]) -> dict[str, Any]:
    events = []
    per_word = []
    for row in multiples:
        mapped = [event for event in row["mapped_events"] if event["mappable"]]
        ranks = [event["truth_rank"] for event in mapped]
        per_word.append(
            {
                "source_identity": row["source_identity"],
                "speaker": row["speaker"],
                "truth_event_count": len(row["mapped_events"]),
                "mapped_event_count": len(mapped),
                "best_rank_among_truth_events": min(ranks) if ranks else None,
            }
        )
        for event_index, event in enumerate(row["mapped_events"]):
            events.append(
                {
                    "source_identity": row["source_identity"],
                    "speaker": row["speaker"],
                    "event_index": event_index,
                    "phone": event["phone"],
                    "boundary": event["boundary"],
                    "position": event["position"],
                    "mappable": event["mappable"],
                    "truth_rank": event.get("truth_rank"),
                }
            )
    mapped_ranks = [event["truth_rank"] for event in events if event["mappable"]]
    best_ranks = [row["best_rank_among_truth_events"] for row in per_word if row["best_rank_among_truth_events"] is not None]
    return {
        "word_count": len(multiples),
        "truth_event_count": len(events),
        "mapped_event_count": len(mapped_ranks),
        "event_mapping_coverage": len(mapped_ranks) / len(events) if events else 0.0,
        "per_event_recoverability": topk_records(mapped_ranks),
        "best_rank_among_any_truth_event_per_word": quantiles(best_ranks),
        "per_word": per_word,
        "per_event": events,
        "limitation": "The frozen candidate family represents one INSERT hypothesis at a time and cannot encode multiple simultaneous additions in one candidate.",
    }


def interpretation_answers(
    recoverability: dict[str, Any],
    speakers: dict[str, Any],
    positions: dict[str, Any],
    phones: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    t1 = recoverability["top_1"]["recall"]
    t5 = recoverability["top_5"]["recall"]
    t10 = recoverability["top_10"]["recall"]
    median = recoverability["rank_distribution"]["median"]
    speaker_top10 = {name: value["top_10"]["recall"] for name, value in speakers.items() if value["N"]}
    weakest_speaker = min(speaker_top10, key=speaker_top10.get)
    strongest_speaker = max(speaker_top10, key=speaker_top10.get)
    position_top10 = {name: value["top_10"]["recall"] for name, value in positions.items() if value["N"]}
    weakest_position = min(position_top10, key=position_top10.get)
    supported_phones = {
        name: value for name, value in phones.items() if value["interpretation_support"] == "sufficient_for_description"
    }
    weakest_phones = sorted(
        supported_phones,
        key=lambda name: (supported_phones[name]["top_10"]["recall"], -supported_phones[name]["N"], name),
    )[:5]
    if t10 >= 0.55 and median <= 10:
        q1 = "YES: the typical exact truth candidate is within the frozen top-ten region."
    elif t10 < 0.40:
        q1 = "NO: most exact truth candidates are ranked below the frozen top-ten region."
    else:
        q1 = "MIXED: a substantial subset is near the top, but this is not the usual outcome strongly enough to pass both rank gates."
    q2 = (
        f"YES: Top-5 gains {t5 - t1:.6f} and Top-10 gains {t10 - t1:.6f} absolute recall over Top-1."
        if t10 - t1 >= 0.10
        else f"NO: Top-10 gains only {t10 - t1:.6f} absolute recall over Top-1."
    )
    if t10 >= 0.55 and t10 - t1 >= 0.10:
        q3 = "YES: many truth events are available just below rank 1, so top-1 selection is a plausible major contributor to low exact-event F1."
        event_pattern = "A: correct events are frequently near the top but not first."
    elif t10 < 0.40:
        q3 = "NO: the correct event is usually too far down the INSERT family for top-1 selection alone to explain the low event F1."
        event_pattern = "B: correct events are generally ranked far down the candidate family."
    else:
        q3 = "PARTLY: both top-1 selection loss and low within-family ranking contribute."
        event_pattern = "C: mixed pattern."
    speaker_range = max(speaker_top10.values()) - min(speaker_top10.values())
    q4 = (
        f"Speaker recoverability varies by {speaker_range:.6f} Top-10 recall; weakest is {weakest_speaker} and strongest is {strongest_speaker}. "
        + ("The variation is substantial." if speaker_range >= 0.30 else "The failure/success pattern is not dominated by only a few speakers.")
    )
    q5 = (
        f"Weakest position by Top-10 is {weakest_position}. The weakest phones with N>=5 are "
        + ", ".join(weakest_phones)
        + "; rare-phone counts are preserved without interpretation."
    )
    q6 = (
        "YES: all four frozen feasibility gates pass, supporting a separately preregistered candidate-level generation."
        if gates["all_pass"]
        else "NO: the frozen recoverability gates do not all pass, so this audit does not justify that next generation."
    )
    return {
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
        "Q4": q4,
        "Q5": q5,
        "Q6": q6,
        "existing_event_f1_context": {
            "r5_1a_exact_event_f1": 0.04389312977099236,
            "r5_3a_exact_event_f1": 0.044044044044044044,
            "classification": event_pattern,
            "context_only_no_binary_scoring_rerun": True,
        },
    }


def build_report(
    mapping: dict[str, Any],
    recovery: dict[str, Any],
    gains: dict[str, Any],
    gaps: dict[str, Any],
    gates: dict[str, Any],
    interpretation: dict[str, Any],
    status: str,
) -> str:
    rank = recovery["rank_distribution"]
    lines = [
        "# R5-4A Resumed INSERT Candidate Recoverability Audit",
        "",
        f"Final status: `{status}`",
        "",
        "## Truth mapping",
        "",
        f"- Addition-positive words/events: {mapping['addition_positive_words']} / {mapping['total_truth_events']}",
        f"- Single-addition exactly mappable: {mapping['single_addition_exactly_mappable']} / {mapping['single_addition_words']}",
        f"- Multiple-addition words/events: {mapping['multiple_addition_words']} / {mapping['multiple_addition_events']}",
        "",
        "## Single-addition recoverability",
        "",
        f"- Top-1 / Top-3 / Top-5 / Top-10: {recovery['top_1']['recall']:.9f} / {recovery['top_3']['recall']:.9f} / {recovery['top_5']['recall']:.9f} / {recovery['top_10']['recall']:.9f}",
        f"- Mean / median truth rank: {rank['mean']:.6f} / {rank['median']:.6f}",
        f"- Q25 / Q75 / Q90 / max: {rank['q25']:.6f} / {rank['q75']:.6f} / {rank['q90']:.6f} / {rank['maximum']:.6f}",
        f"- MRR: {recovery['mean_reciprocal_rank']:.9f}",
        "",
        "## Incremental gains and gaps",
        "",
        f"- Top-1 -> Top-3: {gains['top_1_to_top_3']:.9f}",
        f"- Top-3 -> Top-5: {gains['top_3_to_top_5']:.9f}",
        f"- Top-5 -> Top-10: {gains['top_5_to_top_10']:.9f}",
        f"- Score-gap mean / median: {gaps['distribution']['mean']:.9f} / {gaps['distribution']['median']:.9f}",
        "",
        "## Frozen feasibility gates",
        "",
    ]
    for name in ("G1", "G2", "G3", "G4"):
        gate = gates["gates"][name]
        lines.append(f"- {name}: {gate['result']} ({gate['value']:.12g} {gate['operator']} {gate['threshold']})")
    lines.extend(["", f"Passed: {gates['passed_count']} / 4", "", "## Interpretation", ""])
    for question in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"):
        lines.append(f"- {question}: {interpretation[question]}")
    lines.extend(
        [
            "",
            "No checkpoint, audio, new candidate score, classifier, threshold search, word-level performance metric, VALIDATION, or TEST data was used.",
            "",
        ]
    )
    return "\n".join(lines)


def execute() -> None:
    for name in OUTPUT_NAMES:
        if (R5_4A_DIR / name).exists():
            raise RuntimeError(f"Refusing overwrite of resumed-audit artifact: {name}")

    identities = verify_identities()
    shutil.copyfile(Path(__file__).resolve(), R5_4A_DIR / "r5_4a_resume_audit.py")
    write_json(R5_4A_DIR / "r5_4a_resume_source_identity.json", identities)

    truth, truth_provenance = load_frozen_truth()
    families, candidate_provenance = read_positive_candidate_families(truth)
    singles, multiples, mapping = map_events(truth, families)
    mapping["truth_provenance"] = truth_provenance
    mapping["candidate_provenance"] = candidate_provenance
    write_json(R5_4A_DIR / "r5_4a_truth_mapping_audit.json", mapping)

    recovery, ranks, gaps = analyze_single(singles)
    write_json(R5_4A_DIR / "r5_4a_single_addition_recoverability.json", recovery)
    write_json(R5_4A_DIR / "r5_4a_rank_distribution.json", ranks)
    write_json(R5_4A_DIR / "r5_4a_score_gap_diagnostics.json", gaps)

    gains = {
        "top_1_to_top_3": recovery["top_3"]["recall"] - recovery["top_1"]["recall"],
        "top_3_to_top_5": recovery["top_5"]["recall"] - recovery["top_3"]["recall"],
        "top_5_to_top_10": recovery["top_10"]["recall"] - recovery["top_5"]["recall"],
        "frozen_k": [1, 3, 5, 10],
        "alternative_k_evaluated": False,
    }
    write_json(R5_4A_DIR / "r5_4a_incremental_topk.json", gains)

    speakers, positions, phones = analyze_groups(
        recovery["per_word"], Counter(row["speaker"] for row in singles)
    )
    write_json(R5_4A_DIR / "r5_4a_speaker_diagnostics.json", {"speakers": speakers, "speaker_exclusions": 0})
    write_json(R5_4A_DIR / "r5_4a_position_diagnostics.json", {"positions": positions, "position_specific_rule": False})
    write_json(
        R5_4A_DIR / "r5_4a_phone_diagnostics.json",
        {
            "phones": phones,
            "descriptive_support_marker": "N>=5",
            "rare_phone_counts_preserved": True,
            "phone_specific_rule": False,
        },
    )
    multiple = analyze_multiples(multiples)
    write_json(R5_4A_DIR / "r5_4a_multiple_addition_diagnostics.json", multiple)

    gate_values = {
        "G1": (mapping["single_addition_mapping_coverage"], ">=", 0.99),
        "G2": (recovery["top_5"]["recall"], ">=", 0.40),
        "G3": (recovery["top_10"]["recall"], ">=", 0.55),
        "G4": (recovery["rank_distribution"]["median"], "<=", 10.0),
    }
    gate_records = {}
    for name, (value, operator, threshold) in gate_values.items():
        passed = value >= threshold if operator == ">=" else value <= threshold
        gate_records[name] = {
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "result": "PASS" if passed else "FAIL",
        }
    passed_count = sum(record["result"] == "PASS" for record in gate_records.values())
    gates = {
        "gates": gate_records,
        "passed_count": passed_count,
        "total": 4,
        "all_pass": passed_count == 4,
        "thresholds_frozen_before_R5_4B_candidate_values": True,
    }
    status = (
        "R5_4A_INSERT_CANDIDATE_RECOVERABILITY_FEASIBLE"
        if gates["all_pass"]
        else "R5_4A_INSERT_CANDIDATE_RECOVERABILITY_NOT_CONFIRMED"
    )
    interpretation = interpretation_answers(recovery, speakers, positions, phones, gates)
    interpretation["status"] = status
    interpretation["gates"] = gates
    interpretation["no_new_scientific_method"] = True
    write_json(R5_4A_DIR / "r5_4a_interpretation.json", interpretation)

    protocol = {
        "neural_training": False,
        "checkpoint_loaded": False,
        "checkpoint_inference": False,
        "new_candidate_scores": False,
        "existing_materialized_candidate_scores_read": True,
        "classifier_fitting": False,
        "threshold_search": False,
        "new_word_level_performance_metrics": False,
        "recoverability_metrics": True,
        "train_audio_accessed": False,
        "train_textgrid_accessed": False,
        "validation_paths_resolved": False,
        "validation_accessed": False,
        "test_paths_resolved": False,
        "test_accessed": False,
        "r5_1a_modified": False,
        "r5_2b_modified": False,
        "r5_3a_modified": False,
        "r5_4b_modified": False,
        "r5_4a_contract_modified": False,
    }
    write_json(R5_4A_DIR / "r5_4a_resume_protocol_audit.json", protocol)

    report = build_report(mapping, recovery, gains, gaps, gates, interpretation, status)
    (R5_4A_DIR / "R5_4A_RESUMED_RECOVERABILITY_REPORT.md").write_text(
        report, encoding="utf-8", newline="\n"
    )

    artifact_names = [name for name in OUTPUT_NAMES if name != "R5_4A_RESUME_MANIFEST.json"]
    entries = [
        {
            "relative_path": name,
            "byte_size": (R5_4A_DIR / name).stat().st_size,
            "sha256": sha256(R5_4A_DIR / name),
        }
        for name in artifact_names
    ]
    failures = []
    for entry in entries:
        path = R5_4A_DIR / entry["relative_path"]
        if path.stat().st_size != entry["byte_size"] or sha256(path) != entry["sha256"]:
            failures.append(entry["relative_path"])
    manifest = {
        "stage": "R5-4A resumed frozen INSERT candidate recoverability audit",
        "status": status,
        "manifest_self_excluded": True,
        "artifact_count": len(entries),
        "artifacts": entries,
        "hash_audit": "HASH_AUDIT_PASS" if not failures else "HASH_AUDIT_FAIL",
        "failures": failures,
        "preserved_r5_4a_contract_sha256": DIRECT_IDENTITIES["r5_4a_contract"][1],
        "r5_4b_execution_manifest_sha256": DIRECT_IDENTITIES["r5_4b_execution_manifest"][1],
        "new_checkpoint_inference": False,
        "validation_accessed": False,
        "test_accessed": False,
    }
    write_json(R5_4A_DIR / "R5_4A_RESUME_MANIFEST.json", manifest)
    if failures:
        raise RuntimeError(f"R5_4A_RESUME hash audit failure: {failures}")
    print(
        json.dumps(
            {
                "status": status,
                "gates_passed": passed_count,
                "top_1": recovery["top_1"]["recall"],
                "top_5": recovery["top_5"]["recall"],
                "top_10": recovery["top_10"]["recall"],
                "median_rank": recovery["rank_distribution"]["median"],
                "artifact_count": len(entries),
                "manifest_sha256": sha256(R5_4A_DIR / "R5_4A_RESUME_MANIFEST.json"),
                "hash_audit": manifest["hash_audit"],
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        raise SystemExit("Frozen resumed R5-4A audit requires --execute")
    execute()


if __name__ == "__main__":
    main()
