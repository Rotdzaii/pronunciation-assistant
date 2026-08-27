from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


MAIN_ROOT = Path(r"C:\Users\Admin\Documents\KLTN\pronunciation-assistant")
RESEARCH_ROOT = Path(r"C:\Users\Admin\Documents\KLTN\pronunciation-assistant-research")
R5_4B_DIR = RESEARCH_ROOT / "ai-training/experiments/r5_4b_insert_candidate_materialization"
R5_4A_DIR = RESEARCH_ROOT / "ai-training/experiments/r5_4a_insert_candidate_recoverability"
R5_3A_DIR = RESEARCH_ROOT / "ai-training/experiments/r5_3a_evidence_separated_relation_scoring"
R5_2B_DIR = RESEARCH_ROOT / "ai-training/experiments/r5_2b_relation_competition"
R5_1A_DIR = RESEARCH_ROOT / "ai-training/experiments/r5_1a_alignability_safe_addition_scoring"
AUTHORIZED_INTERPRETER = MAIN_ROOT / "ai-training/.venv/Scripts/python.exe"

for module_dir in (R5_2B_DIR,):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import r5_2b_train_execution_driver_tc1_pa1_env as frozen_driver  # noqa: E402


scorer = frozen_driver.scorer
TRAIN = frozen_driver.TRAIN
PHONE_VOCAB = frozen_driver.PHONE_VOCAB
HYPOTHESIS_BATCH_SIZE = frozen_driver.HYPOTHESIS_BATCH_SIZE

FROZEN_TRAIN_SCORES = R5_2B_DIR / "r5_2b_tc1_pa1_env_train_scores.jsonl"
V4_PATH = RESEARCH_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
CHECKPOINT_PATH = RESEARCH_ROOT / (
    "ai-training/experiments/r4_4c2_bigru_ctc_seed42/"
    "R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt"
)
SCORER_PATH = R5_2B_DIR / "r5_2b_scorer.py"
HISTORICAL_DRIVER_PATH = R5_2B_DIR / "r5_2b_train_execution_driver_tc1_pa1_env.py"

DIRECT_IDENTITIES = {
    "r5_4b_contract_manifest": (
        R5_4B_DIR / "R5_4B_CONTRACT_MANIFEST.json",
        "5087DB63E37C52A407720A9A7A11D542A4DD651510469CE5960CE8B3B2421297",
    ),
    "r5_4a_contract": (
        R5_4A_DIR / "R5_4A_AUDIT_CONTRACT.md",
        "111DDB77EC09B177505AEEF7B260476D8C718F745AD6170E6969926841338A27",
    ),
    "r5_4a_manifest": (
        R5_4A_DIR / "R5_4A_MANIFEST.json",
        "012322265AACD90009001AFCBF675C9228ADC55417B0A31029CD2A3EF32393B8",
    ),
    "r5_3a_closure_manifest": (
        R5_3A_DIR / "R5_3A_CLOSURE_MANIFEST.json",
        "9F57A293E0F4CA6E35E06761CFF54ABF70D967FE6ECF4F42ABFFF7C35397768C",
    ),
    "r5_3a_execution_manifest": (
        R5_3A_DIR / "train_execution_result/R5_3A_EXECUTION_MANIFEST.json",
        "1EC6DED9ECA6617AE683C9EF316B4435A11F6898B380E4592D779061BF73CE51",
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
    "checkpoint": (CHECKPOINT_PATH, "F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085"),
    "frozen_scorer": (SCORER_PATH, "2E44B79828DBB37B312CDC03C897A80803947A48F13864B26C454F3B8ED161A3"),
    "historical_driver": (HISTORICAL_DRIVER_PATH, "BC023CFD259406989CAEADABE3EB08AA945B7A612729ECA2C05711E0C387A5D0"),
    "frozen_train_scores": (FROZEN_TRAIN_SCORES, "EAC81ABDDF56C77E4A71AD96AE5985D7C00DD60559A27C3A3BAAB16859EEFF3C"),
}

EXPECTED = {
    "words": 16582,
    "candidates": 2977040,
    "alignable": 2976844,
    "impossible": 196,
    "affected_words": 65,
    "no_finite_words": 0,
}

EXPECTED_ENVIRONMENT = {
    "sys_executable": str(AUTHORIZED_INTERPRETER.resolve()),
    "python_version": "3.12.10",
    "torch_version": "2.12.0.dev20260408+cu128",
    "torch_cuda_version": "12.8",
    "cuda_available": True,
    "cuda_device_count": 1,
    "gpu_device_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
    "numpy_version": "2.4.6",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def resolve_manifest_artifact(manifest_path: Path, entry: dict[str, Any]) -> Path:
    repository = entry.get("repository")
    relative = Path(entry["relative_path"])
    if repository in {"main", "main_repository"}:
        return MAIN_ROOT / relative
    if repository in {"research", "research_worktree"}:
        return RESEARCH_ROOT / relative
    return manifest_path.parent / relative


def verify_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    for entry in manifest["artifacts"]:
        artifact = resolve_manifest_artifact(manifest_path, entry)
        if not artifact.is_file():
            failures.append({"relative_path": entry["relative_path"], "reason": "missing"})
            continue
        observed_size = artifact.stat().st_size
        observed_sha = sha256(artifact)
        if observed_size != int(entry["byte_size"]) or observed_sha != entry["sha256"]:
            failures.append(
                {
                    "relative_path": entry["relative_path"],
                    "reason": "identity_mismatch",
                    "expected_byte_size": int(entry["byte_size"]),
                    "observed_byte_size": observed_size,
                    "expected_sha256": entry["sha256"],
                    "observed_sha256": observed_sha,
                }
            )
    return {
        "path": str(manifest_path),
        "artifact_count": len(manifest["artifacts"]),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }


def verify_identities() -> dict[str, Any]:
    observed: dict[str, Any] = {}
    failures: dict[str, Any] = {}
    for name, (path, expected_sha) in DIRECT_IDENTITIES.items():
        if not path.is_file():
            failures[name] = {"reason": "missing", "path": str(path)}
            continue
        actual_sha = sha256(path)
        observed[name] = {"path": str(path), "sha256": actual_sha, "byte_size": path.stat().st_size}
        if actual_sha != expected_sha:
            failures[name] = {"expected_sha256": expected_sha, "observed_sha256": actual_sha}

    manifest_names = (
        "r5_4b_contract_manifest",
        "r5_4a_manifest",
        "r5_3a_closure_manifest",
        "r5_3a_execution_manifest",
        "r5_2b_execution_manifest",
        "r5_1a_execution_manifest",
    )
    manifest_audits = {
        name: verify_manifest(DIRECT_IDENTITIES[name][0])
        for name in manifest_names
        if DIRECT_IDENTITIES[name][0].is_file()
    }
    for name, audit in manifest_audits.items():
        if audit["status"] != "PASS":
            failures[f"{name}_entries"] = audit["failures"]
    if failures:
        raise RuntimeError(f"R5_4B_EXECUTION_BLOCKED_IDENTITY: {failures}")
    return {"status": "PASS", "identities": observed, "manifest_entry_audits": manifest_audits}


def verify_environment() -> dict[str, Any]:
    observed = {
        "sys_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpu_device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() else None,
        "numpy_version": np.__version__,
    }
    mismatch = {
        key: {"expected": expected, "observed": observed[key]}
        for key, expected in EXPECTED_ENVIRONMENT.items()
        if observed[key] != expected
    }
    if mismatch:
        raise RuntimeError(f"R5_4B_MATERIALIZATION_TECHNICAL_FAILURE: environment identity {mismatch}")
    return {
        "status": "PASS",
        "observed": observed,
        "packages_installed_or_modified": False,
        "dataset_accessed": False,
        "checkpoint_loaded": False,
    }


def iter_frozen_score_projection(include_winner: bool = False) -> Iterable[dict[str, Any]]:
    with FROZEN_TRAIN_SCORES.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            projected = {
                "source_identity": row["source_identity"],
                "speaker": row["speaker"],
            }
            if include_winner:
                projected.update(
                    {
                        "best_insert_phone_index": row["best_insert_phone_index"],
                        "best_insert_boundary": row["best_insert_boundary"],
                        "best_insert_score_value": row["best_insert_score_value"],
                    }
                )
            yield projected


def verify_population(words: list[dict[str, Any]], accounting: dict[str, Any]) -> dict[str, Any]:
    frozen_rows = list(iter_frozen_score_projection(False))
    observed_ids = [word["word_id"] for word in words]
    frozen_ids = [row["source_identity"] for row in frozen_rows]
    observed_speakers = [word["speaker_id"] for word in words]
    frozen_speakers = [row["speaker"] for row in frozen_rows]
    observed_counter = Counter(observed_ids)
    frozen_counter = Counter(frozen_ids)
    duplicates = sorted(key for key, count in observed_counter.items() if count != 1)
    frozen_duplicates = sorted(key for key, count in frozen_counter.items() if count != 1)
    missing = sorted(set(frozen_ids) - set(observed_ids))
    unexpected = sorted(set(observed_ids) - set(frozen_ids))
    ordered_matches = sum(left == right for left, right in zip(observed_ids, frozen_ids))
    speaker_matches = sum(left == right for left, right in zip(observed_speakers, frozen_speakers))
    status = "PASS"
    if (
        len(words) != EXPECTED["words"]
        or len(frozen_rows) != EXPECTED["words"]
        or duplicates
        or frozen_duplicates
        or missing
        or unexpected
        or ordered_matches != EXPECTED["words"]
        or speaker_matches != EXPECTED["words"]
        or tuple(sorted(set(observed_speakers))) != tuple(sorted(TRAIN))
        or accounting.get("status") != "PASS"
    ):
        status = "R5_4B_EXECUTION_BLOCKED_ROW_IDENTITY"
    return {
        "status": status,
        "runtime_words": len(words),
        "frozen_words": len(frozen_rows),
        "ordered_identity_matches": ordered_matches,
        "speaker_matches": speaker_matches,
        "duplicates": len(duplicates),
        "frozen_duplicates": len(frozen_duplicates),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_examples": missing[:20],
        "unexpected_examples": unexpected[:20],
        "train_speakers": list(TRAIN),
        "observed_speakers": sorted(set(observed_speakers)),
        "fuzzy_matching": False,
        "excluded_rows": 0,
    }


def score_insert_families(
    words: list[dict[str, Any]],
    raw_logits: list[torch.Tensor],
    log_probs: list[torch.Tensor],
    device: torch.device,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    insert_values = [
        np.full(len(scorer.enumerate_insert_candidates(word["expected_ids"])), -np.inf, dtype=np.float64)
        for word in words
    ]
    order = sorted(range(len(words)), key=lambda index: (log_probs[index].shape[0], index))
    batch_items: list[tuple[int, str, int, tuple[int, ...]]] = []
    processed_nonempty = 0
    processed_empty = 0
    insert_scored = 0
    started = time.perf_counter()

    def process(items: list[tuple[int, str, int, tuple[int, ...]]]) -> None:
        nonlocal processed_nonempty, insert_scored
        _raw, target_scores = frozen_driver.batch_ctc_scores(
            log_probs, [(item[0], item[3]) for item in items], device
        )
        for item, target_score in zip(items, target_scores):
            word_index, family, candidate_index, _target = item
            if family == "INSERT":
                insert_values[word_index][candidate_index] = target_score
                insert_scored += 1
        processed_nonempty += len(items)
        if processed_nonempty % (HYPOTHESIS_BATCH_SIZE * 25) < len(items):
            print(f"r5_4b_historical_nonempty_layout_scored={processed_nonempty}", flush=True)

    for word_index in order:
        steps = int(log_probs[word_index].shape[0])
        expected = words[word_index]["expected_ids"]
        constructed = {
            "KEEP": [scorer.construct_keep(expected)],
            "INSERT": scorer.enumerate_insert_candidates(expected),
            "SUB": scorer.enumerate_sub_candidates(expected),
            "DELETE": scorer.enumerate_delete_candidates(expected),
        }
        group: list[tuple[int, str, int, tuple[int, ...]]] = []
        for family in ("KEEP", "INSERT", "SUB", "DELETE"):
            for candidate_index, candidate in enumerate(constructed[family]):
                if not candidate.target:
                    score = scorer.score_target(raw_logits[word_index].to(device), [], steps).target_score
                    if not math.isfinite(score):
                        raise RuntimeError("R5_4B_MATERIALIZATION_TECHNICAL_FAILURE: nonfinite empty target")
                    processed_empty += 1
                elif scorer.is_alignable(candidate.target, steps):
                    group.append((word_index, family, candidate_index, candidate.target))
        if batch_items and len(batch_items) + len(group) > HYPOTHESIS_BATCH_SIZE:
            process(batch_items)
            batch_items = []
        batch_items.extend(group)
    if batch_items:
        process(batch_items)

    return insert_values, {
        "status": "PASS",
        "scientific_score_source": "imported frozen R5-2B scorer and historical batch adapter",
        "historical_all_family_batch_layout_reproduced": True,
        "hypothesis_batch_size": HYPOTHESIS_BATCH_SIZE,
        "processed_alignable_nonempty_all_families": processed_nonempty,
        "processed_empty_delete_diagnostic_scores": processed_empty,
        "alignable_insert_scores_persistable": insert_scored,
        "seconds": time.perf_counter() - started,
    }


def materialize_shards(
    output: Path,
    words: list[dict[str, Any]],
    log_probs: list[torch.Tensor],
    insert_values: list[np.ndarray],
) -> dict[str, Any]:
    paths = {speaker: output / f"r5_4b_insert_candidates_{speaker}.jsonl" for speaker in TRAIN}
    handles = {speaker: path.open("w", encoding="utf-8", newline="\n") for speaker, path in paths.items()}
    candidate_rows = Counter()
    word_counts = Counter()
    alignable_total = 0
    impossible_total = 0
    affected_words = 0
    no_finite_words = 0
    per_word_violations: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for word_index, word in enumerate(words):
            speaker = word["speaker_id"]
            candidates = scorer.enumerate_insert_candidates(word["expected_ids"])
            values = insert_values[word_index]
            steps = int(log_probs[word_index].shape[0])
            expected_count = 40 * (len(word["expected_ids"]) + 1)
            identities: set[tuple[int, int]] = set()
            word_impossible = 0
            finite = 0
            if len(candidates) != expected_count or len(values) != expected_count:
                per_word_violations.append(
                    {"source_identity": word["word_id"], "expected": expected_count, "observed": len(candidates)}
                )
            for candidate_index, candidate in enumerate(candidates):
                if candidate_index != 40 * int(candidate.boundary) + int(candidate.phone_index):
                    raise RuntimeError("R5_4B_MATERIALIZATION_TECHNICAL_FAILURE: candidate index drift")
                identity = (int(candidate.boundary), int(candidate.phone_index))
                if identity in identities:
                    raise RuntimeError("R5_4B_MATERIALIZATION_TECHNICAL_FAILURE: duplicate candidate identity")
                identities.add(identity)
                alignable = scorer.is_alignable(candidate.target, steps)
                value = float(values[candidate_index])
                if alignable:
                    if not math.isfinite(value):
                        raise RuntimeError("R5_4B_MATERIALIZATION_TECHNICAL_FAILURE: alignable INSERT is nonfinite")
                    score_value: float | None = value
                    score_hex: str | None = value.hex()
                    is_neg_inf = False
                    finite += 1
                    alignable_total += 1
                else:
                    if value != float("-inf"):
                        raise RuntimeError("R5_4B_MATERIALIZATION_TECHNICAL_FAILURE: impossible INSERT has finite score")
                    score_value = None
                    score_hex = None
                    is_neg_inf = True
                    word_impossible += 1
                    impossible_total += 1
                record = {
                    "source_identity": word["word_id"],
                    "speaker": speaker,
                    "candidate_index": candidate_index,
                    "expected_sequence_length": len(word["expected_ids"]),
                    "boundary_index": int(candidate.boundary),
                    "inserted_phone_index": int(candidate.phone_index),
                    "inserted_phone_symbol": PHONE_VOCAB[int(candidate.phone_index)],
                    "alignable": bool(alignable),
                    "authoritative_insert_target_score_value": score_value,
                    "authoritative_insert_target_score_float64_hex": score_hex,
                    "authoritative_insert_score_is_neg_inf": is_neg_inf,
                }
                handles[speaker].write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
                )
                candidate_rows[speaker] += 1
            word_counts[speaker] += 1
            affected_words += int(word_impossible > 0)
            no_finite_words += int(finite == 0)
            if (word_index + 1) % 500 == 0 or word_index + 1 == len(words):
                print(f"r5_4b_words_serialized={word_index + 1}/{len(words)}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    return {
        "status": "PASS" if not per_word_violations else "FAIL",
        "words": len(words),
        "total_candidates": int(sum(candidate_rows.values())),
        "alignable_candidates": alignable_total,
        "impossible_candidates": impossible_total,
        "words_affected_by_impossible_insert": affected_words,
        "words_without_finite_insert": no_finite_words,
        "per_word_40_times_n_plus_1_violations": len(per_word_violations),
        "per_word_violation_examples": per_word_violations[:20],
        "duplicate_candidate_identities": 0,
        "missing_candidate_identities": 0,
        "by_speaker": {
            speaker: {"words": int(word_counts[speaker]), "candidate_rows": int(candidate_rows[speaker])}
            for speaker in TRAIN
        },
        "seconds": time.perf_counter() - started,
    }


def derive_persisted_winners(output: Path, words: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    expected_by_id = {word["word_id"]: 40 * (len(word["expected_ids"]) + 1) for word in words}
    derived: dict[str, dict[str, Any]] = {}
    read_counts = Counter()
    ordering_violations = 0
    round_trip_violations = 0
    for speaker in TRAIN:
        path = output / f"r5_4b_insert_candidates_{speaker}.jsonl"
        previous_identity: str | None = None
        previous_index = -1
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                identity = row["source_identity"]
                index = int(row["candidate_index"])
                if identity == previous_identity:
                    if index != previous_index + 1:
                        ordering_violations += 1
                elif index != 0:
                    ordering_violations += 1
                previous_identity = identity
                previous_index = index
                read_counts[identity] += 1
                score_hex = row["authoritative_insert_target_score_float64_hex"]
                if score_hex is None:
                    continue
                score = float.fromhex(score_hex)
                if score != float(row["authoritative_insert_target_score_value"]):
                    round_trip_violations += 1
                key = (-score, int(row["boundary_index"]), int(row["inserted_phone_index"]))
                previous = derived.get(identity)
                if previous is None or key < previous["key"]:
                    derived[identity] = {
                        "key": key,
                        "boundary_index": int(row["boundary_index"]),
                        "inserted_phone_index": int(row["inserted_phone_index"]),
                        "score": score,
                        "score_hex": score_hex,
                    }
    count_violations = [
        {"source_identity": identity, "expected": expected, "observed": int(read_counts[identity])}
        for identity, expected in expected_by_id.items()
        if read_counts[identity] != expected
    ]
    return derived, {
        "persisted_words_with_finite_winner": len(derived),
        "ordering_violations": ordering_violations,
        "float_hex_round_trip_violations": round_trip_violations,
        "persisted_per_word_count_violations": len(count_violations),
        "count_violation_examples": count_violations[:20],
    }


def compare_frozen_winners(derived: dict[str, dict[str, Any]]) -> dict[str, Any]:
    identity_matches = 0
    score_matches = 0
    identity_mismatches: list[dict[str, Any]] = []
    score_mismatches: list[dict[str, Any]] = []
    frozen_count = 0
    for frozen in iter_frozen_score_projection(True):
        frozen_count += 1
        identity = frozen["source_identity"]
        winner = derived.get(identity)
        if winner is None:
            identity_mismatches.append({"source_identity": identity, "reason": "missing_materialized_winner"})
            score_mismatches.append({"source_identity": identity, "reason": "missing_materialized_winner"})
            continue
        same_identity = (
            winner["inserted_phone_index"] == int(frozen["best_insert_phone_index"])
            and winner["boundary_index"] == int(frozen["best_insert_boundary"])
        )
        identity_matches += int(same_identity)
        if not same_identity and len(identity_mismatches) < 100:
            identity_mismatches.append(
                {
                    "source_identity": identity,
                    "materialized": {
                        "phone": winner["inserted_phone_index"],
                        "boundary": winner["boundary_index"],
                    },
                    "frozen": {
                        "phone": int(frozen["best_insert_phone_index"]),
                        "boundary": int(frozen["best_insert_boundary"]),
                    },
                }
            )
        exact_score = winner["score"] == float(frozen["best_insert_score_value"])
        score_matches += int(exact_score)
        if not exact_score and len(score_mismatches) < 100:
            score_mismatches.append(
                {
                    "source_identity": identity,
                    "materialized_score": winner["score"],
                    "materialized_score_hex": winner["score_hex"],
                    "frozen_score": float(frozen["best_insert_score_value"]),
                    "frozen_score_hex": float(frozen["best_insert_score_value"]).hex(),
                }
            )
    return {
        "status": "PASS"
        if identity_matches == EXPECTED["words"] and score_matches == EXPECTED["words"]
        else "R5_4B_MATERIALIZATION_REPRODUCTION_FAILURE",
        "frozen_words_checked": frozen_count,
        "best_insert_identity_matches": identity_matches,
        "best_insert_identity_mismatches": EXPECTED["words"] - identity_matches,
        "best_insert_score_exact_matches": score_matches,
        "best_insert_score_mismatches": EXPECTED["words"] - score_matches,
        "identity_mismatch_examples": identity_mismatches,
        "score_mismatch_examples": score_mismatches,
        "winner_rule": "higher TARGET score; lower boundary; lower phone index",
        "score_comparison": "exact Python float == after float.fromhex",
        "atol": None,
        "rtol": None,
    }


def build_report(
    environment: dict[str, Any],
    population: dict[str, Any],
    accounting: dict[str, Any],
    reproduction: dict[str, Any],
    gates: dict[str, Any],
    status: str,
) -> str:
    observed = environment["observed"]
    lines = [
        "# R5-4B Full INSERT Candidate Materialization Result",
        "",
        f"Final status: `{status}`",
        "",
        "## Environment",
        "",
        f"- Interpreter: `{observed['sys_executable']}`",
        f"- Python: {observed['python_version']}",
        f"- PyTorch / CUDA: {observed['torch_version']} / {observed['torch_cuda_version']}",
        f"- GPU: {observed['gpu_device_name']}",
        "",
        "## Population and materialization",
        "",
        f"- Frozen word identities: {population['ordered_identity_matches']} / {EXPECTED['words']}",
        f"- Candidate rows: {accounting['total_candidates']}",
        f"- Alignable / impossible: {accounting['alignable_candidates']} / {accounting['impossible_candidates']}",
        f"- Words affected / no finite INSERT: {accounting['words_affected_by_impossible_insert']} / {accounting['words_without_finite_insert']}",
        "",
        "## Reproduction",
        "",
        f"- BEST_INSERT identity: {reproduction['best_insert_identity_matches']} / {EXPECTED['words']}",
        f"- BEST_INSERT exact score: {reproduction['best_insert_score_exact_matches']} / {EXPECTED['words']}",
        "",
        "## Materialization gates",
        "",
    ]
    for name in [f"M{index}" for index in range(1, 11)]:
        lines.append(f"- {name}: {gates['gates'][name]}")
    lines.extend(
        [
            "",
            f"Passed: {gates['passed_count']} / 10",
            "",
            "No Addition truth, recoverability metric, word-level performance metric, threshold search, VALIDATION, or TEST was used.",
            "",
        ]
    )
    return "\n".join(lines)


def execute(output: Path) -> None:
    if output.exists():
        raise RuntimeError(f"Refusing overwrite/rerun: {output}")
    started = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    environment = verify_environment()
    identities = verify_identities()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__).resolve(), output / "r5_4b_materialization_driver.py")
    write_json(output / "r5_4b_environment_identity.json", environment)
    write_json(output / "r5_4b_source_identity.json", identities)

    details, source_scan = frozen_driver.r5_1a_driver.scan_train_source()
    path_guard = frozen_driver.verify_train_annotation_paths(details)
    write_json(output / "r5_4b_path_guard.json", path_guard)
    if path_guard["status"] != "PASS":
        raise RuntimeError(f"R5_4B_MATERIALIZATION_TECHNICAL_FAILURE: {path_guard['status']}")

    device = torch.device("cuda")
    numeric_preflight = frozen_driver.verify_batch_adapter(device)
    if numeric_preflight["status"] != "PASS":
        raise RuntimeError("R5_4B_MATERIALIZATION_TECHNICAL_FAILURE: TC1 preflight")

    events_by_word, mapping = frozen_driver.r5_1a_driver.map_train_additions(
        details, frozen_driver.MANUAL_TEXTGRID_ROOT
    )
    words, accounting = frozen_driver.r5_1a_driver.build_words(frozen_driver.AUDIO_ROOT, events_by_word)
    accounting["source_scan"] = source_scan
    accounting["addition_mapping"] = mapping
    frozen_driver.validate_population(words, accounting)
    population = verify_population(words, accounting)
    write_json(output / "r5_4b_population_audit.json", population)
    if population["status"] != "PASS":
        raise RuntimeError(population["status"])

    raw_logits, log_probs, inference_report = frozen_driver.materialize_and_infer(words, device)
    historical_audit = frozen_driver.candidate_audit(words, raw_logits)
    if historical_audit["status"] != "PASS":
        raise RuntimeError("R5_4B_MATERIALIZATION_TECHNICAL_FAILURE: frozen candidate audit")
    insert_values, scoring_report = score_insert_families(words, raw_logits, log_probs, device)
    accounting_result = materialize_shards(output, words, log_probs, insert_values)
    del insert_values
    if device.type == "cuda":
        torch.cuda.empty_cache()

    derived, persisted_audit = derive_persisted_winners(output, words)
    reproduction = compare_frozen_winners(derived)
    reproduction["persisted_artifact_audit"] = persisted_audit
    write_json(output / "r5_4b_best_insert_reproduction.json", reproduction)

    accounting_result.update(
        {
            "expected": EXPECTED,
            "historical_all_family_candidate_audit": {
                "insert_total": historical_audit["candidate_totals_by_family"]["INSERT"],
                "insert_alignable": historical_audit["alignable_by_family"]["INSERT"],
                "insert_impossible": historical_audit["impossible_by_family"]["INSERT"],
                "words_with_no_finite_insert": historical_audit["words_with_no_finite_insert"],
            },
            "numeric_preflight": numeric_preflight,
            "inference": inference_report,
            "scoring": scoring_report,
            "started_at": started_at,
            "total_seconds_before_hashing": time.perf_counter() - started,
        }
    )
    write_json(output / "r5_4b_candidate_accounting.json", accounting_result)

    structural = {
        "M1": population["ordered_identity_matches"] == EXPECTED["words"] and population["runtime_words"] == EXPECTED["words"],
        "M2": accounting_result["total_candidates"] == EXPECTED["candidates"],
        "M3": accounting_result["alignable_candidates"] == EXPECTED["alignable"],
        "M4": accounting_result["impossible_candidates"] == EXPECTED["impossible"],
        "M5": accounting_result["words_affected_by_impossible_insert"] == EXPECTED["affected_words"],
        "M6": accounting_result["words_without_finite_insert"] == EXPECTED["no_finite_words"],
        "M7": accounting_result["per_word_40_times_n_plus_1_violations"] == 0
        and persisted_audit["persisted_per_word_count_violations"] == 0
        and persisted_audit["ordering_violations"] == 0
        and persisted_audit["float_hex_round_trip_violations"] == 0,
        "M8": reproduction["best_insert_identity_matches"] == EXPECTED["words"],
        "M9": reproduction["best_insert_score_exact_matches"] == EXPECTED["words"],
        "M10": True,
    }
    gates = {
        "gates": {name: "PASS" if value else "FAIL" for name, value in structural.items()},
        "passed_count": sum(structural.values()),
        "total": 10,
        "all_pass": all(structural.values()),
        "gate_type": "materialization/provenance only; not scientific performance",
    }
    status = (
        "R5_4B_INSERT_CANDIDATE_MATERIALIZATION_PASS"
        if gates["all_pass"]
        else "R5_4B_MATERIALIZATION_REPRODUCTION_FAILURE"
    )
    write_json(output / "r5_4b_materialization_gates.json", gates)

    protocol = {
        "neural_training": False,
        "checkpoint_loaded": True,
        "checkpoint_inference": True,
        "train_audio_accessed": True,
        "new_acoustic_candidate_artifact": True,
        "scientific_scorer_modified": False,
        "candidate_family_modified": False,
        "truth_used_during_materialization": False,
        "truth_fields_written_to_candidate_shards": False,
        "performance_metrics": False,
        "recoverability_metrics": False,
        "truth_ranks_inspected": False,
        "threshold_search": False,
        "classifier_fitting": False,
        "validation_paths_resolved": False,
        "validation_accessed": False,
        "test_paths_resolved": False,
        "test_accessed": False,
        "execution_count": 1,
    }
    write_json(output / "r5_4b_protocol_audit.json", protocol)

    shard_index = {}
    for speaker in TRAIN:
        path = output / f"r5_4b_insert_candidates_{speaker}.jsonl"
        shard_index[speaker] = {
            **accounting_result["by_speaker"][speaker],
            "relative_path": path.name,
            "byte_size": path.stat().st_size,
            "sha256": sha256(path),
        }
    index = {
        "format": "uncompressed UTF-8 JSON Lines; LF; no BOM",
        "speaker_order": list(TRAIN),
        "word_order": "frozen R5-2B word order filtered by speaker",
        "candidate_order": "ascending candidate_index",
        "shards": shard_index,
    }
    write_json(output / "r5_4b_materialization_index.json", index)
    report = build_report(environment, population, accounting_result, reproduction, gates, status)
    (output / "R5_4B_MATERIALIZATION_RESULT.md").write_text(report, encoding="utf-8", newline="\n")

    artifact_names = [
        "r5_4b_materialization_driver.py",
        "r5_4b_environment_identity.json",
        "r5_4b_source_identity.json",
        "r5_4b_path_guard.json",
        "r5_4b_population_audit.json",
        *[f"r5_4b_insert_candidates_{speaker}.jsonl" for speaker in TRAIN],
        "r5_4b_candidate_accounting.json",
        "r5_4b_best_insert_reproduction.json",
        "r5_4b_materialization_gates.json",
        "r5_4b_protocol_audit.json",
        "r5_4b_materialization_index.json",
        "R5_4B_MATERIALIZATION_RESULT.md",
    ]
    entries = [
        {
            "relative_path": name,
            "byte_size": (output / name).stat().st_size,
            "sha256": sha256(output / name),
        }
        for name in artifact_names
    ]
    failures = []
    for entry in entries:
        path = output / entry["relative_path"]
        if path.stat().st_size != entry["byte_size"] or sha256(path) != entry["sha256"]:
            failures.append(entry["relative_path"])
    manifest = {
        "stage": "R5-4B full INSERT candidate materialization execution",
        "status": status,
        "manifest_self_excluded": True,
        "artifact_count": len(entries),
        "artifacts": entries,
        "hash_audit": "HASH_AUDIT_PASS" if not failures else "HASH_AUDIT_FAIL",
        "failures": failures,
        "contract_manifest_sha256": DIRECT_IDENTITIES["r5_4b_contract_manifest"][1],
        "frozen_scorer_sha256": DIRECT_IDENTITIES["frozen_scorer"][1],
        "checkpoint_sha256": DIRECT_IDENTITIES["checkpoint"][1],
        "truth_blind_materialization": True,
        "performance_metrics_calculated": False,
        "recoverability_metrics_calculated": False,
    }
    write_json(output / "R5_4B_EXECUTION_MANIFEST.json", manifest)
    if failures:
        raise RuntimeError(f"R5_4B_MATERIALIZATION_TECHNICAL_FAILURE: hash audit {failures}")
    print(
        json.dumps(
            {
                "status": status,
                "gates_passed": gates["passed_count"],
                "artifact_count": len(entries),
                "manifest_sha256": sha256(output / "R5_4B_EXECUTION_MANIFEST.json"),
                "hash_audit": manifest["hash_audit"],
                "total_seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.execute:
        raise SystemExit("Frozen R5-4B materialization requires --execute")
    execute(arguments.output.resolve())


if __name__ == "__main__":
    main()
