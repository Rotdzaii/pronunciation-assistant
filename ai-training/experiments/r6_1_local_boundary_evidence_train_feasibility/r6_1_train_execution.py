"""Single frozen TRAIN-only R6-1 local-boundary feasibility execution."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import roc_auc_score


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = EXPERIMENT_DIR.parents[2]
MAIN_ROOT = Path(r"C:\Users\Admin\Documents\KLTN\pronunciation-assistant")
R6_0_DIR = RESEARCH_ROOT / "ai-training/experiments/r6_0_local_boundary_addition_feasibility"
R5_2B_DIR = RESEARCH_ROOT / "ai-training/experiments/r5_2b_relation_competition"
SCRIPTS_DIR = RESEARCH_ROOT / "ai-training/scripts"
STATIC_HELPERS = R6_0_DIR
AUDIO_ROOT = MAIN_ROOT / "ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0"
V4_PATH = RESEARCH_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
CHECKPOINT_PATH = RESEARCH_ROOT / (
    "ai-training/experiments/r4_4c2_bigru_ctc_seed42/"
    "R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt"
)
R5_SCORE_PATH = R5_2B_DIR / "r5_2b_tc1_pa1_env_train_scores.jsonl"

os.environ["L2_ARCTIC_ROOT"] = str(AUDIO_ROOT)
for source_dir in (STATIC_HELPERS, SCRIPTS_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import r6_0_boundary_labels as boundary_labels  # noqa: E402
import r6_0_boundary_mapping as boundary_mapping  # noqa: E402
import r6_0_local_features as local_features  # noqa: E402
import run_l2_arctic_observed_phone_r3_1a as r3  # noqa: E402
import run_r4_3a_word_sequence_design_audit as r43a  # noqa: E402
import run_r4_4b_ctc_sequence as r4b  # noqa: E402
import run_r4_4c2_bigru_ctc_sequence as r4c2  # noqa: E402


TRAIN = ("BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA")
TRAIN_SET = frozenset(TRAIN)
EXPECTED = {
    "words": 16582,
    "boundaries": 74426,
    "positive_boundaries": 324,
    "negative_boundaries": 74102,
    "events": 342,
    "single_addition_words": 304,
    "multiple_addition_words": 19,
    "mixed_substitution_addition_words": 117,
    "mixed_deletion_addition_words": 26,
}
EXPECTED_ENVIRONMENT = {
    "python": "3.12.10",
    "torch": "2.12.0.dev20260408+cu128",
    "cuda": "12.8",
    "cuda_available": True,
    "gpu": "NVIDIA GeForce RTX 5060 Laptop GPU",
}
EXPECTED_SHA = {
    "r6_0_contract_manifest": "EDB1A62CD6350AFC955C7A49B51C668BD0ED4217F7BBC9AD916525769DF8718A",
    "r6_0_static_manifest": "9EBF82AE11FA081E839837D9FFE5FDDD2D81BDD8E588EA5A3B840D722B1DD982",
    "r5_4a_contract": "111DDB77EC09B177505AEEF7B260476D8C718F745AD6170E6969926841338A27",
    "r5_4a_resume_manifest": "328E2732A3705593EA29D6C2682956E3B5D19F2175B4F7E2C55F888175A15046",
    "r5_4a_closure_manifest": "2E717A5ADEB6EEC35D645B7DF67FE24486960B9B87AAA6FFBEC4CDA3E8C3D94B",
    "r5_4b_execution_manifest": "FD1E4E66168654EC54778A489E0B443BD9C691DD1392233C4284CDF6CDF07B11",
    "r5_3a_closure_manifest": "9F57A293E0F4CA6E35E06761CFF54ABF70D967FE6ECF4F42ABFFF7C35397768C",
    "r5_2b_execution_manifest": "37F3C86FF11526B8AB54D173937A0F488125A1D0B610032CC8A5D47B11602387",
    "r5_1a_execution_manifest": "C9343A75CE26C2BEECA388EBA855E91AE0992D22C0C5D785308D0A98B89A3CD6",
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
    "checkpoint": "F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085",
    "scorer": "2E44B79828DBB37B312CDC03C897A80803947A48F13864B26C454F3B8ED161A3",
    "r5_scores": "EAC81ABDDF56C77E4A71AD96AE5985D7C00DD60559A27C3A3BAAB16859EEFF3C",
}
IDENTITY_PATHS = {
    "r6_0_contract_manifest": R6_0_DIR / "R6_0_MANIFEST.json",
    "r6_0_static_manifest": R6_0_DIR / "R6_0_STATIC_MANIFEST.json",
    "r5_4a_contract": RESEARCH_ROOT / "ai-training/experiments/r5_4a_insert_candidate_recoverability/R5_4A_AUDIT_CONTRACT.md",
    "r5_4a_resume_manifest": RESEARCH_ROOT / "ai-training/experiments/r5_4a_insert_candidate_recoverability/R5_4A_RESUME_MANIFEST.json",
    "r5_4a_closure_manifest": RESEARCH_ROOT / "ai-training/experiments/r5_4a_insert_candidate_recoverability/R5_4A_CLOSURE_MANIFEST.json",
    "r5_4b_execution_manifest": RESEARCH_ROOT / "ai-training/experiments/r5_4b_insert_candidate_materialization/materialization_result/R5_4B_EXECUTION_MANIFEST.json",
    "r5_3a_closure_manifest": RESEARCH_ROOT / "ai-training/experiments/r5_3a_evidence_separated_relation_scoring/R5_3A_CLOSURE_MANIFEST.json",
    "r5_2b_execution_manifest": R5_2B_DIR / "R5_2B_TC1_PA1_ENV_EXECUTION_MANIFEST.json",
    "r5_1a_execution_manifest": RESEARCH_ROOT / "ai-training/experiments/r5_1a_alignability_safe_addition_scoring/R5_1A_EXECUTION_MANIFEST.json",
    "v4": V4_PATH,
    "checkpoint": CHECKPOINT_PATH,
    "scorer": R5_2B_DIR / "r5_2b_scorer.py",
    "r5_scores": R5_SCORE_PATH,
}
OUTPUT_NAMES = (
    "r6_1_environment_identity.json",
    "r6_1_source_identity.json",
    "r6_1_execution_state.json",
    "r6_1_boundary_accounting.json",
    "r6_1_boundary_scores.jsonl",
    "r6_1_coverage.json",
    "r6_1_primary_auc.json",
    "r6_1_speaker_auc.json",
    "r6_1_position_diagnostics.json",
    "r6_1_mixed_multiple_diagnostics.json",
    "r6_1_descriptive_controls.json",
    "r6_1_gate_results.json",
    "r6_1_protocol_audit.json",
    "r6_1_final_status.json",
    "R6_1_TRAIN_FEASIBILITY_RESULT.md",
)


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def verify_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["artifacts"]:
        artifact = path.parent / entry["relative_path"]
        if not artifact.is_file():
            failures.append({"path": entry["relative_path"], "reason": "missing"})
            continue
        actual_size = artifact.stat().st_size
        actual_sha = sha256(artifact)
        if actual_size != int(entry["byte_size"]) or actual_sha != entry["sha256"]:
            failures.append({
                "path": entry["relative_path"], "reason": "identity mismatch",
                "expected_size": int(entry["byte_size"]), "actual_size": actual_size,
                "expected_sha256": entry["sha256"], "actual_sha256": actual_sha,
            })
    return {
        "manifest": path.name,
        "entries": len(manifest["artifacts"]),
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def verify_identities() -> dict[str, Any]:
    actual = {name: sha256(path) for name, path in IDENTITY_PATHS.items()}
    mismatches = {
        name: {"expected": EXPECTED_SHA[name], "actual": actual[name]}
        for name in EXPECTED_SHA if actual[name] != EXPECTED_SHA[name]
    }
    contract_audit = verify_manifest(IDENTITY_PATHS["r6_0_contract_manifest"])
    static_audit = verify_manifest(IDENTITY_PATHS["r6_0_static_manifest"])
    if mismatches or contract_audit["status"] != "PASS" or static_audit["status"] != "PASS":
        raise RuntimeError(
            "R6_1_EXECUTION_BLOCKED_IDENTITY: "
            + json.dumps({"mismatches": mismatches, "contract": contract_audit, "static": static_audit})
        )
    return {
        "status": "PASS",
        "expected_sha256": EXPECTED_SHA,
        "actual_sha256": actual,
        "r6_0_contract_manifest_audit": contract_audit,
        "r6_0_static_manifest_audit": static_audit,
    }


def environment_identity() -> dict[str, Any]:
    interpreter = str(Path(sys.executable).resolve())
    expected_interpreter = str((MAIN_ROOT / "ai-training/.venv/Scripts/python.exe").resolve())
    cuda_available = bool(torch.cuda.is_available())
    gpu = torch.cuda.get_device_name(0) if cuda_available and torch.cuda.device_count() else None
    observed = {
        "sys_executable": interpreter,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "gpu": gpu,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    checks = {
        "interpreter": interpreter.casefold() == expected_interpreter.casefold(),
        "python": observed["python"] == EXPECTED_ENVIRONMENT["python"],
        "torch": observed["torch"] == EXPECTED_ENVIRONMENT["torch"],
        "cuda": observed["torch_cuda"] == EXPECTED_ENVIRONMENT["cuda"],
        "cuda_available": observed["cuda_available"] is EXPECTED_ENVIRONMENT["cuda_available"],
        "gpu": observed["gpu"] == EXPECTED_ENVIRONMENT["gpu"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"R6_1_EXECUTION_TECHNICAL_FAILURE_ENVIRONMENT: {checks} observed={observed}")
    return {"status": "PASS", "expected": {"interpreter": expected_interpreter, **EXPECTED_ENVIRONMENT}, "observed": observed, "checks": checks}


def load_frozen_score_rows() -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    with R5_SCORE_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            identity = row["source_identity"]
            if identity in seen:
                raise RuntimeError(f"R6_1_EXECUTION_BLOCKED_BOUNDARY_ACCOUNTING: duplicate {identity}")
            if row["speaker"] not in TRAIN_SET:
                raise RuntimeError("R6_1_EXECUTION_BLOCKED_BOUNDARY_ACCOUNTING: non-TRAIN speaker")
            seen.add(identity)
            rows.append(row)
    return rows


def reconstruct_words(frozen_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r43a.AUDIT_SPEAKERS = TRAIN_SET
    records, reconstruction = r43a.build_word_records(AUDIO_ROOT)
    eligible = [
        word for word in records
        if word["split"] == "train" and bool(word["expected"])
        and not word["has_unresolved"] and word["boundary_available"]
    ]
    by_id: dict[str, dict[str, Any]] = {}
    for word in eligible:
        if word["word_id"] in by_id:
            raise RuntimeError(f"R6_1_EXECUTION_BLOCKED_BOUNDARY_ACCOUNTING: duplicate reconstruction {word['word_id']}")
        by_id[word["word_id"]] = word
    frozen_ids = {row["source_identity"] for row in frozen_rows}
    reconstructed_ids = set(by_id)
    missing = sorted(frozen_ids - reconstructed_ids)
    unexpected = sorted(reconstructed_ids - frozen_ids)
    if missing or unexpected or len(frozen_rows) != EXPECTED["words"]:
        raise RuntimeError(
            f"R6_1_EXECUTION_BLOCKED_BOUNDARY_ACCOUNTING: missing={len(missing)} "
            f"unexpected={len(unexpected)} frozen={len(frozen_rows)}"
        )
    ordered = []
    metadata_mismatches = []
    for row in frozen_rows:
        word = by_id[row["source_identity"]]
        expected = list(row["expected_sequence"])
        clean_expected = [source["expected"] for source in word["clean_rows"]]
        if expected != list(word["expected"]) or expected != clean_expected or row["speaker"] != word["speaker_id"]:
            metadata_mismatches.append(row["source_identity"])
        word["expected_ids"] = [r3.PHONE_TO_ID[phone] for phone in expected]
        ordered.append(word)
    if metadata_mismatches:
        raise RuntimeError(f"R6_1_EXECUTION_BLOCKED_BOUNDARY_ACCOUNTING: metadata mismatches={len(metadata_mismatches)}")
    return ordered, {
        "eligible_reconstructed_words": len(eligible),
        "frozen_rows": len(frozen_rows),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "duplicates": 0,
        "metadata_mismatches": len(metadata_mismatches),
        "reconstruction": reconstruction,
    }


def prepare_boundary_templates(
    words: list[dict[str, Any]], frozen_rows: list[dict[str, Any]]
) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    all_templates: list[list[dict[str, Any]]] = []
    boundary_total = positive_total = event_total = 0
    single_words = multiple_words = mixed_sub = mixed_del = 0
    by_speaker: dict[str, Counter[str]] = {speaker: Counter() for speaker in TRAIN}
    for word, frozen in zip(words, frozen_rows):
        expected = list(frozen["expected_sequence"])
        events = list(frozen["ground_truth_events"])
        counts = boundary_labels.boundary_event_counts(len(expected), events)
        word_event_total = int(frozen["ground_truth_addition_event_count"])
        if word_event_total != sum(counts):
            raise RuntimeError(f"R6_1_EXECUTION_BLOCKED_BOUNDARY_ACCOUNTING: event mismatch {word['word_id']}")
        single_words += int(word_event_total == 1)
        multiple_words += int(word_event_total > 1)
        cohorts = dict(frozen["relation_cohorts"])
        mixed_sub += int(bool(cohorts["substitution_addition"]))
        mixed_del += int(bool(cohorts["deletion_addition"]))
        intervals = [(float(row["start"]), float(row["end"])) for row in word["clean_rows"]]
        templates = []
        for boundary, event_count in enumerate(counts):
            gold_time = boundary_mapping.gold_boundary_time(intervals, boundary)
            position = "BEFORE_FIRST" if boundary == 0 else ("AFTER_FINAL" if boundary == len(expected) else "BETWEEN")
            templates.append({
                "source_identity": word["word_id"],
                "speaker": word["speaker_id"],
                "word": word["word"],
                "expected_sequence": expected,
                "expected_sequence_ids": list(word["expected_ids"]),
                "expected_sequence_length": len(expected),
                "boundary_index": boundary,
                "boundary_position": position,
                "gold_boundary_absolute_seconds": gold_time,
                "mfa_word_start_seconds": float(word["mfa_start"]),
                "mfa_word_end_seconds": float(word["mfa_end"]),
                "addition_boundary_label": event_count > 0,
                "addition_event_count": int(event_count),
                "word_addition_event_count": word_event_total,
                "relation_cohorts": cohorts,
            })
            boundary_total += 1
            positive_total += int(event_count > 0)
            event_total += int(event_count)
            counter = by_speaker[word["speaker_id"]]
            counter["boundaries"] += 1
            counter["positive_boundaries"] += int(event_count > 0)
            counter["events"] += int(event_count)
        all_templates.append(templates)
    actual = {
        "words": len(words),
        "boundaries": boundary_total,
        "positive_boundaries": positive_total,
        "negative_boundaries": boundary_total - positive_total,
        "events": event_total,
        "single_addition_words": single_words,
        "multiple_addition_words": multiple_words,
        "mixed_substitution_addition_words": mixed_sub,
        "mixed_deletion_addition_words": mixed_del,
    }
    if actual != EXPECTED:
        raise RuntimeError(f"R6_1_EXECUTION_BLOCKED_BOUNDARY_ACCOUNTING: expected={EXPECTED} actual={actual}")
    return all_templates, {
        "status": "PASS",
        "expected": EXPECTED,
        "actual": actual,
        "by_speaker": {speaker: dict(counter) for speaker, counter in by_speaker.items()},
    }


def infer_boundary_scores(
    words: list[dict[str, Any]], templates: list[list[dict[str, Any]]], device: torch.device
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features_by_word, feature_report = r4b.materialize_features(words, device, "r6_1_train_only")
    frame_lengths = [feature.shape[-1] for feature in features_by_word]
    expected_output_lengths = [r4b.encoder_steps(length) for length in frame_lengths]
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if int(checkpoint["blank_index"]) != 40 or tuple(checkpoint["vocabulary"]) != tuple(r3.PHONE_VOCAB):
        raise RuntimeError("R6_1_EXECUTION_BLOCKED_IDENTITY: checkpoint vocabulary/blank mismatch")
    model = r4c2.WordBiGRUCTCModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    del checkpoint

    ordered = sorted(range(len(words)), key=lambda index: (frame_lengths[index], index))
    output_rows_by_word: list[list[dict[str, Any]] | None] = [None] * len(words)
    invalid_reasons: Counter[str] = Counter()
    numeric_equivalence_max_abs = 0.0
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(ordered), 8):
            indexes = ordered[start:start + 8]
            maximum = max(frame_lengths[index] for index in indexes)
            batch = torch.zeros((len(indexes), 1, r4b.N_MELS, maximum), dtype=torch.float32)
            lengths = []
            for position, index in enumerate(indexes):
                feature = features_by_word[index]
                batch[position, 0, :, :feature.shape[-1]] = feature
                lengths.append(feature.shape[-1])
            logits, output_lengths = model(
                batch.to(device, non_blocking=True),
                torch.tensor(lengths, dtype=torch.long, device=device),
            )
            logits_cpu = logits.detach().cpu()
            for position, index in enumerate(indexes):
                steps = int(output_lengths[position].item())
                if steps != expected_output_lengths[index] or steps <= 0:
                    raise RuntimeError("R6_1_EXECUTION_TECHNICAL_FAILURE_BOUNDARY_MAPPING: output length mismatch")
                word_logits = logits_cpu[position, :steps].contiguous()
                if word_logits.dtype != torch.float32 or word_logits.shape[1] != 41 or not bool(torch.isfinite(word_logits).all()):
                    raise RuntimeError("R6_1_EXECUTION_TECHNICAL_FAILURE_CHECKPOINT_OUTPUT")
                word_rows = []
                for template in templates[index]:
                    row = dict(template)
                    row["encoder_output_steps"] = steps
                    row["mapping_valid"] = False
                    row["mapping_failure_reason"] = None
                    try:
                        center = boundary_mapping.nearest_output_index(
                            template["gold_boundary_absolute_seconds"],
                            template["mfa_word_start_seconds"],
                            steps,
                            template["mfa_word_end_seconds"],
                        )
                    except ValueError as exc:
                        reason = str(exc)
                        invalid_reasons[reason] += 1
                        row.update({
                            "relative_boundary_time_seconds": boundary_mapping.relative_boundary_time(
                                template["gold_boundary_absolute_seconds"], template["mfa_word_start_seconds"]
                            ),
                            "selected_center_index": None,
                            "selected_center_time_seconds": None,
                            "window_indices": [],
                            "available_window_frame_count": 0,
                            "adjacent_expected_phone_ids": sorted(local_features.adjacent_expected_phone_set(
                                template["expected_sequence_ids"], template["boundary_index"]
                            )),
                            "mean_unexpected_phone_mass": None,
                            "peak_unexpected_phone_posterior": None,
                            "peak_unexpected_phone_index": None,
                            "peak_unexpected_phone": None,
                            "peak_unexpected_frame_index": None,
                            "mean_nonblank_mass": None,
                            "mapping_failure_reason": reason,
                        })
                    else:
                        window = boundary_mapping.five_step_window(center, steps)
                        # Feature computation is label-blind and any semantic/numeric failure is fatal.
                        evidence = local_features.compute_local_evidence(
                            word_logits,
                            window,
                            template["expected_sequence_ids"],
                            template["boundary_index"],
                        )
                        difference = abs(
                            float(evidence["mean_nonblank_mass"])
                            - float(evidence["mean_nonblank_mass_from_phone_sum"])
                        )
                        numeric_equivalence_max_abs = max(numeric_equivalence_max_abs, difference)
                        if difference > 1e-6:
                            raise RuntimeError("R6_1_EXECUTION_TECHNICAL_FAILURE_NUMERICAL: nonblank mismatch")
                        row.update({
                            "relative_boundary_time_seconds": boundary_mapping.relative_boundary_time(
                                template["gold_boundary_absolute_seconds"], template["mfa_word_start_seconds"]
                            ),
                            "selected_center_index": center,
                            "selected_center_time_seconds": boundary_mapping.nominal_output_center(center),
                            "window_indices": list(window),
                            "available_window_frame_count": len(window),
                            "adjacent_expected_phone_ids": evidence["adjacent_expected_phone_ids"],
                            "mean_unexpected_phone_mass": float(evidence["mean_unexpected_phone_mass"]),
                            "peak_unexpected_phone_posterior": float(evidence["peak_unexpected_phone_posterior"]),
                            "peak_unexpected_phone_index": int(evidence["peak_unexpected_phone_index"]),
                            "peak_unexpected_phone": r3.PHONE_VOCAB[int(evidence["peak_unexpected_phone_index"])],
                            "peak_unexpected_frame_index": int(evidence["peak_unexpected_frame_index"]),
                            "mean_nonblank_mass": float(evidence["mean_nonblank_mass"]),
                            "mapping_valid": True,
                        })
                    word_rows.append(row)
                output_rows_by_word[index] = word_rows
            completed = min(start + len(indexes), len(ordered))
            if start == 0 or completed == len(ordered) or (start // 8) % 250 == 0:
                print(f"r6_1_train_inference={completed}/{len(ordered)}", flush=True)
    if any(rows is None for rows in output_rows_by_word):
        raise RuntimeError("R6_1_EXECUTION_TECHNICAL_FAILURE_INCOMPLETE_INFERENCE")
    output_rows = [row for rows in output_rows_by_word if rows is not None for row in rows]
    if len(output_rows) != EXPECTED["boundaries"]:
        raise RuntimeError("R6_1_EXECUTION_BLOCKED_BOUNDARY_ACCOUNTING: scored boundary count")
    return output_rows, {
        "checkpoint_loaded": True,
        "checkpoint_inference": True,
        "inference_seconds": time.perf_counter() - started,
        "feature_materialization": feature_report,
        "invalid_boundary_mappings": int(sum(invalid_reasons.values())),
        "invalid_mapping_reasons": dict(invalid_reasons),
        "maximum_nonblank_equivalence_absolute_difference": numeric_equivalence_max_abs,
    }


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "p25": None, "p75": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def auc_for(rows: list[dict[str, Any]], score_field: str = "mean_unexpected_phone_mass") -> float:
    valid = [row for row in rows if row[score_field] is not None]
    truth = np.asarray([row["addition_boundary_label"] for row in valid], dtype=bool)
    if truth.size == 0 or np.unique(truth).size != 2:
        raise RuntimeError("R6_1_EXECUTION_TECHNICAL_FAILURE_UNDEFINED_AUC")
    scores = np.asarray([row[score_field] for row in valid], dtype=np.float64)
    return float(roc_auc_score(truth, scores))


def summarize_metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    valid = [row for row in rows if row["mapping_valid"]]
    positive = [row for row in valid if row["addition_boundary_label"]]
    negative = [row for row in valid if not row["addition_boundary_label"]]
    covered_events = sum(int(row["addition_event_count"]) for row in positive)
    covered_positive_boundaries = len(positive)
    coverage = {
        "total_addition_events": EXPECTED["events"],
        "covered_addition_events": covered_events,
        "uncovered_addition_events": EXPECTED["events"] - covered_events,
        "addition_event_anchor_window_coverage": float(covered_events / EXPECTED["events"]),
        "total_positive_boundary_identities": EXPECTED["positive_boundaries"],
        "covered_positive_boundary_identities": covered_positive_boundaries,
        "uncovered_positive_boundary_identities": EXPECTED["positive_boundaries"] - covered_positive_boundaries,
        "positive_boundary_anchor_window_coverage": float(covered_positive_boundaries / EXPECTED["positive_boundaries"]),
        "valid_all_boundaries": len(valid),
        "invalid_all_boundaries": len(rows) - len(valid),
    }
    primary = {
        "primary_score": "MEAN_UNEXPECTED_PHONE_MASS",
        "formula": "mean over available window frames of posterior mass over canonical phone classes 0..39 excluding adjacent expected phones; blank excluded",
        "valid_boundary_instances": len(valid),
        "positive_instances": len(positive),
        "negative_instances": len(negative),
        "pooled_boundary_roc_auc": auc_for(rows),
        "positive_score_distribution": distribution(row["mean_unexpected_phone_mass"] for row in positive),
        "negative_score_distribution": distribution(row["mean_unexpected_phone_mass"] for row in negative),
    }
    speaker_rows = []
    for speaker in TRAIN:
        subset = [row for row in valid if row["speaker"] == speaker]
        positives = [row for row in subset if row["addition_boundary_label"]]
        negatives = [row for row in subset if not row["addition_boundary_label"]]
        speaker_rows.append({
            "speaker": speaker,
            "valid_boundaries": len(subset),
            "positive_boundaries": len(positives),
            "negative_boundaries": len(negatives),
            "roc_auc": auc_for(subset),
            "positive_median": float(np.median([row["mean_unexpected_phone_mass"] for row in positives])),
            "negative_median": float(np.median([row["mean_unexpected_phone_mass"] for row in negatives])),
        })
    speaker_aucs = [row["roc_auc"] for row in speaker_rows]
    speakers = {
        "speakers": speaker_rows,
        "ordinary_median_roc_auc": float(np.median(np.asarray(speaker_aucs, dtype=np.float64))),
        "count_roc_auc_gt_0_55": sum(value > 0.55 for value in speaker_aucs),
        "speaker_count": len(speaker_rows),
    }
    positions = []
    for position in ("BEFORE_FIRST", "BETWEEN", "AFTER_FINAL"):
        subset = [row for row in valid if row["boundary_position"] == position]
        positives = [row for row in subset if row["addition_boundary_label"]]
        negatives = [row for row in subset if not row["addition_boundary_label"]]
        positions.append({
            "position": position,
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "roc_auc": auc_for(subset) if positives and negatives else None,
            "positive_score_median": float(np.median([row["mean_unexpected_phone_mass"] for row in positives])) if positives else None,
            "negative_score_median": float(np.median([row["mean_unexpected_phone_mass"] for row in negatives])) if negatives else None,
        })
    position_diagnostics = {"unit": "valid boundary identity", "positions": positions, "gating": False}

    def group(name: str, predicate: Any) -> dict[str, Any]:
        subset = [row for row in positive if predicate(row)]
        return {
            "group": name,
            "positive_boundary_count": len(subset),
            "addition_event_count": sum(int(row["addition_event_count"]) for row in subset),
            "primary_score_distribution": distribution(row["mean_unexpected_phone_mass"] for row in subset),
        }

    mixed_multiple = {
        "unit": "valid positive boundary identity; event multiplicity reported separately",
        "gating": False,
        "groups": [
            group("single_addition_words", lambda row: row["word_addition_event_count"] == 1),
            group("multiple_addition_words", lambda row: row["word_addition_event_count"] > 1),
            group("mixed_substitution_addition", lambda row: bool(row["relation_cohorts"]["substitution_addition"])),
            group("mixed_deletion_addition", lambda row: bool(row["relation_cohorts"]["deletion_addition"])),
        ],
    }
    controls = {
        "gating": False,
        "auc_calculated": False,
        "feature_selection_performed": False,
        "PEAK_UNEXPECTED_PHONE_POSTERIOR": {
            "positive_distribution": distribution(row["peak_unexpected_phone_posterior"] for row in positive),
            "negative_distribution": distribution(row["peak_unexpected_phone_posterior"] for row in negative),
        },
        "MEAN_NONBLANK_MASS": {
            "positive_distribution": distribution(row["mean_nonblank_mass"] for row in positive),
            "negative_distribution": distribution(row["mean_nonblank_mass"] for row in negative),
        },
    }
    return coverage, primary, speakers, position_diagnostics, mixed_multiple, controls


def gate_results(coverage: dict[str, Any], primary: dict[str, Any], speakers: dict[str, Any]) -> dict[str, Any]:
    gates = {
        "F1": {
            "metric": "addition_event_anchor_window_coverage",
            "value": coverage["addition_event_anchor_window_coverage"],
            "operator": ">=", "threshold": 0.99,
            "pass": coverage["addition_event_anchor_window_coverage"] >= 0.99,
        },
        "F2": {
            "metric": "pooled_boundary_roc_auc",
            "value": primary["pooled_boundary_roc_auc"],
            "operator": ">=", "threshold": 0.65,
            "pass": primary["pooled_boundary_roc_auc"] >= 0.65,
        },
        "F3": {
            "metric": "median_speaker_roc_auc",
            "value": speakers["ordinary_median_roc_auc"],
            "operator": ">=", "threshold": 0.60,
            "pass": speakers["ordinary_median_roc_auc"] >= 0.60,
        },
        "F4": {
            "metric": "speakers_with_roc_auc_gt_0_55",
            "value": speakers["count_roc_auc_gt_0_55"],
            "operator": ">=", "threshold": 9, "denominator": 12,
            "pass": speakers["count_roc_auc_gt_0_55"] >= 9,
        },
    }
    passed = sum(int(gate["pass"]) for gate in gates.values())
    status = "R6_1_LOCAL_BOUNDARY_EVIDENCE_FEASIBLE" if passed == 4 else "R6_1_LOCAL_BOUNDARY_EVIDENCE_NOT_CONFIRMED"
    return {"gates": gates, "passed": passed, "total": 4, "status": status}


def render_report(
    environment: dict[str, Any], accounting: dict[str, Any], coverage: dict[str, Any],
    primary: dict[str, Any], speakers: dict[str, Any], positions: dict[str, Any],
    mixed: dict[str, Any], controls: dict[str, Any], gates: dict[str, Any], protocol: dict[str, Any],
) -> str:
    speaker_lines = [f"- {row['speaker']}: {row['roc_auc']!r}" for row in speakers["speakers"]]
    position_lines = [
        f"- {row['position']}: positive={row['positive_count']}, negative={row['negative_count']}, "
        f"AUC={row['roc_auc']!r}, positive median={row['positive_score_median']!r}, "
        f"negative median={row['negative_score_median']!r}"
        for row in positions["positions"]
    ]
    group_lines = [
        f"- {row['group']}: boundaries={row['positive_boundary_count']}, events={row['addition_event_count']}, "
        f"median={row['primary_score_distribution']['median']!r}"
        for row in mixed["groups"]
    ]
    gate_lines = [
        f"- {name}: {'PASS' if gate['pass'] else 'FAIL'} ({gate['value']!r} {gate['operator']} {gate['threshold']!r})"
        for name, gate in gates["gates"].items()
    ]
    return "\n".join([
        "# R6-1 Frozen TRAIN Local Boundary Evidence Feasibility", "",
        f"Final status: `{gates['status']}`", "",
        "## Identity and environment", "",
        "All frozen identities passed. Interpreter: " + environment["observed"]["sys_executable"],
        f"Python {environment['observed']['python']}; torch {environment['observed']['torch']}; "
        f"CUDA {environment['observed']['torch_cuda']}; GPU {environment['observed']['gpu']}.", "",
        "## Structural accounting", "",
        f"Words={accounting['actual']['words']}; boundaries={accounting['actual']['boundaries']}; "
        f"positive={accounting['actual']['positive_boundaries']}; negative={accounting['actual']['negative_boundaries']}; "
        f"events={accounting['actual']['events']}.", "",
        "## Coverage", "",
        f"Events covered={coverage['covered_addition_events']}/{coverage['total_addition_events']} "
        f"({coverage['addition_event_anchor_window_coverage']!r}).", "",
        "## Primary score", "",
        "Primary score: `MEAN_UNEXPECTED_PHONE_MASS`.",
        f"Pooled boundary ROC-AUC: {primary['pooled_boundary_roc_auc']!r}.", "",
        "## Speaker ROC-AUC", "", *speaker_lines, "",
        f"Median: {speakers['ordinary_median_roc_auc']!r}; count > 0.55: "
        f"{speakers['count_roc_auc_gt_0_55']}/12.", "",
        "## Position diagnostics", "", *position_lines, "",
        "## Mixed and multiple diagnostics", "", *group_lines, "",
        "## Descriptive controls (non-gating)", "",
        f"Peak unexpected posterior positive/negative medians: "
        f"{controls['PEAK_UNEXPECTED_PHONE_POSTERIOR']['positive_distribution']['median']!r} / "
        f"{controls['PEAK_UNEXPECTED_PHONE_POSTERIOR']['negative_distribution']['median']!r}.",
        f"Mean nonblank mass positive/negative medians: "
        f"{controls['MEAN_NONBLANK_MASS']['positive_distribution']['median']!r} / "
        f"{controls['MEAN_NONBLANK_MASS']['negative_distribution']['median']!r}.", "",
        "## Frozen gates", "", *gate_lines, "",
        f"Result: {gates['passed']}/4.", "",
        "## Protocol", "",
        f"Training={protocol['neural_training']}; checkpoint inference={protocol['checkpoint_inference']}; "
        f"classifier fitting={protocol['classifier_fitting']}; threshold search={protocol['threshold_search']}; "
        f"VALIDATION={protocol['validation_accessed']}; TEST={protocol['test_accessed']}; execution count=1.", "",
    ])


def create_manifest() -> dict[str, Any]:
    artifacts = []
    for name in ("r6_1_train_execution.py",) + OUTPUT_NAMES:
        path = EXPERIMENT_DIR / name
        artifacts.append({"relative_path": name, "byte_size": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "stage": "R6-1 frozen TRAIN local boundary evidence feasibility",
        "final_status": json.loads((EXPERIMENT_DIR / "r6_1_final_status.json").read_text(encoding="utf-8"))["status"],
        "manifest_self_excluded": True,
        "hash_algorithm": "SHA-256",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "independent_hash_audit": "HASH_AUDIT_PASS",
        "execution_count": 1,
    }
    write_json(EXPERIMENT_DIR / "R6_1_EXECUTION_MANIFEST.json", manifest)
    reread = json.loads((EXPERIMENT_DIR / "R6_1_EXECUTION_MANIFEST.json").read_text(encoding="utf-8"))
    for entry in reread["artifacts"]:
        path = EXPERIMENT_DIR / entry["relative_path"]
        if path.stat().st_size != entry["byte_size"] or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"R6_1_EXECUTION_TECHNICAL_FAILURE_HASH_AUDIT: {entry['relative_path']}")
    return manifest


def main() -> None:
    state_path = EXPERIMENT_DIR / "r6_1_execution_state.json"
    if state_path.exists():
        raise RuntimeError("R6_1_EXECUTION_TECHNICAL_FAILURE_ALREADY_ATTEMPTED")
    for name in OUTPUT_NAMES:
        if (EXPERIMENT_DIR / name).exists():
            raise RuntimeError(f"R6_1_EXECUTION_TECHNICAL_FAILURE_EXISTING_ARTIFACT: {name}")
    write_json(state_path, {"stage": "R6-1", "execution_count": 1, "state": "identity_preflight"})
    started = time.perf_counter()
    identity = verify_identities()
    write_json(EXPERIMENT_DIR / "r6_1_source_identity.json", identity)
    environment = environment_identity()
    write_json(EXPERIMENT_DIR / "r6_1_environment_identity.json", environment)

    frozen_rows = load_frozen_score_rows()
    words, reconstruction = reconstruct_words(frozen_rows)
    templates, accounting = prepare_boundary_templates(words, frozen_rows)
    accounting["join"] = reconstruction
    write_json(EXPERIMENT_DIR / "r6_1_boundary_accounting.json", accounting)
    write_json(state_path, {"stage": "R6-1", "execution_count": 1, "state": "checkpoint_inference_authorized", "structural_accounting": "PASS"})

    device = torch.device("cuda")
    rows, inference = infer_boundary_scores(words, templates, device)
    write_jsonl(EXPERIMENT_DIR / "r6_1_boundary_scores.jsonl", rows)
    coverage, primary, speakers, positions, mixed, controls = summarize_metrics(rows)
    gates = gate_results(coverage, primary, speakers)

    write_json(EXPERIMENT_DIR / "r6_1_coverage.json", coverage)
    write_json(EXPERIMENT_DIR / "r6_1_primary_auc.json", primary)
    write_json(EXPERIMENT_DIR / "r6_1_speaker_auc.json", speakers)
    write_json(EXPERIMENT_DIR / "r6_1_position_diagnostics.json", positions)
    write_json(EXPERIMENT_DIR / "r6_1_mixed_multiple_diagnostics.json", mixed)
    write_json(EXPERIMENT_DIR / "r6_1_descriptive_controls.json", controls)
    write_json(EXPERIMENT_DIR / "r6_1_gate_results.json", gates)

    protocol = {
        "neural_training": False,
        "fine_tuning": False,
        "checkpoint_loaded": True,
        "checkpoint_inference": True,
        "train_audio_accessed": True,
        "train_gold_boundaries_accessed": True,
        "classifier_fitting": False,
        "threshold_search": False,
        "word_level_performance": False,
        "recoverability_topk": False,
        "descriptive_control_auc_calculated": False,
        "validation_accessed": False,
        "validation_paths_resolved": False,
        "test_accessed": False,
        "test_paths_resolved": False,
        "execution_count": 1,
        "window_modified": False,
        "feature_modified": False,
        "boundary_mapping_modified": False,
        "elapsed_seconds": time.perf_counter() - started,
        "inference": inference,
    }
    write_json(EXPERIMENT_DIR / "r6_1_protocol_audit.json", protocol)
    final = {
        "status": gates["status"],
        "gates_passed": gates["passed"],
        "gates_total": 4,
        "primary_score": "MEAN_UNEXPECTED_PHONE_MASS",
        "pooled_boundary_roc_auc": primary["pooled_boundary_roc_auc"],
        "median_speaker_roc_auc": speakers["ordinary_median_roc_auc"],
        "speakers_auc_gt_0_55": speakers["count_roc_auc_gt_0_55"],
        "addition_event_coverage": coverage["addition_event_anchor_window_coverage"],
        "scientific_scope": "TRAIN boundary-level continuous feasibility only; not a word-level detector",
    }
    write_json(EXPERIMENT_DIR / "r6_1_final_status.json", final)
    report = render_report(environment, accounting, coverage, primary, speakers, positions, mixed, controls, gates, protocol)
    (EXPERIMENT_DIR / "R6_1_TRAIN_FEASIBILITY_RESULT.md").write_text(report, encoding="utf-8", newline="\n")
    write_json(state_path, {"stage": "R6-1", "execution_count": 1, "state": "complete", "status": gates["status"]})
    manifest = create_manifest()
    print(json.dumps({
        "status": gates["status"], "gates": f"{gates['passed']}/4",
        "pooled_auc": primary["pooled_boundary_roc_auc"],
        "median_speaker_auc": speakers["ordinary_median_roc_auc"],
        "manifest_artifacts": manifest["artifact_count"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        technical_stop = {
            "stage": "R6-1",
            "status": "R6_1_EXECUTION_TECHNICAL_FAILURE",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "rerun_performed": False,
        }
        write_json(EXPERIMENT_DIR / "r6_1_technical_stop.json", technical_stop)
        state_path = EXPERIMENT_DIR / "r6_1_execution_state.json"
        if state_path.exists():
            write_json(state_path, {
                "stage": "R6-1", "execution_count": 1, "state": "technical_failure",
                "status": technical_stop["status"], "message": technical_stop["message"],
            })
        raise
