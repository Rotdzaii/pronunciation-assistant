from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
R5_1A_DIR = REPO_ROOT / "ai-training/experiments/r5_1a_alignability_safe_addition_scoring"
SCRIPTS_DIR = REPO_ROOT / "ai-training/scripts"
AUDIO_ROOT = REPO_ROOT / "ai-training/datasets/l2-arctic/raw/l2arctic_release_v5.0"
os.environ.setdefault("L2_ARCTIC_ROOT", str(AUDIO_ROOT))
for source_dir in (EXPERIMENT_DIR, R5_1A_DIR, SCRIPTS_DIR):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

import r5_2b_scorer as scorer  # noqa: E402
import r5_1a_scorer as extended_helpers  # noqa: E402
import r5_1a_train_execution_driver as r5_1a_driver  # noqa: E402
import run_l2_arctic_observed_phone_r3_1a as r3  # noqa: E402
import run_r4_4b_ctc_sequence as r4b  # noqa: E402
import run_r4_4c2_bigru_ctc_sequence as r4c2  # noqa: E402


TRAIN = ("BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA")
PHONE_VOCAB = tuple(r3.PHONE_VOCAB)
V4_PATH = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
CHECKPOINT_PATH = REPO_ROOT / (
    "ai-training/experiments/r4_4c2_bigru_ctc_seed42/"
    "R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt"
)
R5_2A_MANIFEST = REPO_ROOT / "ai-training/experiments/r5_2a_addition_fix_feasibility/artifact_hashes.json"
R5_1A_SCORER = R5_1A_DIR / "r5_1a_scorer.py"

CONTRACT_PATH = EXPERIMENT_DIR / "R5_2B_DEVELOPMENT_CONTRACT.json"
PREREG_PATH = EXPERIMENT_DIR / "R5_2B_PREREGISTRATION.md"
CONTRACT_MANIFEST = EXPERIMENT_DIR / "artifact_hashes.json"
STATIC_MANIFEST = EXPERIMENT_DIR / "R5_2B_STATIC_VERIFICATION_MANIFEST.json"
STATIC_RUN = EXPERIMENT_DIR / "r5_2b_static_run_1.json"
FROZEN_SCORER = EXPERIMENT_DIR / "r5_2b_scorer.py"

EXPECTED_SHA = {
    "contract": "55572805C1878D41D4B6C41E1C7A31B2C3725A81050942494F0E1292092FEF38",
    "preregistration": "6D2DD9672222D174B32F0E25448419F42FE179522D2D8AA37C7766C8A1BBFBBA",
    "contract_manifest": "5A1E82FD9E0209DB371F7E313125FAC64434BA6DA6E677CFF20E0AE3336F583E",
    "static_manifest": "0D380ACA5C0CF101E60DFF043221BBE6AE267BA9FFDAD1C677C4E7FFCA23F80F",
    "static_run": "F0E54A0A4DBC1378F424E6C8FE8B0A9861CDD72DCBE72C045B96660324723197",
    "r5_2a_manifest": "F8E476F8FFA6AB2CD0F833B17D667D0D9B6CE3FC6B9FA1A7F5C1A3D71BA753E1",
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
    "checkpoint": "F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085",
    "r5_1a_scorer": "4DE49C9070C973EE44EFBD09DFC063C436779E723D12EC7A7A2BC4A06AF35F90",
    "r5_2b_scorer": "2E44B79828DBB37B312CDC03C897A80803947A48F13864B26C454F3B8ED161A3",
}
IDENTITY_PATHS = {
    "contract": CONTRACT_PATH,
    "preregistration": PREREG_PATH,
    "contract_manifest": CONTRACT_MANIFEST,
    "static_manifest": STATIC_MANIFEST,
    "static_run": STATIC_RUN,
    "r5_2a_manifest": R5_2A_MANIFEST,
    "v4": V4_PATH,
    "checkpoint": CHECKPOINT_PATH,
    "r5_1a_scorer": R5_1A_SCORER,
    "r5_2b_scorer": FROZEN_SCORER,
}
EXPECTED_COUNTS = {
    "words": 16582,
    "positive_words": 323,
    "negative_words": 16259,
    "source_events": 423,
    "runtime_events": 342,
    "multiple_addition_words": 19,
    "mixed_substitution_addition_words": 117,
    "mixed_deletion_addition_words": 26,
    "without_manual_word": 49,
    "without_expected_sequence": 14,
    "unresolved_cooccurring_events": 18,
}
COMPARATORS = {
    "addition_vs_all_nonaddition_roc_auc": 0.7734833025081417,
    "addition_vs_correct_only_roc_auc": 0.8023528214095370,
    "binary_macro_f1": 0.5551978767901391,
    "addition_f1": 0.1379980563654033,
    "correct_only_far": 0.03491295938104449,
    "substitution_negative_far": 0.05016116035455278,
    "deletion_negative_far": 0.03349964362081254,
    "exact_event_f1": 0.04389312977099236,
}
GATES = {
    "G1": ("addition_vs_all_nonaddition_roc_auc", ">=", 0.70),
    "G2": ("addition_vs_correct_only_roc_auc", ">=", 0.70),
    "G3": ("oof_binary_macro_f1", ">", COMPARATORS["binary_macro_f1"]),
    "G4": ("oof_addition_f1", ">", COMPARATORS["addition_f1"]),
    "G5": ("correct_only_false_addition_rate", "<=", COMPARATORS["correct_only_far"]),
    "G6": ("substitution_negative_false_addition_rate", "<", COMPARATORS["substitution_negative_far"]),
    "G7": ("deletion_negative_false_addition_rate", "<", COMPARATORS["deletion_negative_far"]),
    "G8": ("exact_event_f1", ">=", COMPARATORS["exact_event_f1"]),
}
HYPOTHESIS_BATCH_SIZE = 4096
OUTPUT_NAMES = (
    "r5_2b_population_accounting.json",
    "r5_2b_candidate_audit.json",
    "r5_2b_train_scores.jsonl",
    "r5_2b_continuous_metrics.json",
    "r5_2b_loso_fold_thresholds.json",
    "r5_2b_oof_predictions.jsonl",
    "r5_2b_binary_metrics.json",
    "r5_2b_relation_cohort_metrics.json",
    "r5_2b_event_metrics.json",
    "r5_2b_gate_results.json",
    "r5_2b_execution_protocol_audit.json",
    "r5_2b_final_status.json",
    "r5_2b_compute_report.json",
    "R5_2B_TRAIN_DEVELOPMENT_RESULT.md",
    "R5_2B_EXECUTION_MANIFEST.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


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
        actual_hash = sha256(artifact)
        actual_size = artifact.stat().st_size
        if actual_hash != entry["sha256"] or actual_size != int(entry["byte_size"]):
            failures.append({
                "path": entry["relative_path"], "reason": "identity_mismatch",
                "expected_sha256": entry["sha256"], "actual_sha256": actual_hash,
                "expected_size": int(entry["byte_size"]), "actual_size": actual_size,
            })
    return {
        "artifact_count": len(manifest["artifacts"]),
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def verify_identities() -> dict[str, Any]:
    actual = {name: sha256(path) for name, path in IDENTITY_PATHS.items()}
    mismatch = {
        name: {"expected": EXPECTED_SHA[name], "actual": actual[name]}
        for name in EXPECTED_SHA if actual[name] != EXPECTED_SHA[name]
    }
    if mismatch:
        contract_names = {"contract", "preregistration", "contract_manifest", "static_manifest", "static_run", "r5_2b_scorer"}
        status = "R5_2B_EXECUTION_BLOCKED_IDENTITY" if set(mismatch) & contract_names else "R5_2B_EXECUTION_BLOCKED_SOURCE_IDENTITY"
        raise RuntimeError(f"{status}: {mismatch}")
    contract_audit = verify_manifest(CONTRACT_MANIFEST)
    static_audit = verify_manifest(STATIC_MANIFEST)
    if contract_audit["status"] != "PASS" or static_audit["status"] != "PASS":
        raise RuntimeError(f"R5_2B_EXECUTION_BLOCKED_IDENTITY: contract={contract_audit} static={static_audit}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if int(checkpoint["blank_index"]) != 40 or tuple(checkpoint["vocabulary"]) != PHONE_VOCAB:
        raise RuntimeError("R5_2B_EXECUTION_BLOCKED_SOURCE_IDENTITY: checkpoint vocabulary mismatch")
    del checkpoint
    return {
        "status": "PASS", "expected_sha256": EXPECTED_SHA, "actual_sha256": actual,
        "contract_manifest_audit": contract_audit, "static_manifest_audit": static_audit,
    }


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2.0 * precision * recall, precision + recall)
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": precision, "recall": recall, "f1": f1}


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    tp = int(np.sum(truth & predicted)); fp = int(np.sum((~truth) & predicted))
    fn = int(np.sum(truth & (~predicted))); tn = int(np.sum((~truth) & (~predicted)))
    positive = prf(tp, fp, fn); negative = prf(tn, fn, fp)
    return {
        "words": int(truth.size), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": ratio(tp + tn, truth.size),
        "balanced_accuracy": (positive["recall"] + negative["recall"]) / 2.0,
        "binary_macro_f1": (positive["f1"] + negative["f1"]) / 2.0,
        "addition_precision": positive["precision"], "addition_recall": positive["recall"],
        "addition_f1": positive["f1"], "nonaddition_f1": negative["f1"],
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def validate_population(words: list[dict[str, Any]], accounting: dict[str, Any]) -> None:
    actual = {
        "words": accounting["total_words"], "positive_words": accounting["positive_words"],
        "negative_words": accounting["negative_words"], "source_events": accounting["source_addition_events"],
        "runtime_events": accounting["runtime_addition_events"],
        "multiple_addition_words": accounting["multiple_addition_words"],
        "mixed_substitution_addition_words": accounting["mixed_substitution_addition_words"],
        "mixed_deletion_addition_words": accounting["mixed_deletion_addition_words"],
        "without_manual_word": accounting["exclusions"]["without_manual_word"],
        "without_expected_sequence": accounting["exclusions"]["without_expected_sequence"],
        "unresolved_cooccurring_events": accounting["exclusions"]["unresolved_cooccurring_evidence_events"],
    }
    if actual != EXPECTED_COUNTS or len(words) != EXPECTED_COUNTS["words"]:
        accounting["status"] = "R5_2B_EXECUTION_BLOCKED_ROW_ACCOUNTING"
        accounting["expected"] = EXPECTED_COUNTS
        accounting["actual"] = actual
    else:
        accounting["status"] = "PASS"


def materialize_and_infer(words: list[dict[str, Any]], device: torch.device) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any]]:
    features, feature_report = r4b.materialize_features(words, device, "r5_2b_train_only")
    expected_lengths = [r4b.encoder_steps(feature.shape[-1]) for feature in features]
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = r4c2.WordBiGRUCTCModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    raw_logits: list[torch.Tensor | None] = [None] * len(words)
    log_probs: list[torch.Tensor | None] = [None] * len(words)
    lengths = [feature.shape[-1] for feature in features]
    ordered = sorted(range(len(words)), key=lambda index: (lengths[index], index))
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(ordered), 8):
            indexes = ordered[start:start + 8]
            maximum = max(lengths[index] for index in indexes)
            batch = torch.zeros((len(indexes), 1, r4b.N_MELS, maximum), dtype=torch.float32)
            frames = []
            for position, index in enumerate(indexes):
                feature = features[index]
                batch[position, 0, :, :feature.shape[-1]] = feature
                frames.append(feature.shape[-1])
            logits, output_lengths = model(batch.to(device, non_blocking=True), torch.tensor(frames, dtype=torch.long, device=device))
            values = logits.detach().cpu()
            probabilities = torch.log_softmax(values, dim=-1)
            for position, index in enumerate(indexes):
                steps = int(output_lengths[position].item())
                if steps != expected_lengths[index]:
                    raise RuntimeError("R5_2B_EXECUTION_TECHNICAL_FAILURE_IMPLEMENTATION: output length mismatch")
                raw_logits[index] = values[position, :steps].contiguous()
                log_probs[index] = probabilities[position, :steps].contiguous()
            if start == 0 or start + len(indexes) == len(ordered) or (start // 8) % 250 == 0:
                print(f"r5_2b_train_inference={min(start + len(indexes), len(ordered))}/{len(ordered)}", flush=True)
    if any(value is None for value in raw_logits) or any(value is None for value in log_probs):
        raise RuntimeError("R5_2B_EXECUTION_TECHNICAL_FAILURE_IMPLEMENTATION: incomplete inference")
    return (
        [value for value in raw_logits if value is not None],
        [value for value in log_probs if value is not None],
        {"feature_report": feature_report, "inference_seconds": time.perf_counter() - started},
    )


def candidate_audit(words: list[dict[str, Any]], logits: list[torch.Tensor]) -> dict[str, Any]:
    families = ("KEEP", "INSERT", "SUB", "DELETE")
    totals = Counter(); alignable = Counter(); impossible = Counter()
    by_speaker: dict[str, dict[str, Counter[str]]] = {
        speaker: {family: Counter() for family in families} for speaker in TRAIN
    }
    keep_impossible_details = []
    empty_delete_count = one_phone_words = words_no_insert = words_no_sub = words_affected = 0
    for word, evidence in zip(words, logits):
        expected = word["expected_ids"]
        steps = int(evidence.shape[0])
        word["encoder_steps"] = steps
        constructed = {
            "KEEP": [scorer.construct_keep(expected)],
            "INSERT": scorer.enumerate_insert_candidates(expected),
            "SUB": scorer.enumerate_sub_candidates(expected),
            "DELETE": scorer.enumerate_delete_candidates(expected),
        }
        affected = False
        word["candidate_counts"] = {}
        word["alignable_counts"] = {}
        word["impossible_counts"] = {}
        for family in families:
            candidates = constructed[family]
            family_alignable = sum(scorer.is_alignable(item.target, steps) for item in candidates)
            family_impossible = len(candidates) - family_alignable
            word["candidate_counts"][family] = len(candidates)
            word["alignable_counts"][family] = family_alignable
            word["impossible_counts"][family] = family_impossible
            totals[family] += len(candidates); alignable[family] += family_alignable; impossible[family] += family_impossible
            counter = by_speaker[word["speaker_id"]][family]
            counter["total"] += len(candidates); counter["alignable"] += family_alignable; counter["impossible"] += family_impossible
            affected = affected or family_impossible > 0
        keep_target = tuple(expected)
        word["keep_minimum_steps"] = scorer.minimum_ctc_steps(keep_target)
        if word["impossible_counts"]["KEEP"]:
            keep_impossible_details.append({
                "source_identity": word["word_id"], "speaker": word["speaker_id"],
                "expected_length": len(expected), "encoder_T": steps,
                "minimum_steps": word["keep_minimum_steps"],
            })
        one_phone_words += int(len(expected) == 1)
        empty_delete_count += sum(not item.target for item in constructed["DELETE"])
        words_no_insert += int(word["alignable_counts"]["INSERT"] == 0)
        words_no_sub += int(word["alignable_counts"]["SUB"] == 0)
        words_affected += int(affected)
    total_by_family = {family: int(totals[family]) for family in families}
    expected_total = sum(80 * len(word["expected_ids"]) + 41 for word in words)
    if int(sum(totals.values())) != expected_total:
        raise RuntimeError(
            f"R5_2B_EXECUTION_TECHNICAL_FAILURE_IMPLEMENTATION: "
            f"candidate total {sum(totals.values())}!={expected_total}"
        )
    return {
        "status": "PASS" if not keep_impossible_details else "R5_2B_EXECUTION_BLOCKED_KEEP_ALIGNABILITY",
        "words": len(words), "keep_impossible": len(keep_impossible_details),
        "keep_impossible_details": keep_impossible_details,
        "candidate_totals_by_family": total_by_family,
        "alignable_by_family": {family: int(alignable[family]) for family in families},
        "impossible_by_family": {family: int(impossible[family]) for family in families},
        "total_candidates": int(sum(totals.values())),
        "total_alignable": int(sum(alignable.values())), "total_impossible": int(sum(impossible.values())),
        "empty_delete_count": int(empty_delete_count), "one_phone_words": int(one_phone_words),
        "words_with_no_finite_insert": int(words_no_insert), "words_with_no_finite_sub": int(words_no_sub),
        "words_affected_by_impossible_candidates": int(words_affected),
        "by_speaker": {
            speaker: {family: dict(counter) for family, counter in values.items()}
            for speaker, values in by_speaker.items()
        },
        "candidate_count_identity": "80N+41", "expected_total_from_identity": int(expected_total),
    }


def batch_ctc_scores(log_probs: list[torch.Tensor], items: list[tuple[int, tuple[int, ...]]], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    if any(not target for _word_index, target in items):
        raise RuntimeError("R5_2B_EXECUTION_TECHNICAL_FAILURE_EMPTY_TARGET: empty target entered CTCLoss batch")
    maximum = max(log_probs[word_index].shape[0] for word_index, _target in items)
    acoustic = torch.zeros((maximum, len(items), 41), dtype=torch.float32)
    input_lengths = []; target_lengths = []; flat_targets = []
    for column, (word_index, target) in enumerate(items):
        evidence = log_probs[word_index]
        if not scorer.is_alignable(target, evidence.shape[0]):
            raise RuntimeError("R5_2B_EXECUTION_TECHNICAL_FAILURE_IMPLEMENTATION: impossible target entered CTCLoss batch")
        acoustic[:evidence.shape[0], column] = evidence
        input_lengths.append(evidence.shape[0]); target_lengths.append(len(target)); flat_targets.extend(target)
    with torch.no_grad():
        nll = torch.nn.CTCLoss(blank=40, reduction="none", zero_infinity=True)(
            acoustic.to(device), torch.tensor(flat_targets, dtype=torch.long, device=device),
            torch.tensor(input_lengths, dtype=torch.long, device=device),
            torch.tensor(target_lengths, dtype=torch.long, device=device),
        ).detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(nll).all():
        raise RuntimeError("R5_2B_EXECUTION_TECHNICAL_FAILURE_NUMERICAL: alignable non-finite loss")
    raw = -nll; target = raw / np.asarray(target_lengths, dtype=np.float64)
    if not np.isfinite(target).all():
        raise RuntimeError("R5_2B_EXECUTION_TECHNICAL_FAILURE_NUMERICAL: derived non-finite score")
    return raw, target


def verify_batch_adapter(device: torch.device) -> dict[str, Any]:
    logits = ((torch.arange(4 * 41, dtype=torch.float32).reshape(4, 41).remainder(17) - 8.0) / 7.0).to(device)
    frozen = scorer.score_target(logits, [1, 2], 4)
    log_probs = [torch.log_softmax(logits, dim=-1).detach().cpu()]
    raw, target = batch_ctc_scores(log_probs, [(0, (1, 2))], device)
    if float(raw[0]) != frozen.raw_score or float(target[0]) != frozen.target_score:
        raise RuntimeError(f"R5_2B_EXECUTION_TECHNICAL_FAILURE_IMPLEMENTATION: batch adapter mismatch {raw[0]}/{target[0]} vs {frozen.raw_score}/{frozen.target_score}")
    empty_frozen = scorer.score_target(logits, [], 4)
    empty_batch = float(log_probs[0][:, 40].sum().item())
    if empty_batch != empty_frozen.target_score:
        raise RuntimeError(f"R5_2B_EXECUTION_TECHNICAL_FAILURE_EMPTY_TARGET: {empty_batch}!={empty_frozen.target_score}")
    return {
        "status": "PASS", "nonempty_raw_score": float(raw[0]),
        "nonempty_target_score": float(target[0]), "empty_target_score": empty_batch,
    }


def make_hypothesis(target: tuple[int, ...], steps: int, target_score: float) -> scorer.HypothesisScore:
    alignable = scorer.is_alignable(target, steps)
    if not alignable:
        return scorer.HypothesisScore(target, steps, scorer.minimum_ctc_steps(target), False, float("-inf"), float("-inf"), False, False)
    raw = target_score * max(len(target), 1)
    return scorer.HypothesisScore(target, steps, scorer.minimum_ctc_steps(target), True, raw, target_score, bool(target), not target)


def score_train(words: list[dict[str, Any]], log_probs: list[torch.Tensor], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    family_candidates = []
    values: list[dict[str, np.ndarray]] = []
    for word in words:
        expected = word["expected_ids"]
        candidates = {
            "KEEP": [scorer.construct_keep(expected)],
            "INSERT": scorer.enumerate_insert_candidates(expected),
            "SUB": scorer.enumerate_sub_candidates(expected),
            "DELETE": scorer.enumerate_delete_candidates(expected),
        }
        family_candidates.append(candidates)
        values.append({family: np.full(len(items), -np.inf, dtype=np.float64) for family, items in candidates.items()})

    order = sorted(range(len(words)), key=lambda index: (log_probs[index].shape[0], index))
    batch_items: list[tuple[int, str, int, tuple[int, ...]]] = []
    processed_nonempty = processed_empty = 0
    started = time.perf_counter()

    def process(items: list[tuple[int, str, int, tuple[int, ...]]]) -> None:
        nonlocal processed_nonempty
        _raw, target_scores = batch_ctc_scores(log_probs, [(item[0], item[3]) for item in items], device)
        for item, target_score in zip(items, target_scores):
            word_index, family, candidate_index, _target = item
            values[word_index][family][candidate_index] = target_score
        processed_nonempty += len(items)
        if processed_nonempty % (HYPOTHESIS_BATCH_SIZE * 25) < len(items):
            print(f"r5_2b_alignable_nonempty_scored={processed_nonempty}", flush=True)

    for word_index in order:
        steps = log_probs[word_index].shape[0]
        group = []
        for family in ("KEEP", "INSERT", "SUB", "DELETE"):
            for candidate_index, candidate in enumerate(family_candidates[word_index][family]):
                if not candidate.target:
                    explicit = float(log_probs[word_index][:steps, 40].sum().item())
                    if not math.isfinite(explicit):
                        raise RuntimeError("R5_2B_EXECUTION_TECHNICAL_FAILURE_EMPTY_TARGET: non-finite all-blank score")
                    values[word_index][family][candidate_index] = explicit
                    processed_empty += 1
                elif scorer.is_alignable(candidate.target, steps):
                    group.append((word_index, family, candidate_index, candidate.target))
        if batch_items and len(batch_items) + len(group) > HYPOTHESIS_BATCH_SIZE:
            process(batch_items); batch_items = []
        batch_items.extend(group)
    if batch_items:
        process(batch_items)

    output = []
    for word_index, word in enumerate(words):
        steps = int(log_probs[word_index].shape[0])
        scored_families: dict[str, list[scorer.ScoredCandidate]] = {}
        for family in ("KEEP", "INSERT", "SUB", "DELETE"):
            scored_families[family] = [
                scorer.ScoredCandidate(candidate, make_hypothesis(candidate.target, steps, float(values[word_index][family][index])))
                for index, candidate in enumerate(family_candidates[word_index][family])
            ]
        keep = scored_families["KEEP"][0]
        if not keep.hypothesis.alignable or not math.isfinite(keep.hypothesis.target_score):
            raise RuntimeError("R5_2B_EXECUTION_BLOCKED_KEEP_ALIGNABILITY")
        best_insert = scorer.select_best_insert(scored_families["INSERT"])
        best_sub = scorer.select_best_sub(scored_families["SUB"])
        best_delete = scorer.select_best_delete(scored_families["DELETE"])
        best_nonaddition = scorer.select_best_nonaddition(keep, best_sub, best_delete)
        competition = scorer.compute_relation_competition_score(best_insert, best_nonaddition)
        output.append({
            "keep": keep, "best_insert": best_insert, "best_sub": best_sub,
            "best_delete": best_delete, "best_nonaddition": best_nonaddition,
            "relation_competition_score": competition,
        })
    expected_nonempty = sum(
        word["alignable_counts"][family]
        for word in words for family in ("KEEP", "INSERT", "SUB", "DELETE")
    ) - processed_empty
    if processed_nonempty != expected_nonempty:
        raise RuntimeError(f"R5_2B_EXECUTION_TECHNICAL_FAILURE_IMPLEMENTATION: {processed_nonempty}!={expected_nonempty}")
    return output, {
        "alignable_nonempty_hypotheses_scored": processed_nonempty,
        "empty_delete_hypotheses_scored": processed_empty,
        "batch_size": HYPOTHESIS_BATCH_SIZE, "seconds": time.perf_counter() - started,
    }


def distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "count": int(array.size), "negative_infinity_count": int(np.isneginf(array).sum()),
        "finite_count": int(finite.size),
        "minimum": float(np.min(finite)) if finite.size else None,
        "q25": float(np.quantile(finite, 0.25)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "q75": float(np.quantile(finite, 0.75)) if finite.size else None,
        "maximum": float(np.max(finite)) if finite.size else None,
    }


def threshold_selection(scores: np.ndarray, truth: np.ndarray, correct_only: np.ndarray) -> tuple[float, dict[str, Any]]:
    candidates = scorer.threshold_candidates(scores)
    finite = scores[np.isfinite(scores)]
    unique = np.unique(finite)
    finite_mask = np.isfinite(scores)
    tp = int(np.sum(truth & finite_mask)); fp = int(np.sum((~truth) & finite_mask))
    fn = int(np.sum(truth & (~finite_mask))); tn = int(np.sum((~truth) & (~finite_mask)))
    correct_support = int(correct_only.sum()); correct_positive = int(np.sum(correct_only & finite_mask))
    best_record: Mapping[str, float] | None = None; best_metrics = None

    def consider(threshold: float) -> None:
        nonlocal best_record, best_metrics
        positive = prf(tp, fp, fn); negative = prf(tn, fn, fp)
        macro = (positive["f1"] + negative["f1"]) / 2.0
        record = {
            "binary_macro_f1": macro, "addition_f1": positive["f1"],
            "correct_only_far": ratio(correct_positive, correct_support), "threshold": float(threshold),
        }
        if best_record is None or scorer.threshold_choice_key(record) > scorer.threshold_choice_key(best_record):
            best_record = record
            best_metrics = {
                "binary_macro_f1": macro, "addition_f1": positive["f1"],
                "addition_precision": positive["precision"],
                "correct_only_false_addition_rate": record["correct_only_far"],
                "confusion_matrix": [[tn, fp], [fn, tp]],
            }

    consider(float(candidates[0]))
    for value in unique:
        consider(float(value))
        indexes = np.flatnonzero(scores == value)
        positive_removed = int(truth[indexes].sum()); negative_removed = int(indexes.size - positive_removed)
        tp -= positive_removed; fn += positive_removed; fp -= negative_removed; tn += negative_removed
        correct_positive -= int(correct_only[indexes].sum())
    consider(float(candidates[-1]))
    if best_record is None or best_metrics is None:
        raise RuntimeError("R5_2B_EXECUTION_TECHNICAL_FAILURE_IMPLEMENTATION: threshold selection empty")
    return float(best_record["threshold"]), {
        "candidate_count": int(candidates.size), "finite_unique_scores": int(unique.size),
        "lower_edge": float(candidates[0]), "upper_edge": float(candidates[-1]),
        "selected_threshold": float(best_record["threshold"]), "selected_metrics": best_metrics,
    }


def run_loso(words: list[dict[str, Any]], scores: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    speakers = np.asarray([word["speaker_id"] for word in words])
    truth = np.asarray([word["is_addition"] for word in words], dtype=bool)
    correct = np.asarray([word["correct_only"] for word in words], dtype=bool)
    predictions = np.zeros(len(words), dtype=bool); folds = []
    for heldout in TRAIN:
        calibration = speakers != heldout; evaluation = speakers == heldout
        threshold, trace = threshold_selection(scores[calibration], truth[calibration], correct[calibration])
        predictions[evaluation] = scorer.predict_addition(scores[evaluation], threshold)
        folds.append({
            "heldout_speaker": heldout, "calibration_words": int(calibration.sum()),
            "heldout_words": int(evaluation.sum()), **trace,
            "heldout_metrics": binary_metrics(truth[evaluation], predictions[evaluation]),
        })
    return predictions, folds


def false_rate(words: list[dict[str, Any]], predictions: np.ndarray, key: str) -> dict[str, Any]:
    mask = np.asarray([word[key] for word in words], dtype=bool)
    count = int(np.sum(mask & predictions))
    return {"support": int(mask.sum()), "predicted_addition": count, "false_addition_rate": ratio(count, int(mask.sum()))}


def event_metrics(words: list[dict[str, Any]], predictions: np.ndarray, scored: list[dict[str, Any]]) -> dict[str, Any]:
    true_all = Counter(); pred_all = Counter(); positions = ("BEFORE_FIRST", "BETWEEN", "AFTER_FINAL")
    true_pos = {name: Counter() for name in positions}; pred_pos = {name: Counter() for name in positions}
    for index, word in enumerate(words):
        for event in word["true_events"]:
            identity = (word["word_id"], event["boundary"], event["phone_index"])
            true_all[identity] += 1; true_pos[event["position"]][identity] += 1
        best = scored[index]["best_insert"]
        if predictions[index] and best.exists:
            candidate = best.candidate
            identity = (word["word_id"], int(candidate.boundary), int(candidate.phone_index))
            pred_all[identity] += 1
            n = len(word["expected_ids"])
            position = "BEFORE_FIRST" if candidate.boundary == 0 else ("AFTER_FINAL" if candidate.boundary == n else "BETWEEN")
            pred_pos[position][identity] += 1

    def metric(truth: Counter[Any], predicted: Counter[Any]) -> dict[str, Any]:
        tp = int(sum((truth & predicted).values()))
        return {**prf(tp, int(sum(predicted.values())) - tp, int(sum(truth.values())) - tp),
                "true_events": int(sum(truth.values())), "predicted_events": int(sum(predicted.values()))}
    return {
        "exact_event": metric(true_all, pred_all),
        "by_position": {name: metric(true_pos[name], pred_pos[name]) for name in positions},
        "matching": "Counter intersection on (word_id,boundary,phone_index)",
        "multiple_addition_limitation": "one BEST_INSERT maximum per word; additional true events remain false negatives",
    }


def serialize_score(value: float, prefix: str) -> dict[str, Any]:
    payload = scorer.serialize_extended_score(value)
    return {f"{prefix}_value": payload["score_value"], f"{prefix}_is_neg_inf": payload["score_is_neg_inf"]}


def candidate_metadata(best: scorer.BestCandidateResult, family: str) -> dict[str, Any]:
    candidate = best.candidate
    payload = {
        f"best_{family}_exists": bool(best.exists),
        f"best_{family}_score_value": None if not best.exists else float(best.score),
        f"best_{family}_score_is_neg_inf": not best.exists,
    }
    if family == "insert":
        payload.update({
            "best_insert_boundary": None if candidate is None else int(candidate.boundary),
            "best_insert_phone_index": None if candidate is None else int(candidate.phone_index),
            "best_insert_phone": None if candidate is None else PHONE_VOCAB[int(candidate.phone_index)],
        })
    elif family == "sub":
        payload.update({
            "best_sub_position": None if candidate is None else int(candidate.position),
            "best_sub_replacement_phone_index": None if candidate is None else int(candidate.phone_index),
            "best_sub_replacement_phone": None if candidate is None else PHONE_VOCAB[int(candidate.phone_index)],
        })
    elif family == "delete":
        payload.update({
            "best_delete_position": None if candidate is None else int(candidate.position),
            "best_delete_target_empty": None if candidate is None else not bool(candidate.target),
        })
    return payload


def build_report(accounting: dict[str, Any], audit: dict[str, Any], continuous: dict[str, Any], folds: list[dict[str, Any]], binary: dict[str, Any], cohorts: dict[str, Any], events: dict[str, Any], gates: dict[str, Any], status: str) -> str:
    thresholds = [item["selected_threshold"] for item in folds]
    lines = [
        "# R5-2B Frozen TRAIN Development Result", "", f"Status: `{status}`", "",
        "## Population", "",
        f"- Words: {accounting['total_words']:,} ({accounting['positive_words']:,} positive; {accounting['negative_words']:,} negative)",
        f"- Source/runtime addition events: {accounting['source_addition_events']:,}/{accounting['runtime_addition_events']:,}", "",
        "## Candidate audit", "",
        f"- KEEP impossible: {audit['keep_impossible']}",
        f"- Candidates KEEP/INSERT/SUB/DELETE: {audit['candidate_totals_by_family']['KEEP']:,}/{audit['candidate_totals_by_family']['INSERT']:,}/{audit['candidate_totals_by_family']['SUB']:,}/{audit['candidate_totals_by_family']['DELETE']:,}",
        f"- Impossible KEEP/INSERT/SUB/DELETE: {audit['impossible_by_family']['KEEP']:,}/{audit['impossible_by_family']['INSERT']:,}/{audit['impossible_by_family']['SUB']:,}/{audit['impossible_by_family']['DELETE']:,}",
        f"- Empty DELETE / one-phone words: {audit['empty_delete_count']:,}/{audit['one_phone_words']:,}", "",
        "## Continuous scoring", "",
        f"- Addition vs non-addition AUC: {continuous['addition_vs_all_nonaddition_roc_auc']:.9f}",
        f"- Addition vs correct-only AUC: {continuous['addition_vs_correct_only_roc_auc']:.9f}",
        f"- Addition vs substitution-only AUC: {continuous['addition_vs_substitution_only_roc_auc']:.9f}",
        f"- Addition vs deletion-only AUC: {continuous['addition_vs_deletion_only_roc_auc']:.9f}", "",
        "## LOSO decision", "",
        f"- Thresholds: {', '.join(format(value, '.17g') for value in thresholds)}",
        f"- TP/FP/FN/TN: {binary['TP']}/{binary['FP']}/{binary['FN']}/{binary['TN']}",
        f"- Binary Macro-F1: {binary['binary_macro_f1']:.9f}",
        f"- Addition P/R/F1: {binary['addition_precision']:.9f}/{binary['addition_recall']:.9f}/{binary['addition_f1']:.9f}", "",
        "## False-addition mechanism", "",
        f"- Correct-only FAR: {cohorts['false_addition_rates']['correct_only']['false_addition_rate']:.9f}",
        f"- Substitution-negative FAR: {cohorts['false_addition_rates']['substitution_negative']['false_addition_rate']:.9f}",
        f"- Deletion-negative FAR: {cohorts['false_addition_rates']['deletion_negative']['false_addition_rate']:.9f}", "",
        "## Event localization", "",
        f"- Exact-event P/R/F1: {events['exact_event']['precision']:.9f}/{events['exact_event']['recall']:.9f}/{events['exact_event']['f1']:.9f}",
        "- Multiple-addition words remain included; one BEST_INSERT cannot recover every event.", "",
        "## Frozen gates", "",
    ]
    for name in GATES:
        gate = gates["gates"][name]
        lines.append(f"- {name}: **{gate['result']}** ({gate['value']:.12g} {gate['operator']} {gate['threshold']:.12g})")
    lines.extend(["", f"Passed: {gates['passed_count']}/8", "", f"Robust threshold: `{gates['robust_theta']['status']}`", ""])
    return "\n".join(lines)


def execute() -> None:
    started = time.perf_counter(); started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for name in OUTPUT_NAMES:
        if (EXPERIMENT_DIR / name).exists():
            raise RuntimeError(f"Refusing overwrite/rerun: {name}")
    identity = verify_identities()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter_check = verify_batch_adapter(device)

    details, source_scan = r5_1a_driver.scan_train_source()
    events_by_word, mapping = r5_1a_driver.map_train_additions(details, AUDIO_ROOT)
    words, accounting = r5_1a_driver.build_words(AUDIO_ROOT, events_by_word)
    accounting["source_scan"] = source_scan; accounting["addition_mapping"] = mapping
    accounting["population_reused_from"] = "frozen R5-1A TRAIN population"
    validate_population(words, accounting)
    write_json(EXPERIMENT_DIR / "r5_2b_population_accounting.json", accounting)
    if accounting["status"] != "PASS":
        raise RuntimeError(accounting["status"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    raw_logits, log_probs, inference_report = materialize_and_infer(words, device)
    audit = candidate_audit(words, raw_logits)
    write_json(EXPERIMENT_DIR / "r5_2b_candidate_audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError(audit["status"])
    scored, scoring_report = score_train(words, log_probs, device)
    scores = np.asarray([item["relation_competition_score"] for item in scored], dtype=np.float64)
    truth = np.asarray([word["is_addition"] for word in words], dtype=bool)
    correct_only = np.asarray([word["correct_only"] for word in words], dtype=bool)
    substitution_only = np.asarray([
        (not word["is_addition"]) and int(word["substitution"]) > 0 and int(word["deletion"]) == 0 for word in words
    ], dtype=bool)
    deletion_only = np.asarray([
        (not word["is_addition"]) and int(word["deletion"]) > 0 and int(word["substitution"]) == 0 for word in words
    ], dtype=bool)
    cohort_masks = {
        "positive": truth,
        "correct_only": correct_only,
        "substitution_negative": np.asarray([word["substitution_negative"] for word in words], dtype=bool),
        "deletion_negative": np.asarray([word["deletion_negative"] for word in words], dtype=bool),
        "substitution_only_negative": substitution_only,
        "deletion_only_negative": deletion_only,
    }
    speaker_values = np.asarray([word["speaker_id"] for word in words])
    continuous = {
        "addition_vs_all_nonaddition_roc_auc": extended_helpers.extended_real_roc_auc(truth, scores),
        "addition_vs_correct_only_roc_auc": extended_helpers.extended_real_roc_auc(truth[truth | correct_only], scores[truth | correct_only]),
        "addition_vs_substitution_only_roc_auc": extended_helpers.extended_real_roc_auc(truth[truth | substitution_only], scores[truth | substitution_only]),
        "addition_vs_deletion_only_roc_auc": extended_helpers.extended_real_roc_auc(truth[truth | deletion_only], scores[truth | deletion_only]),
        "auc_implementation": "frozen R5-1A extended-real Mann-Whitney average-rank",
        "score_distributions_by_speaker": {speaker: distribution(scores[speaker_values == speaker]) for speaker in TRAIN},
        "score_distributions_by_cohort": {name: distribution(scores[mask]) for name, mask in cohort_masks.items()},
        "descriptive_only": ["addition_vs_substitution_only_roc_auc", "addition_vs_deletion_only_roc_auc", "distributions"],
    }
    write_json(EXPERIMENT_DIR / "r5_2b_continuous_metrics.json", continuous)

    predictions, folds = run_loso(words, scores)
    write_json(EXPERIMENT_DIR / "r5_2b_loso_fold_thresholds.json", {
        "speaker_order": list(TRAIN), "folds": folds,
        "thresholds_in_speaker_order": [item["selected_threshold"] for item in folds],
    })
    binary = binary_metrics(truth, predictions)
    far = {
        "correct_only": false_rate(words, predictions, "correct_only"),
        "substitution_negative": false_rate(words, predictions, "substitution_negative"),
        "deletion_negative": false_rate(words, predictions, "deletion_negative"),
    }
    binary["false_addition_rates"] = far
    write_json(EXPERIMENT_DIR / "r5_2b_binary_metrics.json", binary)

    winner_by_cohort = {}
    for name, mask in cohort_masks.items():
        winner_by_cohort[name] = dict(Counter(
            scored[index]["best_nonaddition"].family for index in np.flatnonzero(mask)
        ))
    comparator_deltas = {
        "correct_only_far": far["correct_only"]["false_addition_rate"] - COMPARATORS["correct_only_far"],
        "substitution_negative_far": far["substitution_negative"]["false_addition_rate"] - COMPARATORS["substitution_negative_far"],
        "deletion_negative_far": far["deletion_negative"]["false_addition_rate"] - COMPARATORS["deletion_negative_far"],
    }
    relation_cohorts = {
        "false_addition_rates": far, "frozen_r5_1a_comparators": COMPARATORS,
        "r5_2b_minus_r5_1a_far_deltas": comparator_deltas,
        "best_nonaddition_winning_family_by_cohort": winner_by_cohort,
        "diagnostic_only": True,
    }
    write_json(EXPERIMENT_DIR / "r5_2b_relation_cohort_metrics.json", relation_cohorts)
    events = event_metrics(words, predictions, scored)
    write_json(EXPERIMENT_DIR / "r5_2b_event_metrics.json", events)

    threshold_by_speaker = {item["heldout_speaker"]: item["selected_threshold"] for item in folds}
    score_rows = []; oof_rows = []
    for index, (word, result) in enumerate(zip(words, scored)):
        best_non = result["best_nonaddition"]
        base = {
            "source_identity": word["word_id"], "speaker": word["speaker_id"], "word": word["word"],
            "expected_sequence": word["expected"], "expected_sequence_ids": word["expected_ids"],
            "addition_label": bool(word["is_addition"]), "ground_truth_addition_event_count": len(word["true_events"]),
            "ground_truth_events": word["true_events"],
            "relation_cohorts": {
                "multiple_addition": len(word["true_events"]) > 1,
                "substitution_addition": bool(word["is_addition"] and int(word["substitution"]) > 0),
                "deletion_addition": bool(word["is_addition"] and int(word["deletion"]) > 0),
                "correct_only": bool(word["correct_only"]),
                "substitution_negative": bool(word["substitution_negative"]),
                "deletion_negative": bool(word["deletion_negative"]),
            },
            "encoder_T": word["encoder_steps"], "keep_minimum_ctc_steps": word["keep_minimum_steps"],
            **serialize_score(result["keep"].hypothesis.target_score, "keep_target_score"),
            "insert_candidate_count": word["candidate_counts"]["INSERT"],
            "insert_alignable_count": word["alignable_counts"]["INSERT"],
            "insert_impossible_count": word["impossible_counts"]["INSERT"],
            **candidate_metadata(result["best_insert"], "insert"),
            "sub_candidate_count": word["candidate_counts"]["SUB"],
            "sub_alignable_count": word["alignable_counts"]["SUB"],
            "sub_impossible_count": word["impossible_counts"]["SUB"],
            **candidate_metadata(result["best_sub"], "sub"),
            "delete_candidate_count": word["candidate_counts"]["DELETE"],
            "delete_alignable_count": word["alignable_counts"]["DELETE"],
            "delete_impossible_count": word["impossible_counts"]["DELETE"],
            **candidate_metadata(result["best_delete"], "delete"),
            "best_nonaddition_family": best_non.family,
            **serialize_score(best_non.score, "best_nonaddition_score"),
            **serialize_score(result["relation_competition_score"], "relation_competition_score_C"),
        }
        score_rows.append(base)
        oof_rows.append({
            **base, "heldout_speaker": word["speaker_id"],
            "fold_threshold": threshold_by_speaker[word["speaker_id"]],
            "predicted_addition": bool(predictions[index]),
        })
    write_jsonl(EXPERIMENT_DIR / "r5_2b_train_scores.jsonl", score_rows)
    write_jsonl(EXPERIMENT_DIR / "r5_2b_oof_predictions.jsonl", oof_rows)

    values = {
        "addition_vs_all_nonaddition_roc_auc": continuous["addition_vs_all_nonaddition_roc_auc"],
        "addition_vs_correct_only_roc_auc": continuous["addition_vs_correct_only_roc_auc"],
        "oof_binary_macro_f1": binary["binary_macro_f1"], "oof_addition_f1": binary["addition_f1"],
        "correct_only_false_addition_rate": far["correct_only"]["false_addition_rate"],
        "substitution_negative_false_addition_rate": far["substitution_negative"]["false_addition_rate"],
        "deletion_negative_false_addition_rate": far["deletion_negative"]["false_addition_rate"],
        "exact_event_f1": events["exact_event"]["f1"],
    }
    gate_records = {}
    for name, (metric, operator, threshold) in GATES.items():
        value = float(values[metric])
        passed = value >= threshold if operator == ">=" else value > threshold if operator == ">" else value <= threshold if operator == "<=" else value < threshold
        gate_records[name] = {
            "metric": metric, "value": value, "operator": operator, "threshold": threshold,
            "result": "PASS" if passed else "FAIL", "full_precision": True,
        }
    passed_count = sum(item["result"] == "PASS" for item in gate_records.values()); all_pass = passed_count == 8
    thresholds = np.asarray([item["selected_threshold"] for item in folds], dtype=np.float64)
    robust = {
        "status": "AUTHORIZED" if all_pass else "R5_2_ROBUST_THETA_NOT_AUTHORIZED",
        "thresholds_speaker_order": thresholds.tolist(), "sorted_thresholds": np.sort(thresholds).tolist(),
        "value": float(np.median(thresholds)) if all_pass else None,
    }
    gates = {"gates": gate_records, "passed_count": passed_count, "total": 8, "all_pass": all_pass, "robust_theta": robust}
    write_json(EXPERIMENT_DIR / "r5_2b_gate_results.json", gates)
    status = "R5_2_RELATION_COMPETITION_DEVELOPMENT_PASS" if all_pass else "R5_2_RELATION_COMPETITION_DEVELOPMENT_NOT_CONFIRMED"
    protocol = {
        "execution_count": 1, "neural_training": False, "optimizer_created": False,
        "train_inference": True, "validation_paths_resolved": False, "validation_accessed": False,
        "validation_rerun": False, "validation_r5_2_scores_created": False,
        "test_paths_resolved": False, "test_audio_accessed": False, "test_inference_run": False,
        "test_performance_consumed": False, "threshold_rule_modified": False,
        "frozen_scorer_modified": False, "r5_1_modified": False, "checkpoint_selection": False,
        "alternative_score_family": False, "score_formula_modified": False,
    }
    write_json(EXPERIMENT_DIR / "r5_2b_execution_protocol_audit.json", protocol)
    final = {
        "status": status, "gates_passed": passed_count, "gates_total": 8,
        "robust_theta": robust, "protocol": protocol,
        "mechanism": {
            "substitution_far_gate": gate_records["G6"]["result"],
            "deletion_far_gate": gate_records["G7"]["result"],
        },
        "multiple_addition_limitation": events["multiple_addition_limitation"],
    }
    write_json(EXPERIMENT_DIR / "r5_2b_final_status.json", final)
    compute = {
        "started_at": started_at, "ended_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "batch_adapter_equivalence": adapter_check, "inference": inference_report,
        "scoring": scoring_report, "total_seconds_before_hashing": time.perf_counter() - started,
    }
    write_json(EXPERIMENT_DIR / "r5_2b_compute_report.json", compute)
    report = build_report(accounting, audit, continuous, folds, binary, relation_cohorts, events, gates, status)
    (EXPERIMENT_DIR / "R5_2B_TRAIN_DEVELOPMENT_RESULT.md").write_text(report, encoding="utf-8")

    artifact_names = ["r5_2b_train_execution_driver.py"] + [name for name in OUTPUT_NAMES if name != "R5_2B_EXECUTION_MANIFEST.json"]
    entries = [
        {"relative_path": name, "byte_size": (EXPERIMENT_DIR / name).stat().st_size, "sha256": sha256(EXPERIMENT_DIR / name)}
        for name in artifact_names
    ]
    failures = [entry["relative_path"] for entry in entries if sha256(EXPERIMENT_DIR / entry["relative_path"]) != entry["sha256"]]
    manifest = {
        "manifest_type": "additive R5-2B frozen TRAIN execution manifest", "self_excluded": True,
        "hash_algorithm": "SHA-256", "artifact_count": len(entries),
        "hash_audit": "HASH_AUDIT_PASS" if not failures else "HASH_AUDIT_FAIL", "failures": failures,
        "artifacts": entries, "preserved_contract_sha256": EXPECTED_SHA["contract"],
        "preserved_static_manifest_sha256": EXPECTED_SHA["static_manifest"],
        "preserved_scorer_sha256": EXPECTED_SHA["r5_2b_scorer"],
    }
    write_json(EXPERIMENT_DIR / "R5_2B_EXECUTION_MANIFEST.json", manifest)
    print(json.dumps({
        "status": status, "gates_passed": passed_count, "manifest_sha256": sha256(EXPERIMENT_DIR / "R5_2B_EXECUTION_MANIFEST.json"),
        "artifact_count": len(entries), "hash_audit": manifest["hash_audit"],
    }, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        raise SystemExit("Frozen R5-2B driver requires --execute")
    execute()


if __name__ == "__main__":
    main()
