"""R4-4D0 TRAIN-only CTC sequence-hypothesis feasibility audit.

This audit freezes the R4-4C2 acoustic model and compares exact CTC sequence
likelihoods for KEEP, DELETE, and all 39 single-phone substitutions.  It never
scores validation or test data and performs no optimization or training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_r4_3a_word_sequence_design_audit as r43a  # noqa: E402
import run_r4_4b_ctc_sequence as r4b  # noqa: E402
import run_r4_4c2_bigru_ctc_sequence as frozen_model  # noqa: E402


REPO_ROOT = r4b.REPO_ROOT
R4C2_DIR = REPO_ROOT / "ai-training/experiments/r4_4c2_bigru_ctc_seed42"
CHECKPOINT = R4C2_DIR / "R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt"
R4C2_MANIFEST = R4C2_DIR / "artifact_hashes.json"
V4_PATH = r4b.V4_PATH
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r4_4d0_ctc_hypothesis_feasibility"
EXPECTED = {
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
    "checkpoint": "F54C9C2361AE78E1C37353AEB338A2DE6722C0B1EC4F885F6B52688CE9E88085",
    "r4c2_manifest": "BE82690C366049E659F5A872256BF23616566E18D37A7AFB8A1E7169629F1DB9",
    "model_config": "AC05095522AB0276C00BA98CCEC603163DE536EB92C58CFA5B37753C411330AB",
    "feature_config": "BB7E638F61628A117B081D7B30BE3346C4774DE808B9595D10F5020554A11809",
    "training_config": "ADF59A88E169A12544E2772BC0F36C29906E29D3B6984A98133745E0FF47A90B",
    "selected_checkpoint": "F47999815CCF406B1B64F0E43A5A640BD4BC70B3935ECE930F346B7195F10F6A",
    "frozen_model_source": "E6639A8BBBB65C41105C151A881736125280E7F66298801DC21D8BD3205695ED",
}
TRAIN_WORDS = 16_259
TRAIN_RELATIONS = {"correct": 48_893, "substitution": 5_867, "deletion": 1_544}
INFERENCE_BATCH_SIZE = 128
HYPOTHESIS_BATCH_SIZE = 4096
BLANK = 40
SPEAKER_MIN_DELETIONS = 30
PHONE_MIN_DELETIONS = 20
PHONE_MIN_NONDELETIONS = 100
LENGTH_BIAS_ABS_SPEARMAN = 0.30
FAMILIES = ("RAW", "TARGET", "TIME")
SIMPLICITY = {"RAW": 0, "TIME": 1, "TARGET": 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def finite(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)), "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)), "p90": float(np.percentile(array, 90)),
        "min": float(array.min()), "max": float(array.max()),
    }


def auc_metrics(labels: Iterable[int], scores: Iterable[float]) -> dict[str, Any]:
    y = np.asarray(list(labels), dtype=np.int8)
    s = np.asarray(list(scores), dtype=np.float64)
    if not y.size or len(np.unique(y)) != 2 or not np.isfinite(s).all():
        return {"count": int(y.size), "positive": int(y.sum()), "negative": int(y.size - y.sum()),
                "roc_auc": None, "pr_auc": None}
    return {
        "count": int(y.size), "positive": int(y.sum()), "negative": int(y.size - y.sum()),
        "roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
    }


def correlation(left: Iterable[float], right: Iterable[float]) -> dict[str, Any]:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"count": int(x.size), "pearson": None, "spearman": None}
    return {
        "count": int(x.size), "pearson": finite(np.corrcoef(x, y)[0, 1]),
        "spearman": finite(spearmanr(x, y).statistic),
    }


def verify_sources() -> dict[str, Any]:
    paths = {
        "v4": V4_PATH, "checkpoint": CHECKPOINT, "r4c2_manifest": R4C2_MANIFEST,
        "model_config": R4C2_DIR / "model_config.json",
        "feature_config": R4C2_DIR / "feature_config.json",
        "training_config": R4C2_DIR / "training_config.json",
        "selected_checkpoint": R4C2_DIR / "selected_checkpoint.json",
        "frozen_model_source": REPO_ROOT / "ai-training/scripts/run_r4_4c2_bigru_ctc_sequence.py",
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    mismatches = {name: {"expected": EXPECTED[name], "actual": value}
                  for name, value in actual.items() if value != EXPECTED[name]}
    manifest = json.loads(R4C2_MANIFEST.read_text(encoding="utf-8"))
    internal = {}
    for filename in ("model_config.json", "feature_config.json", "training_config.json", "selected_checkpoint.json"):
        actual_digest = sha256(R4C2_DIR / filename)
        recorded = manifest["files"].get(filename)
        internal[filename] = {"recorded": recorded, "actual": actual_digest, "match": recorded == actual_digest}
    if mismatches or not all(item["match"] for item in internal.values()):
        raise RuntimeError(f"R4_4D0_SOURCE_VERIFICATION_FAIL: expected={mismatches}, internal={internal}")
    return {"status": "PASS", "expected": EXPECTED, "actual": actual,
            "manifest_internal_identity": internal}


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def load_train_words(audio_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # This is the key data firewall: reconstruction is limited to TRAIN speakers.
    r43a.AUDIT_SPEAKERS = frozenset(r4b.TRAIN_SPEAKERS)
    records, reconstruction = r43a.build_word_records(audio_root)
    words = [word for word in records if word["usable"] and word["split"] == "train"]
    if len(words) != TRAIN_WORDS:
        raise RuntimeError(f"TRAIN word population mismatch: {len(words)}")
    if any(word["speaker_id"] not in r4b.TRAIN_SPEAKERS for word in words):
        raise RuntimeError("Non-TRAIN speaker entered R4-4D0")
    for word in words:
        word["target_ids"] = [r4b.PHONE_TO_ID[phone] for phone in word["observed"]]
        word["expected_ids"] = [r4b.PHONE_TO_ID[phone] for phone in word["expected"]]
    relations = Counter(row["relation"] for word in words for row in word["clean_rows"])
    actual_relations = {name: int(relations[name]) for name in TRAIN_RELATIONS}
    if actual_relations != TRAIN_RELATIONS:
        raise RuntimeError(f"TRAIN relation mismatch: {actual_relations}")
    return words, {"usable_words": len(words), "relation_counts": actual_relations,
                   "expected_positions": sum(len(word["expected_ids"]) for word in words),
                   "empty_manual_targets": sum(not word["target_ids"] for word in words),
                   "speakers": sorted({word["speaker_id"] for word in words}),
                   "reconstruction": reconstruction}


def inference(model: torch.nn.Module, words: list[dict[str, Any]], features: list[torch.Tensor],
              device: torch.device) -> tuple[list[torch.Tensor], list[list[int]], dict[str, Any]]:
    model.eval()
    log_probs: list[torch.Tensor | None] = [None] * len(words)
    decoded: list[list[int] | None] = [None] * len(words)
    lengths = [feature.shape[-1] for feature in features]
    ordered = sorted(range(len(words)), key=lambda index: (lengths[index], index))
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(ordered), INFERENCE_BATCH_SIZE):
            indexes = ordered[start:start + INFERENCE_BATCH_SIZE]
            maximum = max(lengths[index] for index in indexes)
            batch = torch.zeros((len(indexes), 1, r4b.N_MELS, maximum), dtype=torch.float32)
            frame_lengths = []
            for position, index in enumerate(indexes):
                feature = features[index]
                batch[position, 0, :, :feature.shape[-1]] = feature
                frame_lengths.append(feature.shape[-1])
            frame_tensor = torch.tensor(frame_lengths, dtype=torch.long, device=device)
            logits, output_lengths = model(batch.to(device), frame_tensor)
            batch_log_probs = torch.log_softmax(logits, dim=-1).detach().cpu()
            batch_decoded = r4b.greedy_decode(logits, output_lengths)
            for position, index in enumerate(indexes):
                steps = int(output_lengths[position].item())
                log_probs[index] = batch_log_probs[position, :steps].contiguous()
                decoded[index] = batch_decoded[position]
            if start == 0 or start + len(indexes) == len(ordered) or (start // INFERENCE_BATCH_SIZE) % 25 == 0:
                print(f"train_inference={min(start + len(indexes), len(ordered))}/{len(ordered)}", flush=True)
    assert all(item is not None for item in log_probs) and all(item is not None for item in decoded)
    return ([item for item in log_probs if item is not None], [item for item in decoded if item is not None], {
        "words": len(words), "seconds": time.perf_counter() - started,
        "encoder_steps": distribution(item.shape[0] for item in log_probs if item is not None),
    })


def ctc_minimum(target: list[int]) -> int:
    return len(target) + sum(left == right for left, right in zip(target, target[1:]))


def candidate_stream(words: list[dict[str, Any]], word_order: list[int]) -> Iterator[tuple[int, int, str, int, list[int]]]:
    for word_index in word_order:
        expected = words[word_index]["expected_ids"]
        yield word_index, -1, "KEEP", -1, expected
        for position, expected_phone in enumerate(expected):
            yield word_index, position, "DELETE", -1, expected[:position] + expected[position + 1:]
            for phone in range(40):
                if phone != expected_phone:
                    target = expected.copy(); target[position] = phone
                    yield word_index, position, "SUB", phone, target


def score_hypotheses(words: list[dict[str, Any]], log_probs: list[torch.Tensor], device: torch.device,
                     offsets: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    positions = sum(len(word["expected_ids"]) for word in words)
    keep_scores = np.full(len(words), -np.inf, dtype=np.float64)
    delete_scores = np.full(positions, -np.inf, dtype=np.float64)
    best_sub_scores = np.full(positions, -np.inf, dtype=np.float64)
    best_sub_phone = np.full(positions, -1, dtype=np.int16)
    unalignable = Counter(); nonfinite = 0; processed = 0
    criterion = torch.nn.CTCLoss(blank=BLANK, reduction="none", zero_infinity=True)
    word_order = sorted(range(len(words)), key=lambda index: (log_probs[index].shape[0], index))
    stream = candidate_stream(words, word_order)
    started = time.perf_counter()

    def process(batch_items: list[tuple[int, int, str, int, list[int]]]) -> None:
        nonlocal processed, nonfinite
        maximum = max(log_probs[item[0]].shape[0] for item in batch_items)
        acoustic = torch.zeros((maximum, len(batch_items), 41), dtype=torch.float32)
        input_lengths: list[int] = []
        target_lengths: list[int] = []
        flat_targets: list[int] = []
        for column, (word_index, _position, kind, _phone, target) in enumerate(batch_items):
            evidence = log_probs[word_index]
            acoustic[:evidence.shape[0], column] = evidence
            input_lengths.append(evidence.shape[0]); target_lengths.append(len(target)); flat_targets.extend(target)
            if ctc_minimum(target) > evidence.shape[0]:
                unalignable[kind] += 1
        with torch.no_grad():
            nll = criterion(
                acoustic.to(device),
                torch.tensor(flat_targets, dtype=torch.long, device=device),
                torch.tensor(input_lengths, dtype=torch.long, device=device),
                torch.tensor(target_lengths, dtype=torch.long, device=device),
            ).detach().cpu().numpy()
        scores = -nll.astype(np.float64)
        nonfinite += int((~np.isfinite(scores)).sum())
        for item, score in zip(batch_items, scores):
            word_index, position, kind, phone, _target = item
            if kind == "KEEP":
                keep_scores[word_index] = score
            else:
                global_position = offsets[word_index] + position
                if kind == "DELETE":
                    delete_scores[global_position] = score
                elif score > best_sub_scores[global_position] or (
                    score == best_sub_scores[global_position] and phone < best_sub_phone[global_position]
                ):
                    best_sub_scores[global_position] = score; best_sub_phone[global_position] = phone
        processed += len(batch_items)
        if processed % (HYPOTHESIS_BATCH_SIZE * 25) < len(batch_items):
            print(f"hypotheses_scored={processed}", flush=True)

    batch: list[tuple[int, int, str, int, list[int]]] = []
    for item in stream:
        batch.append(item)
        if len(batch) == HYPOTHESIS_BATCH_SIZE:
            process(batch); batch = []
    if batch:
        process(batch)
    expected_count = len(words) + positions * 40
    if processed != expected_count:
        raise RuntimeError(f"Hypothesis count mismatch: {processed} != {expected_count}")
    if not (np.isfinite(keep_scores).all() and np.isfinite(delete_scores).all() and np.isfinite(best_sub_scores).all()):
        raise RuntimeError(f"Non-finite/missing scores: returned_nonfinite={nonfinite}")
    return keep_scores, delete_scores, best_sub_scores, best_sub_phone, {
        "total_hypotheses": processed, "average_hypotheses_per_word": processed / len(words),
        "keep_hypotheses": len(words), "delete_hypotheses": positions,
        "substitution_hypotheses": positions * 39, "batch_size": HYPOTHESIS_BATCH_SIZE,
        "seconds": time.perf_counter() - started, "theoretically_unalignable": dict(unalignable),
        "nonfinite_returned_scores": nonfinite,
    }


def greedy_relations(expected: list[int], decoded: list[int]) -> list[str]:
    output = ["" for _ in expected]
    for operation in r4b.sequence_alignment(expected, decoded):
        index = operation["reference_index"]
        if index is None:
            continue
        output[index] = {
            "MATCH": "correct", "SUBSTITUTION": "substitution", "DELETE_FROM_EXPECTED": "deletion"
        }[operation["operation"]]
    if any(not value for value in output):
        raise RuntimeError("Incomplete greedy relation alignment")
    return output


def family_scores(family: str, keep: float, delete: float, sub: float, n: int, steps: int) -> dict[str, float]:
    if family == "RAW":
        k, d, s = keep, delete, sub
    elif family == "TARGET":
        k, d, s = keep / max(n, 1), delete / max(n - 1, 1), sub / max(n, 1)
    elif family == "TIME":
        k, d, s = keep / steps, delete / steps, sub / steps
    else:
        raise ValueError(family)
    return {
        "keep": k, "delete": d, "best_sub": s,
        "del_vs_keep": d - k, "del_vs_sub": d - s,
        "del_vs_best_nondelete": d - max(k, s), "sub_vs_keep": s - k,
    }


def build_position_rows(words: list[dict[str, Any]], log_probs: list[torch.Tensor], decoded: list[list[int]],
                        keep: np.ndarray, delete: np.ndarray, sub: np.ndarray, sub_phone: np.ndarray,
                        offsets: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for word_index, word in enumerate(words):
        greedy = greedy_relations(word["expected_ids"], decoded[word_index])
        n = len(word["expected_ids"]); steps = log_probs[word_index].shape[0]
        duration = float(word["mfa_end"] - word["mfa_start"])
        all_correct = word["deletion"] == 0 and word["substitution"] == 0
        for position, source in enumerate(word["clean_rows"]):
            global_position = offsets[word_index] + position
            if n == 1:
                location = "single"
            elif position == 0:
                location = "initial"
            elif position == n - 1:
                location = "final"
            else:
                location = "medial"
            row = {
                "word_index": word_index, "word_id": word["word_id"], "speaker": word["speaker_id"],
                "utterance": word["utterance_id"], "word": word["word"], "position": position,
                "expected_phone": source["expected"], "true_relation": source["relation"],
                "true_observed_phone": source["observed"], "word_expected_length": n,
                "word_duration": duration, "encoder_steps": steps, "position_group": location,
                "word_all_correct": all_correct, "word_substitutions": int(word["substitution"]),
                "word_deletions": int(word["deletion"]), "greedy_relation": greedy[position],
                "best_sub_phone": r4b.PHONE_VOCAB[int(sub_phone[global_position])],
            }
            for family in FAMILIES:
                values = family_scores(family, float(keep[word_index]), float(delete[global_position]),
                                       float(sub[global_position]), n, steps)
                for name, value in values.items():
                    row[f"{family.lower()}_{name}"] = value
            rows.append(row)
    return rows


def relation_distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for family in FAMILIES:
        key = f"{family.lower()}_del_vs_best_nondelete"
        output[family] = {relation: distribution(row[key] for row in rows if row["true_relation"] == relation)
                          for relation in ("correct", "substitution", "deletion")}
    return output


def primary_auc(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    nondel = {}; del_sub = {}
    for family in FAMILIES:
        key = f"{family.lower()}_del_vs_best_nondelete"
        nondel[family] = auc_metrics((row["true_relation"] == "deletion" for row in rows),
                                     (row[key] for row in rows))
        subset = [row for row in rows if row["true_relation"] in {"deletion", "substitution"}]
        del_sub[family] = auc_metrics((row["true_relation"] == "deletion" for row in subset),
                                      (row[key] for row in subset))
        correct_subset = [row for row in rows if row["true_relation"] in {"deletion", "correct"}]
        nondel[family]["deletion_vs_correct"] = auc_metrics(
            (row["true_relation"] == "deletion" for row in correct_subset),
            (row[key] for row in correct_subset),
        )
    return nondel, del_sub


def best_sub_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    subset = [row for row in rows if row["true_relation"] == "substitution"]
    correct = sum(row["best_sub_phone"] == row["true_observed_phone"] for row in subset)
    return {family: {"support": len(subset), "correct": correct, "top1_accuracy": correct / len(subset)}
            for family in FAMILIES}


def clean_control(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [row for row in rows if row["word_all_correct"]]
    return {
        family: {
            "words": len({row["word_id"] for row in clean}), "positions": len(clean),
            "distribution": distribution(row[f"{family.lower()}_del_vs_best_nondelete"] for row in clean),
            "deletion_preferred_at_zero": sum(row[f"{family.lower()}_del_vs_best_nondelete"] > 0 for row in clean),
            "deletion_preferred_fraction": sum(row[f"{family.lower()}_del_vs_best_nondelete"] > 0 for row in clean) / len(clean),
        } for family in FAMILIES
    }


def greedy_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicted = [row for row in rows if row["greedy_relation"] == "deletion"]
    return {
        family: {
            "true_deletion": distribution(row[f"{family.lower()}_del_vs_best_nondelete"]
                                           for row in predicted if row["true_relation"] == "deletion"),
            "false_deletion": distribution(row[f"{family.lower()}_del_vs_best_nondelete"]
                                            for row in predicted if row["true_relation"] != "deletion"),
            "true_vs_false_auc": auc_metrics((row["true_relation"] == "deletion" for row in predicted),
                                               (row[f"{family.lower()}_del_vs_best_nondelete"] for row in predicted)),
        } for family in FAMILIES
    }


def group_auc(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return auc_metrics((row["true_relation"] == "deletion" for row in rows), (row[field] for row in rows))


def phone_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"support_rule": {"deletion_min": PHONE_MIN_DELETIONS,
                                                "nondeletion_min": PHONE_MIN_NONDELETIONS}, "phones": {}}
    for phone in r4b.PHONE_VOCAB:
        subset = [row for row in rows if row["expected_phone"] == phone]
        deletion = sum(row["true_relation"] == "deletion" for row in subset)
        nondeletion = len(subset) - deletion
        item = {"support": len(subset), "deletion_support": deletion, "nondeletion_support": nondeletion,
                "sufficient": deletion >= PHONE_MIN_DELETIONS and nondeletion >= PHONE_MIN_NONDELETIONS}
        for family in FAMILIES:
            item[family] = group_auc(subset, f"{family.lower()}_del_vs_best_nondelete")
        output["phones"][phone] = item
    groups = {
        "D_T_R_L": [row for row in rows if row["expected_phone"] in {"D", "T", "R", "L"}],
        "OTHER": [row for row in rows if row["expected_phone"] not in {"D", "T", "R", "L"}],
    }
    output["groups"] = {name: {family: group_auc(subset, f"{family.lower()}_del_vs_best_nondelete")
                                for family in FAMILIES} for name, subset in groups.items()}
    return output


def speaker_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for speaker in sorted(r4b.TRAIN_SPEAKERS):
        subset = [row for row in rows if row["speaker"] == speaker]
        deletion = sum(row["true_relation"] == "deletion" for row in subset)
        output[speaker] = {"positions": len(subset), "deletion_support": deletion,
                           "sufficient": deletion >= SPEAKER_MIN_DELETIONS}
        for family in FAMILIES:
            output[speaker][family] = group_auc(subset, f"{family.lower()}_del_vs_best_nondelete")
    return output


def position_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for name in ("initial", "medial", "final", "single"):
        subset = [row for row in rows if row["position_group"] == name]
        output[name] = {"positions": len(subset),
                        "deletion_support": sum(row["true_relation"] == "deletion" for row in subset)}
        for family in FAMILIES:
            output[name][family] = group_auc(subset, f"{family.lower()}_del_vs_best_nondelete")
    return output


def multi_error_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    predicates = {
        "pure_deletion": lambda row: row["word_deletions"] > 0 and row["word_substitutions"] == 0,
        "substitution_plus_deletion": lambda row: row["word_deletions"] > 0 and row["word_substitutions"] > 0,
        "multiple_deletion": lambda row: row["word_deletions"] > 1,
        "single_deletion_no_substitution": lambda row: row["word_deletions"] == 1 and row["word_substitutions"] == 0,
    }
    output = {}
    for name, predicate in predicates.items():
        subset = [row for row in rows if predicate(row)]
        output[name] = {"words": len({row["word_id"] for row in subset}), "positions": len(subset),
                        "deletion_support": sum(row["true_relation"] == "deletion" for row in subset)}
        for family in FAMILIES:
            key = f"{family.lower()}_del_vs_best_nondelete"
            output[name][family] = {
                "all_positions_auc": group_auc(subset, key),
                "true_deletion_distribution": distribution(row[key] for row in subset
                                                             if row["true_relation"] == "deletion"),
            }
    return output


def select_family(nondel: dict[str, Any], del_sub: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    trace = []
    max_del_sub = max(del_sub[name]["roc_auc"] for name in FAMILIES)
    contenders = [name for name in FAMILIES if max_del_sub - del_sub[name]["roc_auc"] <= 0.01]
    trace.append({"rule": "highest deletion-vs-substitution ROC-AUC; within 0.01 advances",
                  "maximum": max_del_sub, "contenders": contenders})
    if len(contenders) > 1:
        best = max(nondel[name]["roc_auc"] for name in contenders)
        contenders = [name for name in contenders if abs(nondel[name]["roc_auc"] - best) <= 1e-12]
        trace.append({"rule": "higher deletion-vs-nondeletion ROC-AUC", "maximum": best,
                      "contenders": contenders})
    if len(contenders) > 1:
        best = max(nondel[name]["pr_auc"] for name in contenders)
        contenders = [name for name in contenders if abs(nondel[name]["pr_auc"] - best) <= 1e-12]
        trace.append({"rule": "higher deletion PR-AUC", "maximum": best, "contenders": contenders})
    selected = min(contenders, key=lambda name: SIMPLICITY[name])
    trace.append({"rule": "simplicity RAW > TIME > TARGET", "selected": selected})
    return selected, trace


def write_position_csv(rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys())
    with (EXPERIMENT_DIR / "train_position_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def create_future_design(selected: str, selection: dict[str, Any], source_verification: dict[str, Any]) -> None:
    design = {
        "status": "PREREGISTERED_NOT_RUN", "experiment": "R4-4D1",
        "frozen_acoustic_model": {"path": str(CHECKPOINT.relative_to(REPO_ROOT)).replace("\\", "/"),
                                  "sha256": EXPECTED["checkpoint"]},
        "score_family": selected,
        "sequence_hypotheses": "KEEP, DELETE, and 39 substitutions per expected position",
        "decision_score": "DELETE - max(KEEP, BEST_SUB) using exact CTC forward likelihood",
        "train_only_threshold": {
            "search": "all unique TRAIN score thresholds; score >= threshold predicts deletion",
            "constraints": ["TRAIN deletion recall >= 0.45", "TRAIN substitution false-deletion <= 0.25"],
            "selection": ["highest binary Macro-F1", "higher deletion F1", "higher deletion precision",
                          "higher threshold (fewer deletion predictions)"],
            "validation_influence": False,
        },
        "nondeletion_relation": "BEST_SUB > KEEP predicts substitution; otherwise correct",
        "locked_validation_evaluation": {
            "single_run": True, "threshold_frozen_before validation": True,
            "metrics": ["ROC-AUC", "PR-AUC", "binary Macro-F1", "balanced accuracy",
                        "deletion precision/recall/F1", "substitution false-deletion",
                        "3-relation Macro-F1", "speaker and phone diagnostics"],
        },
        "source_verification": source_verification["actual"], "train_audit_selection": selection,
        "test_policy": "R4 TEST CLOSED", "training": False,
    }
    write_json(EXPERIMENT_DIR / "r4_4d1_preregistered_score_design.json", design)


def report_markdown(summary: dict[str, Any]) -> str:
    selected = summary["selection"]["selected_family"]
    lines = [
        "# R4-4D0 CTC Sequence-Hypothesis Feasibility Audit", "",
        f"Final: **{summary['final_status']}**", "",
        "This was a TRAIN-only frozen-checkpoint audit. No neural training, validation candidate scoring, or R4 TEST access occurred.", "",
        "## Primary TRAIN results", "",
        "| Family | Del vs non-del ROC | PR | Del vs sub ROC | PR | Clean zero-pref |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        n = summary["auc_nondeletion"][family]; s = summary["auc_deletion_substitution"][family]
        clean = summary["clean_control"][family]["deletion_preferred_fraction"]
        lines.append(f"| {family} | {n['roc_auc']:.6f} | {n['pr_auc']:.6f} | {s['roc_auc']:.6f} | {s['pr_auc']:.6f} | {clean:.6f} |")
    lines += ["", f"Selected score family: **{selected}**", "",
              f"Raw length-bias verdict: **{summary['length_bias']['verdict']}**", "",
              "## Feasibility gates", ""]
    for name, value in summary["selection"]["gates"].items():
        lines.append(f"- {name}: {'PASS' if value else 'FAIL'}")
    lines += ["", "## Closure", "", "- VALIDATION candidate hypothesis scores: NO",
              "- VALIDATION thresholds/normalization selection: NO", "- R4 TEST accessed: NO",
              "- Neural training: NO", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("R4-4D0 runs only with --execute")
    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite {EXPERIMENT_DIR}")
    started_total = time.perf_counter()
    verification = verify_sources()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)
    score_contract = {
        "acoustic": "frozen R4-4C2 T x 41 log_softmax", "blank": 40,
        "ctc": {"implementation": "torch.nn.CTCLoss", "reduction": "none", "zero_infinity": True,
                "raw_log_score": "-CTCLoss(log_probs,H,input_length,target_length)"},
        "hypotheses": {"KEEP": "full expected sequence", "DELETE": "remove local expected phone",
                       "SUB": "replace local expected phone with each other canonical phone"},
        "families": {"RAW": "raw_log_score", "TARGET": "raw_log_score/max(target_length,1)",
                     "TIME": "raw_log_score/input_length"},
        "primary_score": "DELETE - max(KEEP,BEST_SUB)", "threshold_search": False,
        "selection": ["highest deletion-vs-substitution ROC-AUC", "tie within .01: higher deletion-vs-nondeletion ROC-AUC",
                      "tie: higher deletion PR-AUC", "tie: RAW then TIME then TARGET"],
        "support": {"speaker_min_deletions": SPEAKER_MIN_DELETIONS,
                    "phone_min_deletions": PHONE_MIN_DELETIONS, "phone_min_nondeletions": PHONE_MIN_NONDELETIONS},
        "raw_length_bias_rule": f"RAW_SCORE_LENGTH_BIASED if maximum absolute TRAIN all-correct Spearman association with expected length, word duration, or encoder T >= {LENGTH_BIAS_ABS_SPEARMAN}",
        "strong_broad_consistency": "at least 75% sufficient speakers AUC>0.5 and both D/T/R/L and other-phone ROC-AUC>=0.65",
        "data": "TRAIN speakers only", "validation_candidate_scoring": False, "r4_test": "CLOSED",
    }
    write_json(EXPERIMENT_DIR / "score_contract.json", score_contract)
    write_json(EXPERIMENT_DIR / "preflight.json", {
        "status": "PASS", "source_verification": verification, "git_commit": git_commit(),
        "python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "training_occurred": False, "validation_candidate_scores_calculated": False,
        "validation_thresholds_selected": False, "validation_used_to_choose_normalization": False,
        "test_paths_resolved": False, "test_audio_read": False, "test_posterior_computed": False,
        "test_hypothesis_scored": False,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    audio_root = r4b.r3.require_audio_root()
    words, dataset = load_train_words(audio_root)
    feature_start = time.perf_counter()
    features, feature_report = r4b.materialize_features(words, device, "r4_4d0_train")
    feature_seconds = time.perf_counter() - feature_start
    model = frozen_model.WordBiGRUCTCModel().to(device).eval()
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if frozen_model.parameter_report(model)["total"] != 198_761:
        raise RuntimeError("Frozen model parameter mismatch")
    log_probs, decoded, inference_report = inference(model, words, features, device)

    # Verify finite exact CTC scoring for a real single-phone DELETE -> empty target.
    single_index = next(index for index, word in enumerate(words) if len(word["expected_ids"]) == 1)
    evidence = log_probs[single_index].to(device).unsqueeze(1)
    empty_nll = torch.nn.CTCLoss(blank=BLANK, reduction="none", zero_infinity=True)(
        evidence, torch.empty(0, dtype=torch.long, device=device),
        torch.tensor([evidence.shape[0]], dtype=torch.long, device=device),
        torch.tensor([0], dtype=torch.long, device=device),
    )[0].item()
    empty_check = {"word_id": words[single_index]["word_id"], "encoder_steps": evidence.shape[0],
                   "raw_nll": empty_nll, "raw_log_score": -empty_nll,
                   "finite": bool(math.isfinite(empty_nll)), "status": "PASS" if math.isfinite(empty_nll) else "WARNING"}

    offsets = []; cursor = 0
    for word in words:
        offsets.append(cursor); cursor += len(word["expected_ids"])
    keep, delete, sub, sub_phone, compute = score_hypotheses(words, log_probs, device, offsets)
    rows = build_position_rows(words, log_probs, decoded, keep, delete, sub, sub_phone, offsets)
    if len(rows) != sum(TRAIN_RELATIONS.values()):
        raise RuntimeError("Position population mismatch")

    distributions = relation_distributions(rows)
    nondel_auc, del_sub_auc = primary_auc(rows)
    sub_accuracy = best_sub_accuracy(rows)
    clean = clean_control(rows)
    greedy = greedy_comparison(rows)
    phones = phone_metrics(rows); speakers = speaker_metrics(rows)
    positions = position_metrics(rows); multi = multi_error_metrics(rows)

    correct_rows = [row for row in rows if row["true_relation"] == "correct"]
    raw_delta = [row["raw_del_vs_keep"] for row in correct_rows]
    length_associations = {
        "expected_target_length": correlation(raw_delta, (row["word_expected_length"] for row in correct_rows)),
        "word_duration": correlation(raw_delta, (row["word_duration"] for row in correct_rows)),
        "encoder_steps": correlation(raw_delta, (row["encoder_steps"] for row in correct_rows)),
    }
    maximum_abs = max(abs(item["spearman"]) for item in length_associations.values() if item["spearman"] is not None)
    length_bias = {
        "delta": "RAW DELETE - KEEP", "relation_independent_control": "true correct positions",
        "associations": length_associations, "maximum_absolute_spearman": maximum_abs,
        "threshold": LENGTH_BIAS_ABS_SPEARMAN,
        "verdict": "RAW_SCORE_LENGTH_BIASED" if maximum_abs >= LENGTH_BIAS_ABS_SPEARMAN else "RAW_SCORE_NOT_STRONGLY_LENGTH_BIASED",
        "by_relation": {relation: distribution(row["raw_del_vs_keep"] for row in rows if row["true_relation"] == relation)
                        for relation in ("correct", "substitution", "deletion")},
    }

    selected, selection_trace = select_family(nondel_auc, del_sub_auc)
    sufficient_speakers = [item for item in speakers.values() if item["sufficient"]]
    consistent_speakers = sum(item[selected]["roc_auc"] is not None and item[selected]["roc_auc"] > 0.5
                              for item in sufficient_speakers)
    speaker_fraction = consistent_speakers / len(sufficient_speakers)
    group_auc_selected = {name: phones["groups"][name][selected]["roc_auc"] for name in ("D_T_R_L", "OTHER")}
    gates = {
        "deletion_vs_nondeletion_roc_at_least_0_65": nondel_auc[selected]["roc_auc"] >= 0.65,
        "deletion_vs_substitution_roc_at_least_0_60": del_sub_auc[selected]["roc_auc"] >= 0.60,
        "speaker_direction_consistent_at_least_75_percent": speaker_fraction >= 0.75,
        "clean_zero_threshold_deletion_preference_at_most_25_percent": clean[selected]["deletion_preferred_fraction"] <= 0.25,
    }
    feasible = all(gates.values())
    broad = speaker_fraction >= 0.75 and all(value is not None and value >= 0.65 for value in group_auc_selected.values())
    if feasible and nondel_auc[selected]["roc_auc"] >= 0.75 and del_sub_auc[selected]["roc_auc"] >= 0.70 and broad:
        final_status = "R4_4D0_CTC_HYPOTHESIS_SIGNAL_STRONG"
    elif feasible:
        final_status = "R4_4D0_CTC_HYPOTHESIS_SIGNAL_MODERATE"
    elif (nondel_auc[selected]["deletion_vs_correct"]["roc_auc"] is not None
          and nondel_auc[selected]["deletion_vs_correct"]["roc_auc"] >= 0.65
          and del_sub_auc[selected]["roc_auc"] < 0.60):
        final_status = "R4_4D0_CTC_HYPOTHESIS_MISMATCH_ONLY"
    else:
        final_status = "R4_4D0_CTC_HYPOTHESIS_SIGNAL_WEAK"

    selection = {
        "selected_family": selected, "selection_trace": selection_trace, "gates": gates,
        "feasible": feasible, "sufficient_speakers": len(sufficient_speakers),
        "consistent_speakers": consistent_speakers, "speaker_consistency_fraction": speaker_fraction,
        "group_roc_auc": group_auc_selected, "broad_consistency": broad,
        "strong_thresholds": {"deletion_vs_nondeletion_roc": 0.75, "deletion_vs_substitution_roc": 0.70},
    }

    compute.update({
        "feature_materialization_seconds": feature_seconds, "acoustic_inference": inference_report,
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
        "estimated_seconds_per_typical_word": compute["seconds"] / len(words),
        "total_seconds_before_artifact_hashing": time.perf_counter() - started_total,
    })
    write_json(EXPERIMENT_DIR / "length_bias_audit.json", length_bias)
    write_json(EXPERIMENT_DIR / "train_score_distributions.json", distributions)
    write_json(EXPERIMENT_DIR / "train_auc_metrics.json", {"deletion_vs_nondeletion": nondel_auc,
                                                              "best_sub_phone_accuracy": sub_accuracy,
                                                              "greedy_deletion_comparison": greedy})
    write_json(EXPERIMENT_DIR / "train_deletion_vs_substitution.json", del_sub_auc)
    write_json(EXPERIMENT_DIR / "train_clean_control.json", clean)
    write_json(EXPERIMENT_DIR / "train_phone_metrics.json", phones)
    write_json(EXPERIMENT_DIR / "train_speaker_metrics.json", speakers)
    write_json(EXPERIMENT_DIR / "train_position_metrics.json", positions)
    write_json(EXPERIMENT_DIR / "train_multi_error_metrics.json", multi)
    write_json(EXPERIMENT_DIR / "compute_report.json", {"dataset": dataset, "features": feature_report,
                                                          "empty_target_score": empty_check, **compute})
    write_json(EXPERIMENT_DIR / "selection_result.json", selection)
    write_position_csv(rows)
    final_payload = {
        "status": final_status, "selected_family": selected, "raw_length_bias": length_bias["verdict"],
        "training_occurred": False, "validation_candidate_scores_calculated": False,
        "validation_thresholds_selected": False, "validation_used_to_choose_normalization": False,
        "r4_test_accessed": False, "test_paths_resolved": False, "test_audio_read": False,
        "test_posterior_computed": False, "test_hypothesis_scored": False,
    }
    write_json(EXPERIMENT_DIR / "final_status.json", final_payload)
    summary = {"final_status": final_status, "selection": selection, "length_bias": length_bias,
               "auc_nondeletion": nondel_auc, "auc_deletion_substitution": del_sub_auc,
               "clean_control": clean}
    (EXPERIMENT_DIR / "r4_4d0_report.md").write_text(report_markdown(summary), encoding="utf-8")
    if feasible:
        create_future_design(selected, selection, verification)
    artifacts = {}
    for path in sorted(EXPERIMENT_DIR.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.json":
            artifacts[path.name] = sha256(path)
    write_json(EXPERIMENT_DIR / "artifact_hashes.json", {"algorithm": "SHA-256", "files": artifacts,
                                                           "note": "manifest excludes itself"})
    print(json.dumps({"status": final_status, "selected": selected, "gates": gates,
                      "nondel": nondel_auc[selected], "del_sub": del_sub_auc[selected],
                      "clean": clean[selected]["deletion_preferred_fraction"],
                      "manifest_sha": sha256(EXPERIMENT_DIR / "artifact_hashes.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
