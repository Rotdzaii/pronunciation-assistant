from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_r4_4b_ctc_sequence as r4b  # noqa: E402
import run_r4_4c2_bigru_ctc_sequence as frozen_model  # noqa: E402


REPO_ROOT = r4b.REPO_ROOT
PREREG_DIR = REPO_ROOT / "ai-training/experiments/r4_4c1_bigru_ctc_preregistration"
PREREG_JSON = PREREG_DIR / "r4_4c2_preregistered_design.json"
PREREG_MD = PREREG_DIR / "r4_4c2_preregistered_design.md"
FROZEN_MODEL_SOURCE = REPO_ROOT / "ai-training/scripts/run_r4_4c2_bigru_ctc_sequence.py"
V4_PATH = r4b.V4_PATH
MATCHED_CSV = r4b.MATCHED_CSV
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r4_4c2_bigru_ctc_seed42"
CHECKPOINT_NAME = "R4_4C2_bigru_ctc_phone_sequence_seed42_best_validation_per.pt"
EXPECTED_HASHES = {
    "preregistered_json": "3E9DFD5AED0F77B20A8E9D82CEDC06980A47B949A0E256A2AAE6C94E9420E5E0",
    "preregistered_markdown": "15825D6D50131B7865C86A0CD6904E559E93A89431F67AE4B4CEFA76F91C4B3C",
    "frozen_model_source": "E6639A8BBBB65C41105C151A881736125280E7F66298801DC21D8BD3205695ED",
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
    "matched_control": "D933F674743DA06CC8FAB425CEBF81D9C78505E1BDB4A90204DDB2E1A15B4798",
}
SEED = 42
EPOCHS = 36
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
GRAD_CLIP = 5.0
TIE_TOLERANCE = 1e-12
R4B = {
    "per": 0.602840,
    "binary_macro_f1": 0.476941,
    "deletion_precision": 0.062972,
    "deletion_recall": 0.483589,
    "deletion_f1": 0.111433,
    "correct_false_deletion": 0.256118,
    "substitution_false_deletion": 0.280781,
    "decoded_target_ratio": 0.7757,
    "shorter_fraction": 0.4885,
    "blank_occupancy": 0.8277,
    "blank_posterior_mean": 0.8019,
    "ctc_deletion_edits": 6199,
    "clean_word_false_deletion": 0.2416,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size), "min": float(array.min()), "mean": float(array.mean()),
        "median": float(np.median(array)), "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)), "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)), "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def verify_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "preregistered_json": PREREG_JSON, "preregistered_markdown": PREREG_MD,
        "frozen_model_source": FROZEN_MODEL_SOURCE, "v4": V4_PATH, "matched_control": MATCHED_CSV,
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    mismatch = {name: {"expected": EXPECTED_HASHES[name], "actual": value}
                for name, value in actual.items() if value != EXPECTED_HASHES[name]}
    if mismatch:
        raise RuntimeError(f"R4_4C2_FREEZE_VERIFICATION_FAIL: {mismatch}")
    return json.loads(PREREG_JSON.read_text(encoding="utf-8")), {
        "status": "PASS", "expected": EXPECTED_HASHES, "actual": actual,
    }


def set_determinism() -> dict[str, Any]:
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": SEED, "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_deterministic_warn_only": True, "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def collate(indexes: list[int], features: list[torch.Tensor], words: list[dict[str, Any]],
            device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(features[index].shape[-1] for index in indexes)
    batch = torch.zeros((len(indexes), 1, r4b.N_MELS, maximum), dtype=torch.float32)
    frame_lengths: list[int] = []
    target_lengths: list[int] = []
    target_values: list[int] = []
    for position, index in enumerate(indexes):
        feature = features[index]
        batch[position, 0, :, :feature.shape[-1]] = feature
        frame_lengths.append(feature.shape[-1])
        target = words[index]["target_ids"]
        target_lengths.append(len(target)); target_values.extend(target)
    return (
        batch.to(device, non_blocking=True),
        torch.tensor(target_values, dtype=torch.long, device=device),
        torch.tensor(frame_lengths, dtype=torch.long, device=device),
        torch.tensor(target_lengths, dtype=torch.long, device=device),
    )


def relation_outputs(words: list[dict[str, Any]], decoded_by_index: dict[int, list[int]],
                     word_stats: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    relation_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    total_edits = Counter(); target_total = 0; exact = 0; insertion_phones = Counter()
    for index, word in enumerate(words):
        decoded = decoded_by_index[index]; target = word["target_ids"]
        counts, acoustic_alignment = r4b.edit_counts(target, decoded)
        total_edits.update(counts); target_total += len(target); exact += decoded == target
        relation_alignment = r4b.sequence_alignment(word["expected_ids"], decoded)
        predicted_relations = ["" for _ in word["expected"]]
        predicted_observed = ["" for _ in word["expected"]]
        alignment_names = ["" for _ in word["expected"]]
        insertions: list[str] = []
        for operation in relation_alignment:
            name = operation["operation"]; expected_index = operation["reference_index"]
            decoded_index = operation["hypothesis_index"]
            if expected_index is None:
                phone = r4b.PHONE_VOCAB[decoded[decoded_index]]
                insertions.append(phone); insertion_phones[phone] += 1
            elif name == "MATCH":
                predicted_relations[expected_index] = "correct"
                predicted_observed[expected_index] = word["expected"][expected_index]
                alignment_names[expected_index] = name
            elif name == "SUBSTITUTION":
                predicted_relations[expected_index] = "substitution"
                predicted_observed[expected_index] = r4b.PHONE_VOCAB[decoded[decoded_index]]
                alignment_names[expected_index] = name
            elif name == "DELETE_FROM_EXPECTED":
                predicted_relations[expected_index] = "deletion"
                predicted_observed[expected_index] = "<SIL>"
                alignment_names[expected_index] = name
        for expected_index, source in enumerate(word["clean_rows"]):
            relation_rows.append({
                "word_index": index, "word_id": word["word_id"], "speaker_id": word["speaker_id"],
                "utterance_id": word["utterance_id"], "word": word["word"],
                "word_start": float(word["mfa_start"]), "word_end": float(word["mfa_end"]),
                "expected_phone": word["expected"][expected_index], "expected_phone_index": expected_index,
                "source_csv_row": int(source["source_index"]) + 2,
                "true_relation": source["relation"], "true_observed_phone": source["observed"],
                "predicted_relation": predicted_relations[expected_index],
                "predicted_observed_phone": predicted_observed[expected_index],
                "alignment_operation": alignment_names[expected_index],
                "decoded_sequence": " ".join(r4b.PHONE_VOCAB[token] for token in decoded),
            })
        stats = word_stats[index]
        details.append({
            "word_index": index, "word_id": word["word_id"], "speaker_id": word["speaker_id"],
            "utterance_id": word["utterance_id"], "word": word["word"],
            "word_start": float(word["mfa_start"]), "word_end": float(word["mfa_end"]),
            "expected": list(word["expected"]), "target": list(word["observed"]),
            "target_ids": list(target), "decoded": [r4b.PHONE_VOCAB[token] for token in decoded],
            "decoded_ids": list(decoded), "acoustic_edit_counts": counts,
            "acoustic_alignment": acoustic_alignment, "relation_alignment": relation_alignment,
            "ground_truth_relations": [row["relation"] for row in word["clean_rows"]],
            "predicted_relations": predicted_relations, "predicted_insertions": insertions,
            "contains_deletion": word["deletion"] > 0, "contains_substitution": word["substitution"] > 0,
            "empty_target": len(target) == 0,
            "word_per": counts["errors"] / len(target) if target else (0.0 if not decoded else None),
            **stats,
        })
    return relation_rows, details, {
        "target_phone_denominator": target_total,
        "edit_counts": {name: int(total_edits[name]) for name in ("substitution", "deletion", "insertion", "errors")},
        "phone_error_rate": total_edits["errors"] / target_total,
        "exact_decoded_words": exact, "exact_decoded_sequence_accuracy": exact / len(words),
        "predicted_insertion_phone_counts": dict(insertion_phones.most_common()),
    }


def evaluate(model: nn.Module, words: list[dict[str, Any]], features: list[torch.Tensor],
             criterion: nn.CTCLoss, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval(); decoded_by_index: dict[int, list[int]] = {}; word_stats: dict[int, dict[str, Any]] = {}
    loss_sum = 0.0; seen = 0; blank_posteriors: list[float] = []
    lengths = [feature.shape[-1] for feature in features]
    with torch.no_grad():
        for indexes in r4b.make_evaluation_batches(lengths):
            batch, targets, frame_lengths, target_lengths = collate(indexes, features, words, device)
            logits, output_lengths = model(batch, frame_lengths)
            if logits.shape[1] != r4b.encoder_steps(batch.shape[-1]):
                raise RuntimeError("R4_4C2_TEMPORAL_CONTRACT_FAIL")
            loss = criterion(logits.log_softmax(-1).transpose(0, 1), targets, output_lengths, target_lengths)
            loss_sum += float(loss.item()) * len(indexes); seen += len(indexes)
            decoded_batch = r4b.greedy_decode(logits, output_lengths)
            argmax = logits.argmax(dim=-1).detach().cpu()
            blank_probs = logits.softmax(dim=-1)[..., r4b.BLANK].detach().cpu()
            for position, (index, decoded) in enumerate(zip(indexes, decoded_batch)):
                length = int(output_lengths[position])
                raw = argmax[position, :length]
                probs = blank_probs[position, :length].numpy().astype(np.float64)
                blank_posteriors.extend(probs.tolist())
                decoded_by_index[index] = decoded
                word_stats[index] = {
                    "encoder_timesteps": length,
                    "blank_argmax_occupancy": float((raw == r4b.BLANK).float().mean().item()),
                    "blank_posterior_mean": float(probs.mean()),
                    "blank_posterior_median": float(np.median(probs)),
                }
    rows, details, sequence = relation_outputs(words, decoded_by_index, word_stats)
    binary = r4b.binary_metrics(rows); three = r4b.relation_metrics(rows)
    decoded_total = sum(len(detail["decoded_ids"]) for detail in details)
    shorter = sum(len(detail["decoded_ids"]) < len(detail["target_ids"]) for detail in details)
    metrics = {
        "loss": loss_sum / seen, **sequence, "binary": binary, "three_relation": three,
        "correct_false_deletion_rate": r4b.false_deletion_rate(rows, "correct"),
        "substitution_false_deletion_rate": r4b.false_deletion_rate(rows, "substitution"),
        "decoded_phone_total": decoded_total,
        "aggregate_decoded_target_ratio": decoded_total / sequence["target_phone_denominator"],
        "words_decoded_shorter": shorter, "words_decoded_shorter_fraction": shorter / len(words),
        "blank_argmax_occupancy_mean": float(np.mean([detail["blank_argmax_occupancy"] for detail in details])),
        "blank_posterior_mean": float(np.mean(blank_posteriors)),
        "blank_posterior_median": float(np.median(blank_posteriors)),
    }
    return metrics, rows, details


def train_epoch(model: nn.Module, words: list[dict[str, Any]], features: list[torch.Tensor],
                criterion: nn.CTCLoss, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int) -> dict[str, Any]:
    model.train(); lengths = [feature.shape[-1] for feature in features]
    batches = r4b.make_training_batches(lengths, epoch); loss_sum = 0.0; seen = 0
    for batch_number, indexes in enumerate(batches, 1):
        batch, targets, frame_lengths, target_lengths = collate(indexes, features, words, device)
        optimizer.zero_grad(set_to_none=True)
        logits, output_lengths = model(batch, frame_lengths)
        loss = criterion(logits.log_softmax(-1).transpose(0, 1), targets, output_lengths, target_lengths)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss epoch={epoch} batch={batch_number}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step()
        loss_sum += float(loss.item()) * len(indexes); seen += len(indexes)
        if batch_number % 250 == 0 or batch_number == len(batches):
            print(f"epoch={epoch} train_batch={batch_number}/{len(batches)} loss={loss_sum/seen:.6f}", flush=True)
    return {"ctc_loss": loss_sum / seen}


def save_history(history: list[dict[str, Any]]) -> None:
    write_json(EXPERIMENT_DIR / "epoch_metrics.json", history)
    fields = ["epoch", "train_loss", "validation_loss", "per", "exact_sequence_accuracy", "binary_macro_f1",
              "balanced_accuracy", "deletion_precision", "deletion_recall", "deletion_f1",
              "substitution_false_deletion", "three_relation_macro_f1", "decoded_target_ratio",
              "blank_argmax_occupancy", "epoch_seconds"]
    with (EXPERIMENT_DIR / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for record in history:
            val = record["validation"]
            writer.writerow({
                "epoch": record["epoch"], "train_loss": record["train"]["ctc_loss"],
                "validation_loss": val["loss"], "per": val["phone_error_rate"],
                "exact_sequence_accuracy": val["exact_decoded_sequence_accuracy"],
                "binary_macro_f1": val["binary"]["macro_f1"], "balanced_accuracy": val["binary"]["balanced_accuracy"],
                "deletion_precision": val["binary"]["deletion"]["precision"],
                "deletion_recall": val["binary"]["deletion"]["recall"],
                "deletion_f1": val["binary"]["deletion"]["f1"],
                "substitution_false_deletion": val["substitution_false_deletion_rate"],
                "three_relation_macro_f1": val["three_relation"]["macro_f1"],
                "decoded_target_ratio": val["aggregate_decoded_target_ratio"],
                "blank_argmax_occupancy": val["blank_argmax_occupancy_mean"],
                "epoch_seconds": record["epoch_seconds"],
            })


def matched_ids() -> set[int]:
    with MATCHED_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    identities = {int(row["source_csv_row"]) for row in rows}
    if len(rows) != 1434 or len(identities) != 1434:
        raise RuntimeError("Frozen matched-control identity failure")
    return identities


def write_exports(rows: list[dict[str, Any]], details: list[dict[str, Any]]) -> None:
    fields = ["speaker_id", "utterance_id", "word_id", "word", "word_start", "word_end", "expected_phone",
              "expected_phone_index", "source_csv_row", "true_relation", "true_observed_phone",
              "predicted_relation", "predicted_observed_phone", "decoded_sequence", "alignment_operation",
              "word_position", "is_matched_control_row"]
    with (EXPERIMENT_DIR / "validation_phone_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows: writer.writerow({field: row[field] for field in fields})
    with (EXPERIMENT_DIR / "validation_word_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for detail in details:
            payload = {key: value for key, value in detail.items() if key not in {"target_ids", "decoded_ids"}}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def under_generation(metrics: dict[str, Any], rows: list[dict[str, Any]], details: list[dict[str, Any]]) -> dict[str, Any]:
    correct_word_indexes = {detail["word_index"] for detail in details
                            if set(detail["ground_truth_relations"]) == {"correct"}}
    clean_rows = [row for row in rows if row["word_index"] in correct_word_indexes]
    clean_fd = sum(row["predicted_relation"] == "deletion" for row in clean_rows) / len(clean_rows)
    result = {
        "manual_target_total_phones": metrics["target_phone_denominator"],
        "decoded_total_phones": metrics["decoded_phone_total"],
        "aggregate_decoded_target_ratio": metrics["aggregate_decoded_target_ratio"],
        "words_decoded_shorter": metrics["words_decoded_shorter"],
        "words_decoded_shorter_fraction": metrics["words_decoded_shorter_fraction"],
        "blank_argmax_occupancy_mean": metrics["blank_argmax_occupancy_mean"],
        "blank_posterior_mean": metrics["blank_posterior_mean"],
        "blank_posterior_median": metrics["blank_posterior_median"],
        "ctc_target_phone_deletion_edit_count": metrics["edit_counts"]["deletion"],
        "clean_all_correct_words": len(correct_word_indexes), "clean_word_rows": len(clean_rows),
        "clean_word_false_expected_phone_deletion_rate": clean_fd,
    }
    result["r4_4b_reference"] = R4B
    result["deltas_current_minus_r4_4b"] = {
        "aggregate_decoded_target_ratio": result["aggregate_decoded_target_ratio"] - R4B["decoded_target_ratio"],
        "words_decoded_shorter_fraction": result["words_decoded_shorter_fraction"] - R4B["shorter_fraction"],
        "blank_argmax_occupancy_mean": result["blank_argmax_occupancy_mean"] - R4B["blank_occupancy"],
        "blank_posterior_mean": result["blank_posterior_mean"] - R4B["blank_posterior_mean"],
        "ctc_target_phone_deletion_edit_count": result["ctc_target_phone_deletion_edit_count"] - R4B["ctc_deletion_edits"],
        "clean_word_false_expected_phone_deletion_rate": clean_fd - R4B["clean_word_false_deletion"],
    }
    return result


def gate_and_classify(metrics: dict[str, Any], diagnostics: dict[str, Any], under: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    speaker_gate = {}
    for speaker, block in diagnostics["speakers"].items():
        support = block["binary"]["deletion"]["support"]; recall = block["binary"]["deletion"]["recall"]
        speaker_gate[speaker] = {"support": support, "recall": recall, "eligible": support >= 30,
                                 "pass": support < 30 or recall >= 0.25}
    gates = {
        "PER_le_0_55": metrics["phone_error_rate"] <= .55,
        "binary_macro_f1_ge_0_70": metrics["binary"]["macro_f1"] >= .70,
        "deletion_recall_ge_0_45": metrics["binary"]["deletion"]["recall"] >= .45,
        "deletion_f1_ge_0_40": metrics["binary"]["deletion"]["f1"] >= .40,
        "substitution_false_deletion_le_0_25": metrics["substitution_false_deletion_rate"] <= .25,
        "matched_macro_f1_ge_0_60": diagnostics["matched"]["macro_f1"] >= .60,
        "matched_deletion_f1_ge_0_55": diagnostics["matched"]["deletion"]["f1"] >= .55,
        "speaker_recall_gate": all(item["pass"] for item in speaker_gate.values()),
        "three_relation_macro_f1_ge_0_40": metrics["three_relation"]["macro_f1"] >= .40,
    }
    per_improvement = R4B["per"] - metrics["phone_error_rate"]
    binary_improvement = metrics["binary"]["macro_f1"] - R4B["binary_macro_f1"]
    deletion_improvement = metrics["binary"]["deletion"]["f1"] - R4B["deletion_f1"]
    components = {
        "decoded_target_ratio_plus_0_05": under["aggregate_decoded_target_ratio"] - R4B["decoded_target_ratio"] >= .05,
        "shorter_fraction_minus_0_05": R4B["shorter_fraction"] - under["words_decoded_shorter_fraction"] >= .05,
        "blank_occupancy_minus_0_05": R4B["blank_occupancy"] - under["blank_argmax_occupancy_mean"] >= .05,
        "deletion_edits_minus_10pct": under["ctc_target_phone_deletion_edit_count"] <= R4B["ctc_deletion_edits"] * .90,
        "clean_false_deletion_minus_0_05": R4B["clean_word_false_deletion"] - under["clean_word_false_expected_phone_deletion_rate"] >= .05,
    }
    development = {
        "per_improvement": per_improvement, "per_improvement_pass": per_improvement >= .02,
        "binary_macro_f1_improvement": binary_improvement, "binary_improvement_pass": binary_improvement >= .02,
        "deletion_f1_improvement": deletion_improvement, "deletion_improvement_pass": deletion_improvement >= .03,
        "under_generation_components": components, "under_generation_pass_count": sum(components.values()),
        "under_generation_improved": sum(components.values()) >= 2,
    }
    all_hard = all(gates.values())
    all_development = development["per_improvement_pass"] and development["binary_improvement_pass"] \
        and development["deletion_improvement_pass"] and development["under_generation_improved"]
    any_development = development["per_improvement_pass"] or development["binary_improvement_pass"] \
        or development["deletion_improvement_pass"] or development["under_generation_improved"]
    status = ("R4_4C2_BIGRU_CTC_CONFIRMED" if all_hard else
              "R4_4C2_BIGRU_CTC_IMPROVED" if all_development else
              "R4_4C2_BIGRU_CTC_MIXED" if any_development else
              "R4_4C2_BIGRU_CTC_NOT_IMPROVED")
    return {"hard_gates": gates, "speaker_gate": speaker_gate, "all_hard_gates_pass": all_hard}, {
        "status": status, "development": development,
    }


def main() -> int:
    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite experiment: {EXPERIMENT_DIR}")
    prereg, verification = verify_sources()
    determinism = set_determinism()
    audio_root = r4b.r3.require_audio_root()
    words, dataset = r4b.load_words(audio_root)
    train_words = [word for word in words if word["split"] == "train"]
    validation_words = [word for word in words if word["split"] == "validation"]
    if (len(train_words), len(validation_words)) != (16259, 7728):
        raise RuntimeError("Dataset population mismatch")
    empty_counts = (sum(not word["target_ids"] for word in train_words),
                    sum(not word["target_ids"] for word in validation_words))
    if empty_counts != (28, 15): raise RuntimeError(f"Empty target mismatch: {empty_counts}")
    matched = matched_ids()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    model = frozen_model.WordBiGRUCTCModel().to(device)
    parameters = frozen_model.parameter_report(model)
    if parameters["total"] != 198761: raise RuntimeError("Frozen parameter count mismatch")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)
    write_json(EXPERIMENT_DIR / "preflight.json", {
        "status": "PREPROCESSING", "freeze_verification": verification, "git_commit": git_commit(),
        "training_started": False, "test_paths_resolved": False, "test_audio_read": False,
        "test_target_reconstruction": False, "test_inference": False, "test_metrics": False,
    })
    write_json(EXPERIMENT_DIR / "determinism.json", {
        **determinism, "python": platform.python_version(), "pytorch": torch.__version__,
        "cuda": torch.version.cuda, "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cuda_ctc_backward_warning_observed": None,
    })
    write_json(EXPERIMENT_DIR / "dataset_summary.json", {
        **dataset, "usable_words": {"train": len(train_words), "validation": len(validation_words)},
        "target_phones": {"train": sum(len(word["target_ids"]) for word in train_words),
                          "validation": sum(len(word["target_ids"]) for word in validation_words)},
        "empty_targets": {"train": empty_counts[0], "validation": empty_counts[1]},
        "addition_words_excluded": True, "test_speakers_excluded": sorted(r4b.TEST_SPEAKERS),
    })
    write_json(EXPERIMENT_DIR / "feature_config.json", prereg["audio_and_features"])
    write_json(EXPERIMENT_DIR / "model_config.json", {
        "cnn": prereg["cnn_encoder"], "bigru": prereg["bigru"], "packing": prereg["variable_length_handling"],
        "head": prereg["head"], "parameters": parameters, "initialization": prereg["initialization"],
    })
    write_json(EXPERIMENT_DIR / "training_config.json", {
        **prereg["training"], "loss": prereg["loss"], "checkpoint_selection": prereg["checkpoint_selection"],
        "classification": prereg["result_classification_precedence"],
    })

    train_features, train_feature_report = r4b.materialize_features(train_words, device, "train")
    validation_features, validation_feature_report = r4b.materialize_features(validation_words, device, "validation")
    if train_feature_report["unalignable_words"] or validation_feature_report["unalignable_words"]:
        write_json(EXPERIMENT_DIR / "final_status.json", {
            "status": "R4_4C2_TEMPORAL_CONTRACT_FAIL", "training_occurred": False,
            "r4_test_accessed": False, "train": train_feature_report, "validation": validation_feature_report,
        })
        print("R4_4C2_TEMPORAL_CONTRACT_FAIL"); return 0
    for frames in sorted({feature.shape[-1] for feature in train_features + validation_features}):
        expected = r4b.encoder_steps(frames)
        actual = int(model.encoder_output_lengths(torch.tensor([frames]))[0])
        if actual != expected: raise RuntimeError("R4_4C2_TEMPORAL_CONTRACT_FAIL")
    preflight = json.loads((EXPERIMENT_DIR / "preflight.json").read_text(encoding="utf-8"))
    preflight.update({"status": "PASS", "temporal_contract": "PASS", "train_features": train_feature_report,
                      "validation_features": validation_feature_report, "training_started": True})
    write_json(EXPERIMENT_DIR / "preflight.json", preflight)

    criterion = nn.CTCLoss(blank=40, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    checkpoint_path = EXPERIMENT_DIR / CHECKPOINT_NAME
    history: list[dict[str, Any]] = []; best_per = math.inf; best_f1 = -1.0; best_epoch = 0
    captured_warnings: list[str] = []
    training_started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        if epoch == 1:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                train_metrics = train_epoch(model, train_words, train_features, criterion, optimizer, device, epoch)
                captured_warnings.extend(str(item.message) for item in caught)
        else:
            train_metrics = train_epoch(model, train_words, train_features, criterion, optimizer, device, epoch)
        val, _, _ = evaluate(model, validation_words, validation_features, criterion, device)
        record = {"epoch": epoch, "train": train_metrics, "validation": val,
                  "epoch_seconds": time.perf_counter() - epoch_started}
        history.append(record); save_history(history)
        current_per = val["phone_error_rate"]; current_f1 = val["binary"]["deletion"]["f1"]
        better = current_per < best_per - TIE_TOLERANCE or (
            abs(current_per - best_per) <= TIE_TOLERANCE and current_f1 > best_f1 + TIE_TOLERANCE)
        if better:
            best_per, best_f1, best_epoch = current_per, current_f1, epoch
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch, "validation_per": current_per,
                "validation_deletion_f1": current_f1, "vocabulary": list(r4b.PHONE_VOCAB),
                "blank_index": 40, "model_config": {"cnn": prereg["cnn_encoder"], "bigru": prereg["bigru"],
                                                      "head": prereg["head"]},
                "training_config": prereg["training"], "fresh_random_initialization": True,
                "r3_checkpoint_loaded": False, "r4_4b_checkpoint_loaded": False,
            }, checkpoint_path)
        print(f"epoch={epoch} train_loss={train_metrics['ctc_loss']:.6f} val_loss={val['loss']:.6f} "
              f"per={current_per:.6f} exact={val['exact_decoded_sequence_accuracy']:.6f} "
              f"mf1={val['binary']['macro_f1']:.6f} del_p={val['binary']['deletion']['precision']:.6f} "
              f"del_r={val['binary']['deletion']['recall']:.6f} del_f1={current_f1:.6f} "
              f"sub_fd={val['substitution_false_deletion_rate']:.6f} "
              f"three={val['three_relation']['macro_f1']:.6f} ratio={val['aggregate_decoded_target_ratio']:.6f} "
              f"blank={val['blank_argmax_occupancy_mean']:.6f} best={best_epoch}", flush=True)
    training_seconds = time.perf_counter() - training_started
    determinism_payload = json.loads((EXPERIMENT_DIR / "determinism.json").read_text(encoding="utf-8"))
    determinism_payload["captured_epoch1_warnings"] = captured_warnings
    determinism_payload["cuda_ctc_backward_warning_observed"] = any("determin" in item.lower() for item in captured_warnings)
    determinism_payload["strict_cuda_ctc_backward_guarantee"] = not determinism_payload["cuda_ctc_backward_warning_observed"]
    write_json(EXPERIMENT_DIR / "determinism.json", determinism_payload)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected, rows, details = evaluate(model, validation_words, validation_features, criterion, device)
    if abs(selected["phone_error_rate"] - best_per) > 1e-12: raise RuntimeError("Selected checkpoint mismatch")
    diagnostics = r4b.selected_diagnostics(validation_words, rows, details, selected, matched)
    for speaker, block in diagnostics["speakers"].items():
        speaker_details = [detail for detail in details if detail["speaker_id"] == speaker]
        block["decoded_phone_total"] = sum(len(detail["decoded_ids"]) for detail in speaker_details)
        block["decoded_target_ratio"] = block["decoded_phone_total"] / block["target_phones"]
        block["blank_argmax_occupancy"] = float(np.mean([detail["blank_argmax_occupancy"] for detail in speaker_details]))
    under = under_generation(selected, rows, details)
    gates, classification = gate_and_classify(selected, diagnostics, under)
    train_selected, _, train_details = evaluate(model, train_words, train_features, criterion, device)
    train_gap = {
        "train": {"per": train_selected["phone_error_rate"],
                  "decoded_target_ratio": train_selected["aggregate_decoded_target_ratio"],
                  "blank_argmax_occupancy": train_selected["blank_argmax_occupancy_mean"]},
        "validation": {"per": selected["phone_error_rate"],
                       "decoded_target_ratio": selected["aggregate_decoded_target_ratio"],
                       "blank_argmax_occupancy": selected["blank_argmax_occupancy_mean"]},
        "validation_minus_train_per": selected["phone_error_rate"] - train_selected["phone_error_rate"],
    }
    checkpoint_hash = sha256(checkpoint_path)

    write_json(EXPERIMENT_DIR / "selected_checkpoint.json", {
        "path": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": checkpoint_hash,
        "selected_epoch": best_epoch, "validation_per": selected["phone_error_rate"],
        "selection": prereg["checkpoint_selection"],
    })
    write_json(EXPERIMENT_DIR / "validation_per_metrics.json", {
        "loss": selected["loss"], "phone_error_rate": selected["phone_error_rate"],
        "per_delta_current_minus_r4_4b": selected["phone_error_rate"] - R4B["per"],
        "exact_decoded_sequence_accuracy": selected["exact_decoded_sequence_accuracy"],
        "target_phone_denominator": selected["target_phone_denominator"], "decoded_phone_total": selected["decoded_phone_total"],
        "edit_counts": selected["edit_counts"],
    })
    write_json(EXPERIMENT_DIR / "validation_binary_metrics.json", selected["binary"])
    write_json(EXPERIMENT_DIR / "validation_3class_metrics.json", selected["three_relation"])
    write_json(EXPERIMENT_DIR / "matched_control_metrics.json", diagnostics["matched"])
    write_json(EXPERIMENT_DIR / "speaker_metrics.json", diagnostics["speakers"])
    write_json(EXPERIMENT_DIR / "phone_metrics.json", {
        "observed_target_acoustic_metrics": diagnostics["acoustic_classes"],
        "deletion_by_expected_phone": diagnostics["phones"], "deletion_phone_groups": diagnostics["phone_groups"],
        "required_focus": {phone: diagnostics["acoustic_classes"][phone]
                           for phone in ("AW", "UH", "G", "NG", "TH", "OW", "AA", "DH", "R", "L", "AX", "OY", "ZH")},
    })
    write_json(EXPERIMENT_DIR / "position_metrics.json", diagnostics["positions"])
    write_json(EXPERIMENT_DIR / "multi_error_metrics.json", diagnostics["multi_error"])
    write_json(EXPERIMENT_DIR / "under_generation_metrics.json", under)
    write_json(EXPERIMENT_DIR / "empty_target_metrics.json", diagnostics["empty"])
    write_json(EXPERIMENT_DIR / "insertion_diagnostics.json", diagnostics["insertions"])
    write_json(EXPERIMENT_DIR / "train_validation_gap.json", train_gap)
    write_exports(rows, details)

    peak_f1 = max(history, key=lambda item: (item["validation"]["binary"]["deletion"]["f1"], -item["epoch"]))["epoch"]
    peak_mf1 = max(history, key=lambda item: (item["validation"]["binary"]["macro_f1"], -item["epoch"]))["epoch"]
    last = history[-1]["validation"]
    trend = {
        "selected_epoch": best_epoch, "minimum_validation_per": best_per,
        "peak_deletion_f1_epoch": peak_f1, "peak_binary_macro_f1_epoch": peak_mf1,
        "epoch_36": last, "selected_is_epoch_36": best_epoch == 36,
        "last_six_per": [item["validation"]["phone_error_rate"] for item in history[-6:]],
        "last_six_loss": [item["validation"]["loss"] for item in history[-6:]],
        "still_improving_at_36": best_epoch == 36,
        "overfitting_detected": best_epoch <= 30 and all(item["validation"]["phone_error_rate"] > best_per for item in history[-5:]),
        "extension_authorized": False,
    }
    final = {
        "status": classification["status"], "classification": classification,
        "hard_gate_report": gates, "selected_epoch": best_epoch,
        "training_seconds": training_seconds, "device": str(device),
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "training_occurred": True, "neural_runs_in_task": 1,
        "r4_test_accessed": False, "test_paths_resolved": False, "test_audio_read": False,
        "test_target_reconstruction": False, "test_inference": False, "test_metrics": False,
        "checkpoint_sha256": checkpoint_hash, "stop_policy_applies": True,
    }
    write_json(EXPERIMENT_DIR / "training_trend.json", trend)
    write_json(EXPERIMENT_DIR / "final_status.json", final)
    report = f"""# R4-4C2 Locked CNN+BiGRU CTC Experiment

RESEARCH_ONLY / NOT_PRODUCTION / R4_TEST_CLOSED

- Final: `{classification['status']}`
- Selected epoch: {best_epoch}
- Validation PER: {selected['phone_error_rate']:.6f} (improvement {R4B['per'] - selected['phone_error_rate']:+.6f})
- Binary Macro-F1: {selected['binary']['macro_f1']:.6f}
- Deletion P/R/F1: {selected['binary']['deletion']['precision']:.6f} / {selected['binary']['deletion']['recall']:.6f} / {selected['binary']['deletion']['f1']:.6f}
- Decoded/target ratio: {under['aggregate_decoded_target_ratio']:.6f}
- Blank argmax occupancy: {under['blank_argmax_occupancy_mean']:.6f}
- All frozen gates pass: {gates['all_hard_gates_pass']}
- Exactly one neural run: YES; R4 TEST access: NO
- Checkpoint SHA-256: `{checkpoint_hash}`
"""
    (EXPERIMENT_DIR / "r4_4c2_report.md").write_text(report, encoding="utf-8")
    artifact_names = [path.name for path in EXPERIMENT_DIR.iterdir() if path.name != "artifact_hashes.json"]
    write_json(EXPERIMENT_DIR / "artifact_hashes.json", {
        "algorithm": "SHA-256", "files": {name: sha256(EXPERIMENT_DIR / name) for name in sorted(artifact_names)},
        "note": "manifest excludes itself",
    })
    print(json.dumps({"final": final, "selected": selected, "under_generation": under,
                      "train_validation": train_gap, "trend": trend}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
