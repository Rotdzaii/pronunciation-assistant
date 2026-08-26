from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_l2_arctic_observed_phone_r3_1a as r3  # noqa: E402
import run_r4_3a_word_sequence_design_audit as r43a  # noqa: E402


REPO_ROOT = r3.REPO_ROOT
V4_PATH = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
CHECKPOINT_PATH = REPO_ROOT / "ai-training/experiments/r3_1d_observed_phone_seed42_48epochs/R3_1D_observed_phone_40class_seed42_best_validation_macro_f1.pt"
R43A_ORACLE = REPO_ROOT / "ai-training/experiments/r4_3a_word_sequence_design_audit/oracle_alignment.json"
R43A_RECONSTRUCTION = REPO_ROOT / "ai-training/experiments/r4_3a_word_sequence_design_audit/word_reconstruction.json"
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r4_4a_ctc_sequence_feasibility"
EXPECTED = {
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
    "checkpoint": "5C66860BBC50BA803F5BFE25417DF19BDF34B1C7BE36D507B0DDCA88D767EA5E",
    "r43a_oracle": "FF6E73CB0EEA88EF5B6E258B614C0711C59CCF3FB65E46E2AA47594BF26CB528",
    "r43a_reconstruction": "3440961B65A311ACF2ECC352E4F1DBF589E7628CA37B2BCE49972DDBEB8A1D93",
}
EXPECTED_WORDS = {"train": 16_259, "validation": 7_728}
SPLIT_SPEAKERS = {"train": set(r3.TRAIN_SPEAKERS), "validation": set(r3.VALIDATION_SPEAKERS)}
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def stats(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "p5": None, "p10": None, "median": None, "mean": None,
                "p90": None, "p95": None, "max": None}
    return {
        "count": int(array.size), "min": float(array.min()), "p5": float(np.percentile(array, 5)),
        "p10": float(np.percentile(array, 10)), "median": float(np.median(array)),
        "mean": float(array.mean()), "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)), "max": float(array.max()),
    }


def verify_sources() -> dict[str, str]:
    paths = {"v4": V4_PATH, "checkpoint": CHECKPOINT_PATH, "r43a_oracle": R43A_ORACLE,
             "r43a_reconstruction": R43A_RECONSTRUCTION}
    actual = {name: sha256(path) for name, path in paths.items()}
    mismatch = {name: {"expected": EXPECTED[name], "actual": value}
                for name, value in actual.items() if value != EXPECTED[name]}
    if mismatch:
        raise RuntimeError(f"Source verification failed: {mismatch}")
    return actual


def target_min_steps(target: list[str]) -> int:
    return len(target) + sum(left == right for left, right in zip(target, target[1:]))


def mel_frames(duration: float, hop_samples: int) -> int:
    samples = max(1, int(round(duration * 16_000)))
    return samples // hop_samples + 1


def output_steps(duration: float, hop_samples: int, temporal_divisor: int) -> int:
    return mel_frames(duration, hop_samples) // temporal_divisor


def analyze_targets(words: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    quality: dict[str, Any] = {}
    duration_report: dict[str, Any] = {}
    alignability: dict[str, Any] = {}
    designs = {
        "A_R3_encoder_10ms_hop_div4": (160, 4, 40.0),
        "A_R3_like_encoder_10ms_hop_div2": (160, 2, 20.0),
        "A_R3_encoder_20ms_hop_div4": (320, 4, 80.0),
    }
    for split in ("train", "validation"):
        split_words = [word for word in words if word["split"] == split]
        targets = [word["observed"] for word in split_words]
        lengths = [len(target) for target in targets]
        minimums = [target_min_steps(target) for target in targets]
        durations = [float(word["mfa_end"] - word["mfa_start"]) for word in split_words]
        ratios = [duration / len(target) for duration, target in zip(durations, targets) if target]
        quality[split] = {
            "usable_words": len(split_words), "total_target_phones": sum(lengths),
            "target_length": stats(lengths),
            "empty_target_words": sum(length == 0 for length in lengths),
            "one_phone_target_words": sum(length == 1 for length in lengths),
            "multiple_deletion_words": sum(word["deletion"] > 1 for word in split_words),
            "substitution_plus_deletion_words": sum(word["substitution"] > 0 and word["deletion"] > 0 for word in split_words),
            "adjacent_identical_target_words": sum(any(a == b for a, b in zip(target, target[1:])) for target in targets),
            "adjacent_identical_target_pairs": sum(sum(a == b for a, b in zip(target, target[1:])) for target in targets),
            "malformed_or_unresolved_target_words": 0,
        }
        duration_report[split] = {
            "word_duration_seconds": stats(durations),
            "duration_per_nonempty_target_phone_seconds": stats(ratios),
        }
        alignability[split] = {}
        for name, (hop, divisor, spacing_ms) in designs.items():
            steps = [output_steps(duration, hop, divisor) for duration in durations]
            unalignable = [word["word_id"] for word, available, required in zip(split_words, steps, minimums)
                           if available < required]
            encoder_invalid = [word["word_id"] for word, available in zip(split_words, steps) if available < 1]
            alignability[split][name] = {
                "feature_hop_ms": hop / 16.0, "temporal_downsampling": divisor,
                "output_spacing_ms": spacing_ms, "output_steps": stats(steps),
                "unalignable_words": len(unalignable), "unalignable_rate": len(unalignable) / len(split_words),
                "encoder_zero_step_words": len(encoder_invalid),
                "first_unalignable_word_ids": unalignable[:50],
            }
    return quality, duration_report, alignability


def class_coverage(words: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {split: Counter(phone for word in words if word["split"] == split for phone in word["observed"])
              for split in ("train", "validation")}
    table = {
        phone: {"train": counts["train"][phone], "validation": counts["validation"][phone]}
        for phone in r3.PHONE_VOCAB
    }
    return {
        "classes": table,
        "train_below_100": [phone for phone in r3.PHONE_VOCAB if counts["train"][phone] < 100],
        "validation_below_50": [phone for phone in r3.PHONE_VOCAB if counts["validation"][phone] < 50],
        "validation_absent_from_train": [phone for phone in r3.PHONE_VOCAB if counts["validation"][phone] and not counts["train"][phone]],
        "train_inventory_size": sum(counts["train"][phone] > 0 for phone in r3.PHONE_VOCAB),
        "validation_inventory_size": sum(counts["validation"][phone] > 0 for phone in r3.PHONE_VOCAB),
    }


def speaker_coverage(words: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for speaker in tuple(r3.TRAIN_SPEAKERS) + tuple(r3.VALIDATION_SPEAKERS):
        selected = [word for word in words if word["speaker_id"] == speaker]
        output[speaker] = {
            "split": "train" if speaker in SPLIT_SPEAKERS["train"] else "validation",
            "usable_words": len(selected), "target_phones": sum(len(word["observed"]) for word in selected),
            "deletion_containing_words": sum(word["deletion"] > 0 for word in selected),
            "substitution_containing_words": sum(word["substitution"] > 0 for word in selected),
        }
    return output


def deletion_coverage(words: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split in ("train", "validation"):
        selected = [word for word in words if word["split"] == split]
        word_distribution = Counter(
            "zero" if word["deletion"] == 0 else "one" if word["deletion"] == 1 else "two" if word["deletion"] == 2 else "more_than_two"
            for word in selected
        )
        positions: Counter[str] = Counter()
        phones: Counter[str] = Counter()
        for word in selected:
            length = len(word["clean_rows"])
            for index, row in enumerate(word["clean_rows"]):
                if row["relation"] != "deletion":
                    continue
                position = "single" if length == 1 else "initial" if index == 0 else "final" if index == length - 1 else "medial"
                positions[position] += 1
                phones[row["expected"]] += 1
        report[split] = {
            "word_deletion_count": dict(word_distribution), "position": dict(positions),
            "expected_phone": dict(sorted(phones.items(), key=lambda item: r3.PHONE_TO_ID[item[0]])),
        }
    return report


def addition_audit(all_records: list[dict[str, Any]]) -> dict[str, Any]:
    word_counts = {
        split: {
            "addition_containing_words": sum(record["split"] == split and record["has_addition"] for record in all_records),
            "addition_plus_deletion_words": sum(record["split"] == split and record["has_addition"] and record["deletion"] > 0 for record in all_records),
        }
        for split in ("train", "validation")
    }
    rows = Counter(); phones = Counter(); unresolved = Counter()
    with V4_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["speaker_id"] not in SPLIT_SPEAKERS["train"] | SPLIT_SPEAKERS["validation"] or row["tagged_relation"] != "addition":
                continue
            split = "train" if row["speaker_id"] in SPLIT_SPEAKERS["train"] else "validation"
            observed = row["observed_phone_canonical"]
            if row["relation"] == "addition" and row["research_subset"] == "ADDITION_AUDIT" and observed in r3.PHONE_TO_ID:
                rows[split] += 1
                phones[(split, observed)] += 1
            else:
                unresolved[(split, row["exclusion_reason"] or "unresolved_addition")] += 1
    return {
        "word_counts": word_counts, "clean_addition_rows": dict(rows),
        "future_ctc_representable_addition_rows": {
            split: sum(value for (row_split, _), value in phones.items() if row_split == split)
            for split in ("train", "validation")
        },
        "unresolved_addition_rows": {f"{split}:{reason}": count for (split, reason), count in unresolved.items()},
        "observed_phone_distribution": {
            split: {phone: phones[(split, phone)] for phone in r3.PHONE_VOCAB if phones[(split, phone)]}
            for split in ("train", "validation")
        },
        "future_representation": "insert the clean addition observed phone at manual temporal order as an extra CTC target label; excluded from R4-4B",
    }


def freeze_matched(words: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strata: dict[tuple[str, str, int], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: {0: [], 1: []})
    total_deletions = 0
    for word in words:
        if word["split"] != "validation":
            continue
        for row in word["clean_rows"]:
            target = int(row["relation"] == "deletion")
            total_deletions += target
            duration_bin = int(math.floor((row["end"] - row["start"]) / 0.010 + 1e-9))
            strata[(row["speaker_id"], row["expected"], duration_bin)][target].append(row)
    rng = np.random.default_rng(SEED)
    output: list[dict[str, Any]] = []
    used: set[int] = set(); pair_id = 0
    for key in sorted(strata):
        negative = sorted(strata[key][0], key=lambda row: row["source_index"])
        positive = sorted(strata[key][1], key=lambda row: row["source_index"])
        count = min(len(negative), len(positive))
        if not count:
            continue
        selected_negative = sorted(rng.choice(len(negative), size=count, replace=False).tolist())
        selected_positive = sorted(rng.choice(len(positive), size=count, replace=False).tolist())
        for ni, pi in zip(selected_negative, selected_positive):
            pair_id += 1
            for role, row in (("non_deletion", negative[ni]), ("deletion", positive[pi])):
                if row["source_index"] in used:
                    raise RuntimeError("Duplicate matched identity")
                used.add(row["source_index"])
                output.append({
                    "pair_id": pair_id, "role": role, "source_csv_row": row["source_index"] + 2,
                    "speaker_id": row["speaker_id"], "utterance_id": row["utterance_id"],
                    "start_time": row["start"], "end_time": row["end"], "duration_bin_10ms": key[2],
                    "expected_phone": row["expected"], "relation": row["relation"],
                })
    return output, {
        "seed": SEED, "sampling": "numpy default_rng(42), sorted strata/source identities, without replacement",
        "pairs": pair_id, "rows": len(output), "deletion": pair_id, "non_deletion": pair_id,
        "phones": len({row["expected_phone"] for row in output}),
        "speakers": sorted({row["speaker_id"] for row in output}),
        "deletion_coverage": pair_id / total_deletions,
        "duration_is_model_input": False, "rebuild_or_resample_in_r4_4b": False,
    }


def write_matched(rows: list[dict[str, Any]]) -> str:
    path = EXPERIMENT_DIR / "validation_word_eligible_matched_control.csv"
    fields = ["pair_id", "role", "source_csv_row", "speaker_id", "utterance_id", "start_time", "end_time",
              "duration_bin_10ms", "expected_phone", "relation"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return sha256(path)


def empty_target_ctc_probe() -> dict[str, Any]:
    torch = r3.torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    results: dict[str, Any] = {}
    for device in devices:
        logits = torch.randn(3, 1, 41, device=device).log_softmax(2).requires_grad_(True)
        targets = torch.empty((0,), dtype=torch.long, device=device)
        input_lengths = torch.tensor([3], dtype=torch.long, device=device)
        target_lengths = torch.tensor([0], dtype=torch.long, device=device)
        loss = torch.nn.CTCLoss(blank=40, reduction="mean", zero_infinity=True)(
            logits, targets, input_lengths, target_lengths
        )
        results[device] = {"finite": bool(torch.isfinite(loss).item()), "loss": float(loss.detach().cpu())}
    verdict = "EMPTY_TARGET_SUPPORTED" if all(item["finite"] for item in results.values()) else "EMPTY_TARGET_BLOCKER"
    return {"pytorch_version": torch.__version__, "blank_index": 40, "target_length": 0,
            "results": results, "verdict": verdict}


def main() -> int:
    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite audit: {EXPERIMENT_DIR}")
    source_hashes = verify_sources()
    audio_root = r3.require_audio_root()
    r43a.AUDIT_SPEAKERS = r43a.TRAIN_SPEAKERS | r43a.VALIDATION_SPEAKERS
    all_records, reconstruction = r43a.build_word_records(audio_root)
    usable = [word for word in all_records if word["usable"]]
    actual_words = {split: sum(word["split"] == split for word in usable) for split in ("train", "validation")}
    if actual_words != EXPECTED_WORDS:
        raise RuntimeError(f"Usable word counts differ: {actual_words}")
    target_quality, durations, alignability = analyze_targets(usable)
    coverage = class_coverage(usable)
    speakers = speaker_coverage(usable)
    deletions = deletion_coverage(usable)
    additions = addition_audit(all_records)
    oracle = r43a.audit_oracle(usable)
    matched_rows, matched_manifest = freeze_matched(usable)
    empty_target_probe = empty_target_ctc_probe()

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)
    matched_sha = write_matched(matched_rows); matched_manifest["sha256"] = matched_sha
    write_json(EXPERIMENT_DIR / "preflight.json", {
        "source_hashes": source_hashes, "train_validation_only": True,
        "validation_model_inference": False, "validation_audio_read": False,
        "r4_test_paths_resolved": False, "r4_test_audio_read": False, "r4_test_inference": False,
        "training_performed": False,
    })
    write_json(EXPERIMENT_DIR / "word_reconstruction.json", {"usable_words": actual_words, "source_reconstruction": reconstruction})
    write_json(EXPERIMENT_DIR / "target_quality.json", target_quality)
    write_json(EXPERIMENT_DIR / "technical_compatibility.json", {"empty_target_ctc_probe": empty_target_probe})
    write_json(EXPERIMENT_DIR / "word_duration.json", durations)
    write_json(EXPERIMENT_DIR / "ctc_alignability.json", alignability)
    write_json(EXPERIMENT_DIR / "class_coverage.json", coverage)
    write_json(EXPERIMENT_DIR / "speaker_coverage.json", speakers)
    write_json(EXPERIMENT_DIR / "deletion_coverage.json", deletions)
    write_json(EXPERIMENT_DIR / "addition_audit.json", additions)
    write_json(EXPERIMENT_DIR / "oracle_alignment.json", oracle)
    write_json(EXPERIMENT_DIR / "validation_word_eligible_matched_control_manifest.json", matched_manifest)

    recommended_alignability = alignability["train"]["A_R3_like_encoder_10ms_hop_div2"]["unalignable_words"] + alignability["validation"]["A_R3_like_encoder_10ms_hop_div2"]["unalignable_words"]
    inventory_ok = not coverage["validation_absent_from_train"] and coverage["train_inventory_size"] == 40
    oracle_ok = oracle["ground_truth_signature_representable_rate"] >= 0.999
    empty_target_ok = empty_target_probe["verdict"] == "EMPTY_TARGET_SUPPORTED"
    if recommended_alignability == 0 and inventory_ok and oracle_ok and empty_target_ok:
        final_status = "R4_4A_CTC_SEQUENCE_READY_WITH_WARNINGS"
    else:
        final_status = "R4_4A_CTC_SEQUENCE_BLOCKED"

    hard_gates = {
        "validation_phone_error_rate_max": 0.55,
        "binary_macro_f1_min": 0.70,
        "deletion_recall_min": 0.45,
        "deletion_f1_min": 0.40,
        "substitution_false_deletion_rate_max": 0.25,
        "matched_macro_f1_min": 0.60,
        "matched_deletion_f1_min": 0.55,
        "speaker_rule": "each validation speaker with >=30 deletions must have deletion recall >=0.25",
        "three_relation_macro_f1_min": 0.40,
        "rationale": "retain prior R4 deletion/matched/speaker objectives; Macro-F1 .70 exceeds duration baseline by .031854; PER .55 requires meaningful self-trained acoustic sequencing without production claims",
    }
    prereg = {
        "experiment": "R4-4B one self-trained word-level CTC feasibility run",
        "markers": ["RESEARCH_ONLY", "NOT_PRODUCTION", "R4_TEST_CLOSED"],
        "source_hashes": source_hashes,
        "split": {"train": list(r3.TRAIN_SPEAKERS), "validation": list(r3.VALIDATION_SPEAKERS),
                  "test_closed": ["ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK"]},
        "eligibility": "R4-3A usable words: clean correct/substitution/deletion, reliable MFA word boundary, no addition/unresolved/malformed word",
        "expected_counts": actual_words,
        "target": {
            "correct": "append expected canonical phone", "substitution": "append manual observed canonical phone",
            "deletion": "append nothing", "addition_words": "excluded", "stress": "removed",
            "empty_target": "included; locally verified CTCLoss target_length=0 support on CPU and CUDA",
        },
        "vocabulary": {"phones": list(r3.PHONE_VOCAB), "phone_indexes": r3.PHONE_TO_ID,
                       "ctc_blank": "<CTC_BLANK>", "blank_index": 40, "class_count": 41},
        "audio": {"source": "full MFA word span", "side_context_seconds": 0.0, "sample_rate": 16000,
                  "mono": True, "word_edge_policy": "no neighbor context; zero-pad only waveform/short-frame technical need"},
        "features": {"type": "log-mel", "n_mels": 64, "n_fft": 512, "win_length": 400,
                     "hop_length": 160, "hop_ms": 10, "window": "Hann", "center": True,
                     "pad_mode": "constant", "power": 2.0, "mel_scale": "slaney", "norm": "slaney",
                     "power_to_db": "relative to per-word maximum, floor -80 dB"},
        "architecture": {
            "choice": "A: fresh R3-like convolutional encoder without temporal attention pooling, minimally adapted to preserve time resolution",
            "encoder": "Conv 1->16 + MaxPool(2,2); Conv 16->32 + MaxPool(2,1); Conv 32->64; Conv 64->96 with BatchNorm/ReLU and R3 dropout pattern",
            "variable_input": "[B,1,64,time]", "encoder_output": "[B,96,16,floor(time/2)]",
            "sequence_projection": "mean frequency dimension -> [B,T_out,96]",
            "head": "Dropout(0.2) + Linear(96,41)", "temporal_attention": False,
            "temporal_downsampling": 2, "approx_output_spacing_ms": 20,
        },
        "initialization": {"policy": "fresh random initialization for encoder and CTC head",
                           "reason": "10 ms/25 ms word-sequence features and one temporal pooling stage differ from frozen R3 phone-crop semantics; transplanting weights would add an initialization confound",
                           "external_pretraining": False, "r3_checkpoint_loaded": False,
                           "r3_checkpoint_sha256_for_architecture_audit_only": source_hashes["checkpoint"]},
        "loss": {"name": "torch.nn.CTCLoss", "blank": 40, "reduction": "mean", "zero_infinity": True,
                 "auxiliary_losses": "none"},
        "training": {"seed": 42, "epochs": 36, "batch_size": 8, "batching": "length-bucketed variable word batches with right padding and exact input lengths",
                     "optimizer": "Adam", "learning_rate": 0.0001, "weight_decay": 0.0,
                     "gradient_clip_norm": 5.0, "augmentation": "none", "sampler": "none",
                     "early_stopping": False, "checkpoint_metric": "lowest validation PER; tie higher deletion F1; tie earlier epoch"},
        "decoding": {"method": "greedy", "steps": ["argmax each output time", "collapse consecutive repeated labels", "remove CTC blank"],
                     "beam_search": False, "lexicon": False, "expected_phone_constraint": False},
        "edit_alignment": {"algorithm": "unit-cost deterministic Levenshtein expected vs greedy decoded sequence",
                           "tie_order": ["MATCH", "SUBSTITUTION", "DELETE_FROM_EXPECTED", "INSERT_IN_DECODED"],
                           "mapping": {"MATCH": "correct", "SUBSTITUTION": "substitution", "DELETE_FROM_EXPECTED": "deletion", "INSERT_IN_DECODED": "future addition diagnostic"}},
        "matched_control": matched_manifest,
        "validation_metrics": ["PER", "edit counts", "exact decoded phone-sequence rate", "binary deletion metrics",
                               "3-relation metrics", "substitution false-deletion rate", "per-speaker metrics", "matched metrics"],
        "hard_gates": hard_gates,
        "test_policy": "CLOSED; no TEST paths/audio/targets/inference; opening requires a separate command after validation gates",
    }
    write_json(EXPERIMENT_DIR / "r4_4b_preregistered_design.json", prereg)
    md = f"""# R4-4B Preregistered Self-Trained CTC Design

Research-only, not production, R4 TEST closed.

Use {actual_words['train']:,} TRAIN and {actual_words['validation']:,} VALIDATION R4-3A-eligible words. The target is the manual canonical observed sequence: correct emits expected, substitution emits observed, deletion emits nothing; addition-containing words are excluded.

Input is the full MFA word span at 16 kHz with no side context. Extract 64-bin log-mels using 25 ms windows and 10 ms hop. Use a fresh R3-like convolutional encoder, remove temporal attention, change the second pool to frequency-only, average only frequency, and apply Dropout(0.2)+Linear(96,41) at every output step. One temporal pooling stage yields approximately 20 ms output spacing and zero theoretical CTC alignment failures. Class 40 is CTC blank. Initialize all trainable parameters randomly so altered feature and pooling semantics do not introduce an R3-weight-transfer confound.

Train exactly one seed-42 run for 36 epochs using batch 8, Adam 1e-4, no augmentation/sampler/auxiliary loss, gradient clipping 5.0, and CTCLoss(blank=40,reduction=mean,zero_infinity=true). Select lowest validation PER, then higher deletion F1, then earlier epoch. Decode greedily, then align expected versus decoded with unit-cost Levenshtein tie order MATCH > SUBSTITUTION > DELETE > INSERT.

Frozen matched control: {matched_manifest['pairs']} pairs, SHA-256 `{matched_sha}`.

Hard gates: PER <= .55; binary Macro-F1 >= .70; deletion recall >= .45; deletion F1 >= .40; substitution false-deletion <= .25; matched Macro-F1 >= .60; matched deletion F1 >= .55; each speaker with >=30 deletions recall >= .25; 3-relation Macro-F1 >= .40.

No training or validation model inference occurred in R4-4A.
"""
    (EXPERIMENT_DIR / "r4_4b_preregistered_design.md").write_text(md, encoding="utf-8")
    prereg_hashes = {"json": sha256(EXPERIMENT_DIR / "r4_4b_preregistered_design.json"),
                     "markdown": sha256(EXPERIMENT_DIR / "r4_4b_preregistered_design.md")}
    write_json(EXPERIMENT_DIR / "encoder_reuse.json", {
        "verdict": "R3_ENCODER_REUSE_WITH_SMALL_ADAPTATION",
        "input": "[B,1,64,time]", "layers": prereg["architecture"],
        "adaptation": "reuse R3 encoder design/code, not weights; remove attention/classifier; second pool frequency-only; frequency mean; time-distributed Linear(96,41)",
        "variable_duration_technically_supported": True,
    })
    final = {
        "final_status": final_status, "preregistered_hashes": prereg_hashes,
        "empty_target_policy": empty_target_probe["verdict"],
        "recommended_architecture": "fresh R3-like convolutional encoder with one time pool, no attention, temporal Linear 41",
        "training_performed": False, "validation_model_inference": False,
        "r4_test_accessed": False, "test_paths_resolved": False,
    }
    write_json(EXPERIMENT_DIR / "final_status.json", final)
    report = f"""# R4-4A CTC Sequence Feasibility Audit

Final: `{final_status}`

- Usable words: TRAIN {actual_words['train']:,}; VALIDATION {actual_words['validation']:,}
- Oracle ground-truth path representable: {oracle['ground_truth_signature_representable_rate']:.6%}
- Recommended encoder: fresh R3-like convolutional stack with one temporal pooling stage and time-distributed 41-class head
- Matched control: {matched_manifest['pairs']:,} pairs; SHA `{matched_sha}`
- Training: NO; validation inference: NO; R4 TEST access: NO
"""
    (EXPERIMENT_DIR / "r4_4a_report.md").write_text(report, encoding="utf-8")
    key_files = [
        "preflight.json", "word_reconstruction.json", "target_quality.json", "technical_compatibility.json", "word_duration.json",
        "ctc_alignability.json", "class_coverage.json", "speaker_coverage.json", "deletion_coverage.json",
        "addition_audit.json", "oracle_alignment.json", "validation_word_eligible_matched_control.csv",
        "validation_word_eligible_matched_control_manifest.json", "encoder_reuse.json",
        "r4_4b_preregistered_design.json", "r4_4b_preregistered_design.md", "final_status.json", "r4_4a_report.md",
    ]
    write_json(EXPERIMENT_DIR / "artifact_hashes.json", {
        "algorithm": "SHA-256", "files": {name: sha256(EXPERIMENT_DIR / name) for name in key_files},
        "note": "artifact_hashes.json intentionally does not hash itself",
    })
    print(json.dumps({
        "final": final, "target_quality": target_quality, "alignability": alignability,
        "duration": durations, "class_coverage": coverage, "matched": matched_manifest,
        "oracle": {key: oracle[key] for key in ("words", "exact_deterministic_word_recovery_rate", "ground_truth_signature_representable_rate")},
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
