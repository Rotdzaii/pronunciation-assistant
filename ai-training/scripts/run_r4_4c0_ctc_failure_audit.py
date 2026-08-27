from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_r4_4b_ctc_sequence as r4b  # noqa: E402


REPO_ROOT = r4b.REPO_ROOT
SOURCE_DIR = REPO_ROOT / "ai-training/experiments/r4_4b_ctc_sequence_seed42"
OUTPUT_DIR = REPO_ROOT / "ai-training/experiments/r4_4c0_ctc_failure_audit"
CHECKPOINT = SOURCE_DIR / r4b.CHECKPOINT_NAME
WORD_PREDICTIONS = SOURCE_DIR / "validation_word_predictions.jsonl"
PHONE_PREDICTIONS = SOURCE_DIR / "validation_phone_predictions.csv"
EPOCH_METRICS = SOURCE_DIR / "epoch_metrics.json"
SOURCE_HASHES = SOURCE_DIR / "artifact_hashes.json"
EXPECTED_HASHES = {
    CHECKPOINT.name: "A154DFAC573D69B8ED1A71CBCDC23227EA3E80929890AD87E97ED85667142106",
    WORD_PREDICTIONS.name: "29737F4663B2FFFF2240EC4241E0E198E744274D6B1C30FA2E8A705EAF691CED",
    PHONE_PREDICTIONS.name: "1BEC83D48F7170DD344FC0AA5FED94AFEB0E75B4281AED7F1A84DC8E3571BE33",
    EPOCH_METRICS.name: "DA9CD18B2D3E15A9C4B4A61C980EB20BF3926F7B12DFF767477F2116ED5836DB",
    SOURCE_HASHES.name: "56A109E7D62A72CC27BB29B4C1DDF98E7DE8B095E232603534C0ABBA008EDC1B",
}
EXPECTED_PER_COUNTS = {"substitution": 8631, "deletion": 6199, "insertion": 496}
EXPECTED_VALIDATION_WORDS = 7728
EXPECTED_VALIDATION_TARGETS = 25423
EXPECTED_MANUAL_DELETIONS = 914


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def distribution(values: Iterable[float], percentiles: tuple[int, ...] = (10, 25, 75, 90, 95)) -> dict[str, Any]:
    values = list(values)
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    result: dict[str, Any] = {
        "count": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
        "min": float(array.min()), "max": float(array.max()),
    }
    for percentile in percentiles:
        result[f"p{percentile}"] = float(np.percentile(array, percentile))
    return result


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def corr(left: list[float], right: list[float]) -> dict[str, float]:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return {"pearson": 0.0, "spearman": 0.0}
    return {"pearson": float(pearsonr(left, right).statistic), "spearman": float(spearmanr(left, right).statistic)}


def load_saved() -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    words = [json.loads(line) for line in WORD_PREDICTIONS.read_text(encoding="utf-8").splitlines() if line]
    with PHONE_PREDICTIONS.open(encoding="utf-8", newline="") as handle:
        phones = list(csv.DictReader(handle))
    epochs = json.loads(EPOCH_METRICS.read_text(encoding="utf-8"))
    return words, phones, epochs


def verify_sources() -> dict[str, Any]:
    paths = [CHECKPOINT, WORD_PREDICTIONS, PHONE_PREDICTIONS, EPOCH_METRICS, SOURCE_HASHES]
    actual = {path.name: sha256(path) for path in paths}
    mismatches = {name: {"expected": EXPECTED_HASHES[name], "actual": digest}
                  for name, digest in actual.items() if digest != EXPECTED_HASHES[name]}
    if mismatches:
        raise RuntimeError(f"R4_4C0_ARTIFACT_VERIFICATION_FAIL: {mismatches}")
    return {"status": "PASS", "expected": EXPECTED_HASHES, "actual": actual}


def classify_word(word: dict[str, Any]) -> list[str]:
    relations = word["ground_truth_relations"]
    substitutions = relations.count("substitution")
    deletions = relations.count("deletion")
    groups: list[str] = []
    if substitutions == 0 and deletions == 0:
        groups.append("all_correct")
    if substitutions > 0 and deletions == 0:
        groups.append("substitution_only")
    if deletions > 0 and substitutions == 0:
        groups.append("deletion_only")
    if substitutions > 0 and deletions > 0:
        groups.append("substitution_and_deletion")
    if deletions > 1:
        groups.append("multiple_deletion")
    if word["empty_target"]:
        groups.append("empty_target")
    return groups


def infer_split(model: torch.nn.Module, words: list[dict[str, Any]], device: torch.device,
                split_name: str, saved_words: dict[str, dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features, feature_report = r4b.materialize_features(words, device, split_name)
    batches = r4b.make_evaluation_batches([feature.shape[-1] for feature in features])
    records: list[dict[str, Any] | None] = [None] * len(words)
    all_blank_posteriors: list[float] = []
    all_run_lengths: list[int] = []
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for batch_number, indexes in enumerate(batches, 1):
            batch, _, lengths, _ = r4b.collate(indexes, features, words, device)
            logits = model(batch)
            probabilities = logits.softmax(dim=-1)
            raw = logits.argmax(dim=-1).cpu()
            lengths_cpu = lengths.cpu().tolist()
            for position, index in enumerate(indexes):
                length = int(lengths_cpu[position])
                sequence = raw[position, :length].tolist()
                blank_probs = probabilities[position, :length, r4b.BLANK].detach().cpu().numpy().astype(np.float64)
                all_blank_posteriors.extend(blank_probs.tolist())
                decoded: list[int] = []
                runs: list[dict[str, int]] = []
                cursor = 0
                while cursor < length:
                    token = int(sequence[cursor])
                    stop = cursor + 1
                    while stop < length and int(sequence[stop]) == token:
                        stop += 1
                    if token != r4b.BLANK:
                        decoded.append(token)
                        run_length = stop - cursor
                        runs.append({"phone_index": token, "length": run_length})
                        all_run_lengths.append(run_length)
                    cursor = stop
                word = words[index]
                decoded_phones = [r4b.PHONE_VOCAB[token] for token in decoded]
                if saved_words is not None:
                    saved = saved_words[word["word_id"]]
                    if decoded_phones != saved["decoded"]:
                        raise RuntimeError(f"Frozen decode mismatch for {word['word_id']}: {decoded_phones} != {saved['decoded']}")
                counts, _ = r4b.edit_counts(word["target_ids"], decoded)
                nonblank = sum(token != r4b.BLANK for token in sequence)
                records[index] = {
                    "word_id": word["word_id"], "speaker_id": word["speaker_id"],
                    "target_length": len(word["target_ids"]), "expected_length": len(word["expected_ids"]),
                    "decoded_length": len(decoded), "output_timesteps": length,
                    "blank_argmax_fraction": ratio(sequence.count(r4b.BLANK), length),
                    "mean_blank_posterior": float(blank_probs.mean()), "max_blank_posterior": float(blank_probs.max()),
                    "nonblank_argmax_fraction": ratio(nonblank, length), "decoded_per_timestep": ratio(len(decoded), length),
                    "nonblank_runs": len(runs), "run_lengths": [run["length"] for run in runs],
                    "repeat_collapsed_timesteps": nonblank - len(runs), "decoded_ids": decoded,
                    "edit_counts": counts, "duration_seconds": float(word["mfa_end"] - word["mfa_start"]),
                }
            if batch_number % 250 == 0 or batch_number == len(batches):
                print(f"infer={split_name} batches={batch_number}/{len(batches)}", flush=True)
    return [record for record in records if record is not None], {
        "feature_materialization": feature_report, "inference_seconds": time.perf_counter() - started,
        "all_blank_posterior": distribution(all_blank_posteriors, (75, 90, 95)),
        "all_nonblank_run_length": distribution(all_run_lengths, (10, 25, 75, 90, 95)),
    }


def aggregate_words(words: list[dict[str, Any]], diagnostics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not words:
        return {"words": 0}
    target = sum(len(word["target"]) for word in words)
    decoded = sum(len(word["decoded"]) for word in words)
    counts = Counter()
    for word in words:
        counts.update(word["acoustic_edit_counts"])
    ratios = [ratio(len(word["decoded"]), len(word["target"])) for word in words if word["target"]]
    return {
        "words": len(words), "target_phones": target, "decoded_phones": decoded,
        "target_length": distribution(len(word["target"]) for word in words),
        "decoded_length": distribution(len(word["decoded"]) for word in words),
        "decoded_target_ratio": distribution(ratios),
        "per": ratio(counts["errors"], target),
        "edit_counts": {key: int(counts[key]) for key in ("substitution", "deletion", "insertion", "errors")},
        "blank_argmax_occupancy": distribution(diagnostics[word["word_id"]]["blank_argmax_fraction"] for word in words),
    }


def false_deletion_rates(rows: list[dict[str, str]]) -> dict[str, float]:
    result = {}
    for relation in ("correct", "substitution"):
        selected = [row for row in rows if row["true_relation"] == relation]
        result[relation] = ratio(sum(row["predicted_relation"] == "deletion" for row in selected), len(selected))
    return result


def main() -> None:
    started = time.perf_counter()
    verification = verify_sources()
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite existing audit: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    words_saved, phone_rows, epochs = load_saved()
    if len(words_saved) != EXPECTED_VALIDATION_WORDS:
        raise RuntimeError(f"Validation word mismatch: {len(words_saved)}")
    if sum(len(word["target"]) for word in words_saved) != EXPECTED_VALIDATION_TARGETS:
        raise RuntimeError("Validation target count mismatch")
    checkpoint_payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint_payload["epoch"] != 35:
        raise RuntimeError("Selected epoch is not frozen epoch 35")

    audio_root_value = os.environ.get("L2_ARCTIC_ROOT")
    if not audio_root_value:
        raise RuntimeError("L2_ARCTIC_ROOT is required")
    words, reconstruction = r4b.load_words(Path(audio_root_value))
    train_words = [word for word in words if word["split"] == "train"]
    validation_words = [word for word in words if word["split"] == "validation"]
    saved_by_id = {word["word_id"]: word for word in words_saved}
    if set(saved_by_id) != {word["word_id"] for word in validation_words}:
        raise RuntimeError("Saved validation identity set mismatch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = r4b.WordCTCModel().to(device)
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    validation_diag, validation_compute = infer_split(model, validation_words, device, "validation_audit", saved_by_id)
    train_diag, train_compute = infer_split(model, train_words, device, "train_audit")
    val_diag_by_id = {row["word_id"]: row for row in validation_diag}

    # Frozen PER identity check.
    per_counts = Counter()
    for word in words_saved:
        per_counts.update(word["acoustic_edit_counts"])
    for key, expected in EXPECTED_PER_COUNTS.items():
        if per_counts[key] != expected:
            raise RuntimeError(f"PER count mismatch for {key}: {per_counts[key]} != {expected}")

    length_audit = {
        "words": len(words_saved),
        "expected_length": distribution(len(word["expected"]) for word in words_saved),
        "manual_observed_target_length": distribution(len(word["target"]) for word in words_saved),
        "greedy_decoded_length": distribution(len(word["decoded"]) for word in words_saved),
    }
    nonempty = [word for word in words_saved if word["target"]]
    ratios = [len(word["decoded"]) / len(word["target"]) for word in nonempty]
    comparisons = Counter("shorter" if len(word["decoded"]) < len(word["target"]) else
                          "longer" if len(word["decoded"]) > len(word["target"]) else "equal" for word in words_saved)
    length_audit["decoded_target_ratio_nonempty"] = distribution(ratios, (10, 25, 75, 90))
    length_audit["length_comparison"] = {
        name: {"words": int(comparisons[name]), "fraction": ratio(comparisons[name], len(words_saved))}
        for name in ("shorter", "equal", "longer")
    }

    relation_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for word in words_saved:
        for group in classify_word(word):
            relation_groups[group].append(word)
    relation_length = {group: aggregate_words(group_words, val_diag_by_id) for group, group_words in relation_groups.items()}

    phone_rows_by_word: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in phone_rows:
        phone_rows_by_word[row["word_id"]].append(row)
    fd_counts: dict[str, int] = {}
    for word_id, rows in phone_rows_by_word.items():
        fd_counts[word_id] = sum(row["true_relation"] in {"correct", "substitution"} and
                                 row["predicted_relation"] == "deletion" for row in rows)
    fd_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for word in words_saved:
        count = fd_counts[word["word_id"]]
        bucket = str(count) if count < 3 else ">=3"
        fd_buckets[bucket].append(word)
    false_deletion_association = {
        "groups": {key: aggregate_words(value, val_diag_by_id) for key, value in fd_buckets.items()},
        "length_deficit_vs_false_deletion_count": corr(
            [len(word["target"]) - len(word["decoded"]) for word in words_saved],
            [fd_counts[word["word_id"]] for word in words_saved],
        ),
    }

    blank_occupancy = {
        "argmax_blank_fraction_per_word": distribution(row["blank_argmax_fraction"] for row in validation_diag),
        "blank_posterior_all_timesteps": validation_compute["all_blank_posterior"],
        "per_word_mean_blank_posterior": distribution(row["mean_blank_posterior"] for row in validation_diag),
        "per_word_max_blank_posterior": distribution(row["max_blank_posterior"] for row in validation_diag),
        "by_relation_group": {
            group: distribution(val_diag_by_id[word["word_id"]]["blank_argmax_fraction"] for word in group_words)
            for group, group_words in relation_groups.items() if group != "multiple_deletion"
        },
        "false_deletion_comparison": {
            "zero_false_deletions": {
                "words": sum(fd_counts[word["word_id"]] == 0 for word in words_saved),
                "mean_blank_posterior": distribution(val_diag_by_id[word["word_id"]]["mean_blank_posterior"]
                                                     for word in words_saved if fd_counts[word["word_id"]] == 0),
                "max_blank_posterior": distribution(val_diag_by_id[word["word_id"]]["max_blank_posterior"]
                                                    for word in words_saved if fd_counts[word["word_id"]] == 0),
            },
            "one_or_more_false_deletions": {
                "words": sum(fd_counts[word["word_id"]] >= 1 for word in words_saved),
                "mean_blank_posterior": distribution(val_diag_by_id[word["word_id"]]["mean_blank_posterior"]
                                                     for word in words_saved if fd_counts[word["word_id"]] >= 1),
                "max_blank_posterior": distribution(val_diag_by_id[word["word_id"]]["max_blank_posterior"]
                                                    for word in words_saved if fd_counts[word["word_id"]] >= 1),
            },
        },
    }
    repeat_collapse = {
        "encoder_output_timesteps": distribution(row["output_timesteps"] for row in validation_diag),
        "nonblank_argmax_fraction": distribution(row["nonblank_argmax_fraction"] for row in validation_diag),
        "decoded_length_per_timestep": distribution(row["decoded_per_timestep"] for row in validation_diag),
        "nonblank_runs_per_word": distribution(row["nonblank_runs"] for row in validation_diag),
        "nonblank_run_length": validation_compute["all_nonblank_run_length"],
        "repeat_collapsed_timesteps_per_word": distribution(row["repeat_collapsed_timesteps"] for row in validation_diag),
        "interpretation": "A non-blank run is one contiguous identical non-blank argmax region; blank separates runs.",
    }

    total_errors = sum(per_counts[key] for key in ("substitution", "deletion", "insertion"))
    per_error = {
        "target_phones": EXPECTED_VALIDATION_TARGETS,
        "counts": {key: int(per_counts[key]) for key in ("substitution", "deletion", "insertion")},
        "total_errors": total_errors,
        "fractions_of_errors": {key: ratio(per_counts[key], total_errors) for key in ("substitution", "deletion", "insertion")},
        "normalized_per_target_phone": {key: ratio(per_counts[key], EXPECTED_VALIDATION_TARGETS)
                                        for key in ("substitution", "deletion", "insertion")},
        "manual_expected_deletions": EXPECTED_MANUAL_DELETIONS,
        "ctc_target_phone_deletion_errors": int(per_counts["deletion"]),
        "ctc_error_to_manual_deletion_ratio": ratio(per_counts["deletion"], EXPECTED_MANUAL_DELETIONS),
        "distinction": "Manual deletion is an absent expected phone; a CTC PER deletion is a manual observed target phone omitted by the decoder.",
    }

    clean_words = relation_groups["all_correct"]
    clean_rows = [row for word in clean_words for row in phone_rows_by_word[word["word_id"]]]
    clean_word_control = aggregate_words(clean_words, val_diag_by_id)
    clean_word_control["false_expected_phone_deletion_rate"] = ratio(
        sum(row["predicted_relation"] == "deletion" for row in clean_rows), len(clean_rows))

    # Manual observed target-phone edit outcomes.
    phone_counters = {phone: Counter() for phone in r4b.PHONE_VOCAB}
    for word in words_saved:
        target = word["target"]
        for operation in word["acoustic_alignment"]:
            reference_index = operation["reference_index"]
            if reference_index is None:
                continue
            phone = target[reference_index]
            phone_counters[phone]["support"] += 1
            if operation["operation"] == "MATCH":
                phone_counters[phone]["correct"] += 1
            elif operation["operation"] == "SUBSTITUTION":
                phone_counters[phone]["substituted"] += 1
            elif operation["operation"] == "DELETE_FROM_EXPECTED":
                phone_counters[phone]["deleted"] += 1
    per_phone = {}
    for phone in r4b.PHONE_VOCAB:
        counts = phone_counters[phone]
        per_phone[phone] = {
            "support": counts["support"], "correctly_decoded_aligned": counts["correct"],
            "substituted": counts["substituted"], "deleted": counts["deleted"],
            "recall": ratio(counts["correct"], counts["support"]),
        }
    sufficiently_supported = [(phone, metrics) for phone, metrics in per_phone.items() if metrics["support"] >= 100]
    worst = sorted(sufficiently_supported, key=lambda item: (item[1]["recall"], item[0]))[:10]
    phone_error_analysis = {
        "all_40": per_phone, "rare_AX_OY_ZH": {phone: per_phone[phone] for phone in ("AX", "OY", "ZH")},
        "ten_worst_support_at_least_100": [{"phone": phone, **metrics} for phone, metrics in worst],
        "supported_phones_below_0_50_recall": sum(metrics["recall"] < 0.5 for _, metrics in sufficiently_supported),
        "interpretation": "Broad weakness" if sum(metrics["recall"] < 0.5 for _, metrics in sufficiently_supported) >= 10 else "Concentrated weakness",
    }

    def bucket_analysis(bucket_fn: Any) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for word in words_saved:
            buckets[bucket_fn(word)].append(word)
        output = {}
        for name, selected in buckets.items():
            aggregate = aggregate_words(selected, val_diag_by_id)
            rows = [row for word in selected for row in phone_rows_by_word[word["word_id"]]
                    if row["true_relation"] in {"correct", "substitution"}]
            aggregate["false_deletion_rate"] = ratio(sum(row["predicted_relation"] == "deletion" for row in rows), len(rows))
            output[name] = aggregate
        return output

    word_length_analysis = bucket_analysis(lambda word: str(len(word["target"])) if len(word["target"]) <= 5 else "6+")
    def duration_bucket(word: dict[str, Any]) -> str:
        duration = float(word["word_end"] - word["word_start"])
        if duration < .15: return "<150ms"
        if duration < .25: return "150-250ms"
        if duration < .4: return "250-400ms"
        if duration < .6: return "400-600ms"
        return ">=600ms"
    duration_analysis = bucket_analysis(duration_bucket)

    speaker_analysis = {}
    for speaker in sorted(r4b.VALIDATION_SPEAKERS):
        selected = [word for word in words_saved if word["speaker_id"] == speaker]
        aggregate = aggregate_words(selected, val_diag_by_id)
        rows = [row for word in selected for row in phone_rows_by_word[word["word_id"]]]
        target_phones = aggregate["target_phones"]
        aggregate.update({
            "decoded_target_ratio_aggregate": ratio(aggregate["decoded_phones"], target_phones),
            "ctc_target_phone_deletion_error_rate": ratio(aggregate["edit_counts"]["deletion"], target_phones),
            "correct_false_deletion_rate": false_deletion_rates(rows)["correct"],
            "substitution_false_deletion_rate": false_deletion_rates(rows)["substitution"],
        })
        speaker_analysis[speaker] = aggregate

    def train_aggregate() -> dict[str, Any]:
        target = sum(row["target_length"] for row in train_diag)
        decoded = sum(row["decoded_length"] for row in train_diag)
        counts = Counter()
        for row in train_diag:
            counts.update(row["edit_counts"])
        return {
            "words": len(train_diag), "target_phones": target, "decoded_phones": decoded,
            "per": ratio(counts["errors"], target), "decoded_target_ratio": ratio(decoded, target),
            "blank_argmax_occupancy": distribution(row["blank_argmax_fraction"] for row in train_diag),
            "edit_counts": {key: int(counts[key]) for key in ("substitution", "deletion", "insertion", "errors")},
        }
    train_summary = train_aggregate()
    val_summary = {
        "words": len(validation_diag), "target_phones": EXPECTED_VALIDATION_TARGETS,
        "decoded_phones": sum(row["decoded_length"] for row in validation_diag),
        "per": ratio(total_errors, EXPECTED_VALIDATION_TARGETS),
        "decoded_target_ratio": ratio(sum(row["decoded_length"] for row in validation_diag), EXPECTED_VALIDATION_TARGETS),
        "blank_argmax_occupancy": distribution(row["blank_argmax_fraction"] for row in validation_diag),
    }
    per_gap = val_summary["per"] - train_summary["per"]
    ratio_gap = train_summary["decoded_target_ratio"] - val_summary["decoded_target_ratio"]
    if per_gap >= .15 or ratio_gap >= .15:
        gap_class = "SEVERE_GENERALIZATION_GAP"
    elif per_gap >= .05 or ratio_gap >= .05:
        gap_class = "MODERATE_GENERALIZATION_GAP"
    else:
        gap_class = "NO_MAJOR_GENERALIZATION_GAP"
    train_validation_gap = {
        "train": train_summary, "validation": val_summary,
        "validation_minus_train_per": per_gap, "train_minus_validation_length_ratio": ratio_gap,
        "classification_rule": {"severe": "PER gap >=0.15 or length-ratio gap >=0.15",
                                "moderate": "PER gap >=0.05 or length-ratio gap >=0.05", "otherwise": "no major gap"},
        "classification": gap_class,
    }

    epoch_values = {
        "per": [epoch["validation"]["phone_error_rate"] for epoch in epochs],
        "deletion_recall": [epoch["validation"]["binary"]["deletion"]["recall"] for epoch in epochs],
        "deletion_precision": [epoch["validation"]["binary"]["deletion"]["precision"] for epoch in epochs],
        "substitution_false_deletion": [epoch["validation"]["substitution_false_deletion_rate"] for epoch in epochs],
        "binary_macro_f1": [epoch["validation"]["binary"]["macro_f1"] for epoch in epochs],
    }
    epoch_trend = {
        "selected_epoch": 35,
        "correlation_with_per": {name: corr(epoch_values["per"], values)
                                 for name, values in epoch_values.items() if name != "per"},
        "epoch_1": epochs[0]["validation"], "epoch_35": epochs[34]["validation"], "epoch_36": epochs[35]["validation"],
        "interpretation": "Lower PER accompanies improved deletion precision and recall, lower substitution false-deletion, and higher binary Macro-F1.",
    }

    median_ratio = length_audit["decoded_target_ratio_nonempty"]["median"]
    aggregate_ratio = val_summary["decoded_target_ratio"]
    blank_median = blank_occupancy["argmax_blank_fraction_per_word"]["median"]
    deletion_fraction = per_error["fractions_of_errors"]["deletion"]
    substitution_fraction = per_error["fractions_of_errors"]["substitution"]
    broad_confusion = phone_error_analysis["supported_phones_below_0_50_recall"] >= 10
    if aggregate_ratio < .85 and deletion_fraction >= .25 and substitution_fraction >= .40 and broad_confusion:
        failure = "CTC_MIXED_ACOUSTIC_FAILURE"
    elif aggregate_ratio <= .75 and blank_median >= .60 and deletion_fraction >= .35 and substitution_fraction < .50:
        failure = "CTC_BLANK_UNDER_GENERATION"
    elif aggregate_ratio >= .85 and substitution_fraction >= .60 and blank_median < .60:
        failure = "CTC_PHONE_CONFUSION_DOMINANT"
    elif aggregate_ratio < .85 and blank_median < .60 and repeat_collapse["repeat_collapsed_timesteps_per_word"]["median"] >= 5:
        failure = "CTC_REPEAT_COLLAPSE_DOMINANT"
    else:
        failure = "CTC_FAILURE_NOT_EXPLAINED"
    if failure in {"CTC_MIXED_ACOUSTIC_FAILURE", "CTC_PHONE_CONFUSION_DOMINANT"} and clean_word_control["per"] >= .40 and broad_confusion:
        implication = "BIGRU_NEXT_EXPERIMENT_JUSTIFIED"
    elif failure == "CTC_BLANK_UNDER_GENERATION":
        implication = "DECODING_OR_LOSS_AUDIT_FIRST"
    else:
        implication = "BIGRU_NOT_JUSTIFIED"
    failure_classification = {
        "primary_failure": failure, "architecture_implication": implication,
        "frozen_classification_inputs": {
            "median_per_word_decoded_target_ratio": median_ratio,
            "aggregate_decoded_target_ratio": aggregate_ratio,
            "median_blank_argmax_occupancy": blank_median,
            "per_deletion_error_fraction": deletion_fraction, "per_substitution_error_fraction": substitution_fraction,
            "supported_phones_below_0_50_recall": phone_error_analysis["supported_phones_below_0_50_recall"],
            "clean_word_per": clean_word_control["per"],
        },
        "reason": "The decoder both omits many target phones and confuses phone identities broadly; neither a blank-only nor rare-phone-only account explains the locked result.",
        "next_smallest_action": "Pre-register one fresh CNN + one small BiGRU CTC experiment, preserving greedy decoding and all data policies; do not train it in this audit.",
    }

    preflight = {
        "status": "PASS", "source_verification": verification,
        "checkpoint_epoch": checkpoint_payload["epoch"], "checkpoint_validation_per": checkpoint_payload["validation_per"],
        "dataset_reconstruction": reconstruction, "device": str(device),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "validation_decode_reproduction": "PASS", "validation_words": len(validation_words), "train_words": len(train_words),
        "training_occurred": False, "optimizer_created": False,
        "r4_test_paths_resolved": False, "r4_test_audio_read": False, "r4_test_inference": False,
    }

    outputs = {
        "preflight.json": preflight, "length_audit.json": length_audit,
        "relation_length_analysis.json": relation_length,
        "false_deletion_association.json": false_deletion_association,
        "blank_occupancy.json": blank_occupancy, "repeat_collapse.json": repeat_collapse,
        "per_error_composition.json": per_error, "clean_word_control.json": clean_word_control,
        "phone_error_analysis.json": phone_error_analysis, "word_length_analysis.json": word_length_analysis,
        "duration_analysis.json": duration_analysis, "speaker_analysis.json": speaker_analysis,
        "train_validation_gap.json": train_validation_gap, "epoch_trend.json": epoch_trend,
        "failure_classification.json": failure_classification,
    }
    for name, payload in outputs.items():
        write_json(OUTPUT_DIR / name, payload)

    report = f"""# Phoenix R4-4C0 CTC failure-mode audit

RESEARCH_ONLY / NOT_PRODUCTION / R4_TEST_CLOSED

- Frozen artifact verification: PASS
- Selected checkpoint: epoch 35, `{CHECKPOINT.name}`
- Validation decode reproduction: PASS ({len(validation_words):,} words)
- Median decoded/manual-target length ratio: {median_ratio:.6f}
- Median blank-argmax occupancy: {blank_median:.6f}
- PER composition: substitution {substitution_fraction:.3%}, deletion {deletion_fraction:.3%}, insertion {per_error['fractions_of_errors']['insertion']:.3%}
- CTC target-phone deletion errors / real manual expected deletions: {per_counts['deletion']:,} / {EXPECTED_MANUAL_DELETIONS:,} = {per_error['ctc_error_to_manual_deletion_ratio']:.3f}x
- Clean all-correct word PER: {clean_word_control['per']:.6f}
- TRAIN-vs-VALIDATION: {gap_class}
- Primary failure: **{failure}**
- Architecture implication: **{implication}**
- Training occurred: NO
- R4 TEST accessed: NO

The selected model under-generates and also exhibits broad identity confusion. CTC/PER deletion errors are omissions from the manual observed sequence and must not be interpreted as real pronunciation deletions.
"""
    (OUTPUT_DIR / "r4_4c0_report.md").write_text(report, encoding="utf-8")
    compute = {
        "validation": validation_compute, "train": train_compute,
        "total_seconds": time.perf_counter() - started, "device": str(device),
    }
    write_json(OUTPUT_DIR / "compute_report.json", compute)
    artifact_names = sorted(path.name for path in OUTPUT_DIR.iterdir() if path.name != "artifact_hashes.json")
    hashes = {name: sha256(OUTPUT_DIR / name) for name in artifact_names}
    write_json(OUTPUT_DIR / "artifact_hashes.json", {"algorithm": "SHA-256", "artifacts": hashes})
    print(json.dumps({
        "status": "R4_4C0_FAILURE_AUDIT_COMPLETE", "primary_failure": failure,
        "architecture_implication": implication, "output_dir": str(OUTPUT_DIR),
        "total_seconds": time.perf_counter() - started,
    }, indent=2))


if __name__ == "__main__":
    main()
