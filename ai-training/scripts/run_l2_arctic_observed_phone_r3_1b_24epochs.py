from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_l2_arctic_observed_phone_r3_1a as r3a  # noqa: E402


REPO_ROOT = r3a.REPO_ROOT
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r3_1b_observed_phone_seed42_24epochs"
R3A_EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r3_1a_observed_phone_seed42"
EPOCHS = 24
CHECKPOINT_NAME = "R3_1B_observed_phone_40class_seed42_best_validation_macro_f1.pt"

# Pre-registered before R3-1B training. Every epoch 1-12 must stay within these
# absolute deltas from the saved R3-1A trajectory before epoch 13 is allowed.
REPRODUCTION_TOLERANCES = {
    "validation_loss": 0.05,
    "top1_accuracy": 0.01,
    "macro_f1": 0.01,
    "balanced_accuracy": 0.01,
    "correct_origin_top1": 0.01,
    "correct_origin_macro_f1": 0.01,
    "substitution_origin_top1": 0.01,
    "substitution_origin_macro_f1": 0.01,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked R3-1B 24-epoch observed-phone baseline.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def config_payload(summary: dict[str, Any], class_weights: list[float]) -> dict[str, Any]:
    payload = r3a.config_payload(summary, class_weights)
    payload.update({
        "experiment": "R3-1B 40-class observed-phone extended-training feasibility",
        "parent_experiment": "R3-1A",
        "hypothesis": "R3-1A stopped while validation was still improving",
        "only_changed_variable": {"max_epochs": {"from": 12, "to": 24}},
        "epochs": EPOCHS,
        "fresh_random_initialization": True,
        "loaded_r3_1a_weights": False,
        "epoch_12_reproduction_gate": {
            "reference": "r3_1a_observed_phone_seed42/epoch_metrics.csv",
            "absolute_tolerances": REPRODUCTION_TOLERANCES,
            "policy": "stop before epoch 13 if any epoch 1-12 field exceeds tolerance",
        },
    })
    return payload


def read_epoch_csv(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: float(value) if key != "epoch" else int(value) for key, value in row.items() if key in {"epoch", *REPRODUCTION_TOLERANCES}}
            for row in csv.DictReader(handle)
        ]


def reproduction_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    reference_path = R3A_EXPERIMENT_DIR / "epoch_metrics.csv"
    if not reference_path.is_file():
        raise FileNotFoundError(f"R3-1A trajectory is required: {reference_path}")
    reference = read_epoch_csv(reference_path)
    if len(reference) != 12 or len(records) != 12:
        raise RuntimeError(f"Reproduction comparison requires 12+12 epochs, got {len(reference)}+{len(records)}")
    comparisons = []
    failures = []
    maximum_deltas = {field: 0.0 for field in REPRODUCTION_TOLERANCES}
    for ref, record in zip(reference, records):
        metrics = record["validation"]
        actual = {
            "validation_loss": metrics["loss"],
            "top1_accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "correct_origin_top1": metrics["relation_source"]["correct"]["accuracy"],
            "correct_origin_macro_f1": metrics["relation_source"]["correct"]["macro_f1"],
            "substitution_origin_top1": metrics["relation_source"]["substitution"]["accuracy"],
            "substitution_origin_macro_f1": metrics["relation_source"]["substitution"]["macro_f1"],
        }
        deltas = {field: abs(actual[field] - ref[field]) for field in REPRODUCTION_TOLERANCES}
        for field, delta in deltas.items():
            maximum_deltas[field] = max(maximum_deltas[field], delta)
            if delta > REPRODUCTION_TOLERANCES[field] + 1e-12:
                failures.append(
                    f"epoch {record['epoch']} {field}: delta={delta:.9f} > {REPRODUCTION_TOLERANCES[field]:.9f}"
                )
        comparisons.append({"epoch": record["epoch"], "reference": ref, "actual": actual, "absolute_delta": deltas})
    return {
        "status": "PASS" if not failures else "FAIL",
        "reference": str(reference_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "tolerances": REPRODUCTION_TOLERANCES,
        "maximum_absolute_deltas": maximum_deltas,
        "failures": failures,
        "epochs": comparisons,
    }


def confusion_pairs(matrix: list[list[int]], limit: int = 20) -> list[dict[str, Any]]:
    pairs = []
    for actual, row in enumerate(matrix):
        support = sum(row)
        for predicted, count in enumerate(row):
            if actual != predicted and count:
                pairs.append({
                    "actual": r3a.PHONE_VOCAB[actual], "predicted": r3a.PHONE_VOCAB[predicted],
                    "count": count, "actual_class_rate": count / support,
                })
    return sorted(pairs, key=lambda item: (-item["count"], item["actual"], item["predicted"]))[:limit]


def comparison_and_trend(records: list[dict[str, Any]], selected: dict[str, Any], best_epoch: int) -> tuple[dict[str, Any], dict[str, Any]]:
    old_report = json.loads((R3A_EXPERIMENT_DIR / "selected_validation_report.json").read_text(encoding="utf-8"))
    old_metrics = old_report["metrics"]
    old_per_class = json.loads((R3A_EXPERIMENT_DIR / "per_class_metrics.json").read_text(encoding="utf-8"))
    per_class_changes = {
        phone: {
            "r3_1a_recall": old_per_class[phone]["recall"], "r3_1b_recall": selected["per_class"][phone]["recall"],
            "recall_delta": selected["per_class"][phone]["recall"] - old_per_class[phone]["recall"],
            "r3_1a_f1": old_per_class[phone]["f1"], "r3_1b_f1": selected["per_class"][phone]["f1"],
            "f1_delta": selected["per_class"][phone]["f1"] - old_per_class[phone]["f1"],
        }
        for phone in r3a.PHONE_VOCAB
    }
    comparison = {
        "selected_epoch": best_epoch,
        "overall": {
            field: {"r3_1a": old_metrics[field], "r3_1b": selected[field], "delta": selected[field] - old_metrics[field]}
            for field in ("loss", "accuracy", "macro_f1", "balanced_accuracy", "macro_precision", "macro_recall", "top3_accuracy")
        },
        "per_class": per_class_changes,
        "zero_recall_r3_1a": [phone for phone, item in old_per_class.items() if item["recall"] == 0.0],
        "zero_recall_r3_1b": [phone for phone, item in selected["per_class"].items() if item["recall"] == 0.0],
        "focus_phones": {phone: per_class_changes[phone] for phone in ("AX", "OY", "ZH", "D", "TH", "G")},
        "top_confusion_pairs_r3_1b": confusion_pairs(selected["confusion_matrix"]),
    }
    epoch_metrics = []
    for record in records[11:]:
        metrics = record["validation"]
        epoch_metrics.append({
            "epoch": record["epoch"], "validation_loss": metrics["loss"], "top1_accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"], "balanced_accuracy": metrics["balanced_accuracy"],
            "substitution_origin_top1": metrics["relation_source"]["substitution"]["accuracy"],
        })
    epoch12, epoch24 = epoch_metrics[0], epoch_metrics[-1]
    best_after_12 = max(epoch_metrics[1:], key=lambda item: item["macro_f1"])
    if best_epoch >= 22 and epoch24["macro_f1"] > epoch12["macro_f1"] and epoch24["validation_loss"] < epoch12["validation_loss"]:
        classification = "CONTINUED_IMPROVEMENT"
    elif min(item["validation_loss"] for item in epoch_metrics) < epoch24["validation_loss"] and best_epoch < 22:
        classification = "BEGINNING_OVERFIT"
    else:
        classification = "PLATEAU"
    trend = {
        "epochs_12_through_24": epoch_metrics,
        "epoch_12_to_24_delta": {
            field: epoch24[field] - epoch12[field]
            for field in ("validation_loss", "top1_accuracy", "macro_f1", "balanced_accuracy", "substitution_origin_top1")
        },
        "best_macro_f1_after_epoch_12": best_after_12,
        "classification": classification,
        "selected_epoch_is_24": best_epoch == 24,
        "training_budget_still_limiting": best_epoch == 24,
    }
    return comparison, trend


def main() -> int:
    args = parse_args()
    r3a.set_seed(r3a.SEED)
    audio_root = r3a.require_audio_root()
    split_rows, summary = r3a.load_and_validate_rows(audio_root)
    train_validation_rows = split_rows["train"] + split_rows["validation"]
    audio_preflight = r3a.preflight_audio(train_validation_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_row = split_rows["train"][0]
    probe_audio, _ = r3a.librosa.load(probe_row["_audio_path"], sr=r3a.SAMPLE_RATE, mono=True)
    probe_window = r3a.centered_window(np.asarray(probe_audio, dtype=np.float32), probe_row["_start"], probe_row["_end"])
    with torch.no_grad():
        probe_feature = r3a.FixedLogMel().to(device)(torch.from_numpy(probe_window).unsqueeze(0).to(device))
    if probe_window.shape != (r3a.WINDOW_SAMPLES,) or tuple(probe_feature.shape[1:]) != r3a.EXPECTED_FEATURE_SHAPE:
        raise RuntimeError(f"Feature preflight failed: {probe_window.shape}, {tuple(probe_feature.shape)}")
    train_counts = r3a.Counter(row["_target"] for row in split_rows["train"])
    class_weights = [len(split_rows["train"]) / (len(r3a.PHONE_VOCAB) * train_counts[index]) for index in r3a.ALL_LABELS]
    if not all(math.isfinite(weight) and weight > 0 for weight in class_weights):
        raise RuntimeError("Class weights are not finite and positive")
    print(json.dumps({"dataset": summary, "audio": audio_preflight, "feature_shape": list(probe_feature.shape), "device": str(device)}, indent=2))
    if args.preflight_only:
        print("R3-1B preflight PASS; TEST audio not accessed; no training/artifacts.")
        return 0

    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite existing experiment directory: {EXPERIMENT_DIR}")
    EXPERIMENT_DIR.mkdir(parents=True)
    r3a.EXPERIMENT_DIR = EXPERIMENT_DIR
    config = config_payload(summary, class_weights)
    r3a.write_json(EXPERIMENT_DIR / "config.json", config)
    environment = r3a.environment_payload(audio_root, device, summary["dataset_sha256"])
    environment.update({"experiment": "R3-1B", "test_audio_accessed": False})
    r3a.write_json(EXPERIMENT_DIR / "environment.json", environment)
    r3a.write_json(EXPERIMENT_DIR / "preflight.json", {
        "dataset": summary, "audio": audio_preflight, "feature_shape": list(probe_feature.shape),
        "status": "PASS", "test_audio_accessed": False,
    })
    r3a.write_json(EXPERIMENT_DIR / "phone_vocab.json", {"class_to_index": r3a.PHONE_TO_ID, "index_to_class": list(r3a.PHONE_VOCAB)})
    r3a.write_json(EXPERIMENT_DIR / "class_weights.json", {
        "formula": "N_train / (40 * train_count[c])", "train_rows": len(split_rows["train"]),
        "weights": {r3a.PHONE_VOCAB[i]: class_weights[i] for i in r3a.ALL_LABELS},
        "train_support": {r3a.PHONE_VOCAB[i]: train_counts[i] for i in r3a.ALL_LABELS},
    })

    run_started = time.perf_counter()
    train_dataset = r3a.materialize_audio_features(split_rows["train"], device, "train")
    validation_dataset = r3a.materialize_audio_features(split_rows["validation"], device, "validation")
    generator = torch.Generator().manual_seed(r3a.SEED)
    train_loader = DataLoader(
        train_dataset, batch_size=r3a.BATCH_SIZE, shuffle=True, generator=generator,
        num_workers=r3a.NUM_WORKERS, pin_memory=device.type == "cuda",
    )
    model = r3a.SmallPronunciationCNNAttention().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=r3a.LEARNING_RATE, weight_decay=0.0)
    substitution_supported = [r3a.PHONE_TO_ID[phone] for phone in summary["substitution_supported_classes"]]
    checkpoint_path = EXPERIMENT_DIR / CHECKPOINT_NAME
    records: list[dict[str, Any]] = []
    best_epoch, best_macro_f1, best_substitution_accuracy = 0, -1.0, -1.0

    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_sum, train_seen = 0.0, 0
        for features, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} train"):
            features, targets = features.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(targets)
            train_seen += len(targets)
        validation_metrics, _ = r3a.evaluate(
            model, validation_dataset, split_rows["validation"], criterion, device, substitution_supported
        )
        record = {
            "epoch": epoch, "train_loss": train_loss_sum / train_seen, "validation": validation_metrics,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        records.append(record)
        r3a.save_epoch_outputs(records)
        macro_f1 = validation_metrics["macro_f1"]
        substitution_accuracy = validation_metrics["relation_source"]["substitution"]["accuracy"]
        better = macro_f1 > best_macro_f1 + 1e-12 or (
            abs(macro_f1 - best_macro_f1) <= 1e-12 and substitution_accuracy > best_substitution_accuracy + 1e-12
        )
        if better:
            best_epoch, best_macro_f1, best_substitution_accuracy = epoch, macro_f1, substitution_accuracy
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch, "validation_macro_f1": macro_f1,
                "validation_substitution_origin_accuracy": substitution_accuracy, "class_to_index": r3a.PHONE_TO_ID,
                "config": config, "fresh_random_initialization": True, "loaded_r3_1a_weights": False,
            }, checkpoint_path)
        print(
            f"epoch={epoch} val_loss={validation_metrics['loss']:.6f} top1={validation_metrics['accuracy']:.6f} "
            f"macro_f1={macro_f1:.6f} balanced={validation_metrics['balanced_accuracy']:.6f} "
            f"top3={validation_metrics['top3_accuracy']:.6f} sub_top1={substitution_accuracy:.6f} "
            f"sub_macro_f1={validation_metrics['relation_source']['substitution']['macro_f1']:.6f}"
        )
        if epoch == 12:
            reproduction = reproduction_report(records)
            r3a.write_json(EXPERIMENT_DIR / "reproduction_epochs_1_12.json", reproduction)
            if reproduction["status"] != "PASS":
                elapsed = time.perf_counter() - run_started
                r3a.write_json(EXPERIMENT_DIR / "final_status.json", {
                    "status": "R3_1B_REPRODUCTION_FAIL", "test_opened": False, "test_eligible": False,
                    "completed_epochs": 12, "failures": reproduction["failures"], "elapsed_seconds": elapsed,
                })
                print("R3_1B_REPRODUCTION_FAIL; stopped before epoch 13; TEST not accessed")
                return 0
            print("R3-1B epochs 1-12 reproduction PASS; continuing to epoch 13")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_metrics, selected_details = r3a.evaluate(
        model, validation_dataset, split_rows["validation"], criterion, device, substitution_supported, collect_details=True
    )
    elapsed = time.perf_counter() - run_started
    outputs = r3a.save_selected_outputs(
        selected_metrics, selected_details, split_rows["validation"], summary, best_epoch, elapsed, checkpoint_path
    )
    comparison, trend = comparison_and_trend(records, selected_metrics, best_epoch)
    r3a.write_json(EXPERIMENT_DIR / "comparison_with_r3_1a.json", comparison)
    r3a.write_json(EXPERIMENT_DIR / "trend_epochs_12_24.json", trend)
    status = "R3_1B_PASS_VALIDATION" if outputs["gates"]["all_pass"] else "R3_1B_VALIDATION_FAIL"
    final = {
        "status": status, "test_opened": False, "test_eligible": bool(outputs["gates"]["all_pass"]),
        "selected_epoch": best_epoch, "selected_epoch_is_24": best_epoch == 24,
        "training_budget_still_limiting": best_epoch == 24, "training_and_validation_seconds": elapsed,
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "reproduction_epochs_1_12": "PASS", "gates": outputs["gates"],
    }
    r3a.write_json(EXPERIMENT_DIR / "final_status.json", final)
    r3a.write_json(EXPERIMENT_DIR / "model_metadata.json", {
        "name": "SmallPronunciationCNNAttention", "task": "R3-1B observed-phone 40-class extended training",
        "markers": ["RESEARCH_ONLY", "NOT_PRODUCTION", "NOT_RUNTIME_CONNECTED"],
        "fresh_random_initialization": True, "loaded_r3_1a_weights": False, "only_changed_variable": "max_epochs 12 -> 24",
        "selected_epoch": best_epoch, "checkpoint": checkpoint_path.name,
        "validation_macro_f1": selected_metrics["macro_f1"], "class_to_index": r3a.PHONE_TO_ID,
        "test_opened": False, "test_eligible": final["test_eligible"],
    })
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
