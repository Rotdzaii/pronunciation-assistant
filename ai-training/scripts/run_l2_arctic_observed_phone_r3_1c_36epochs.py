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
import run_l2_arctic_observed_phone_r3_1b_24epochs as r3b  # noqa: E402


REPO_ROOT = r3a.REPO_ROOT
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r3_1c_observed_phone_seed42_36epochs"
R3B_EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r3_1b_observed_phone_seed42_24epochs"
EPOCHS = 36
CHECKPOINT_NAME = "R3_1C_observed_phone_40class_seed42_best_validation_macro_f1.pt"

REPRODUCTION_TOLERANCES = dict(r3b.REPRODUCTION_TOLERANCES)
MEANINGFUL_MACRO_F1_GAIN = 0.005
MEANINGFUL_LOSS_DROP = 0.010
SUBSTITUTION_REGRESSION_TOLERANCE = 0.020
OVERFIT_MACRO_F1_DROP = 0.020
OVERFIT_LOSS_INCREASE = 0.030
NEAR_BUDGET_EPOCHS = (34, 35, 36)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked R3-1C 36-epoch observed-phone baseline.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def config_payload(summary: dict[str, Any], class_weights: list[float]) -> dict[str, Any]:
    payload = r3b.config_payload(summary, class_weights)
    payload.pop("epoch_12_reproduction_gate", None)
    payload.update({
        "experiment": "R3-1C 40-class observed-phone extended training to 36 epochs",
        "parent_experiment": "R3-1B",
        "hypothesis": "The unchanged observed-phone model still benefits from training beyond epoch 24",
        "only_changed_variable": {"max_epochs": {"from": 24, "to": 36}},
        "epochs": EPOCHS,
        "fresh_random_initialization": True,
        "loaded_r3_1b_weights": False,
        "epoch_24_reproduction_gate": {
            "reference": "r3_1b_observed_phone_seed42_24epochs/epoch_metrics.csv",
            "absolute_tolerances": REPRODUCTION_TOLERANCES,
            "policy": "stop before epoch 25 if any epoch 1-24 field exceeds tolerance",
        },
        "trend_rules_preregistered": {
            "continued_improvement": {
                "late_best_macro_f1_gain_min": MEANINGFUL_MACRO_F1_GAIN,
                "late_min_validation_loss_drop_min": MEANINGFUL_LOSS_DROP,
                "late_best_substitution_top1_regression_max": SUBSTITUTION_REGRESSION_TOLERANCE,
            },
            "overfitting": {
                "selected_epoch_at_most": 30,
                "last3_mean_macro_f1_below_best_by": OVERFIT_MACRO_F1_DROP,
                "last3_mean_validation_loss_above_best_by": OVERFIT_LOSS_INCREASE,
                "training_loss_must_continue_decreasing": True,
            },
            "budget_still_limiting_if_selected_epoch_in": list(NEAR_BUDGET_EPOCHS),
            "otherwise": "PLATEAU / TRAINING_PLATEAU_REACHED",
        },
    })
    return payload


def reproduction_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    reference_path = R3B_EXPERIMENT_DIR / "epoch_metrics.csv"
    if not reference_path.is_file():
        raise FileNotFoundError(f"R3-1B trajectory is required: {reference_path}")
    reference = r3b.read_epoch_csv(reference_path)
    if len(reference) != 24 or len(records) != 24:
        raise RuntimeError(f"Reproduction comparison requires 24+24 epochs, got {len(reference)}+{len(records)}")
    comparisons, failures = [], []
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
                failures.append(f"epoch {record['epoch']} {field}: delta={delta:.9f} > {REPRODUCTION_TOLERANCES[field]:.9f}")
        comparisons.append({"epoch": record["epoch"], "reference": ref, "actual": actual, "absolute_delta": deltas})
    return {
        "status": "PASS" if not failures else "FAIL",
        "reference": str(reference_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "tolerances": REPRODUCTION_TOLERANCES,
        "maximum_absolute_deltas": maximum_deltas,
        "failures": failures,
        "epochs": comparisons,
    }


def compact_epoch(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record["validation"]
    return {
        "epoch": record["epoch"], "train_loss": record["train_loss"], "validation_loss": metrics["loss"],
        "top1_accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"], "macro_precision": metrics["macro_precision"],
        "top3_accuracy": metrics["top3_accuracy"],
        "correct_origin_top1": metrics["relation_source"]["correct"]["accuracy"],
        "correct_origin_macro_f1": metrics["relation_source"]["correct"]["macro_f1"],
        "substitution_origin_top1": metrics["relation_source"]["substitution"]["accuracy"],
        "substitution_origin_macro_f1": metrics["relation_source"]["substitution"]["macro_f1"],
    }


def trend_report(records: list[dict[str, Any]], selected_epoch: int) -> dict[str, Any]:
    compact = [compact_epoch(record) for record in records]
    early, late = compact[:24], compact[24:]
    best_early = max(early, key=lambda item: (item["macro_f1"], item["substitution_origin_top1"], -item["epoch"]))
    best_late = max(late, key=lambda item: (item["macro_f1"], item["substitution_origin_top1"], -item["epoch"]))
    all_best = max(compact, key=lambda item: (item["macro_f1"], item["substitution_origin_top1"], -item["epoch"]))
    best_loss_item = min(compact, key=lambda item: item["validation_loss"])
    last3 = compact[-3:]
    last3_mean_f1 = float(np.mean([item["macro_f1"] for item in last3]))
    last3_mean_loss = float(np.mean([item["validation_loss"] for item in last3]))
    meaningful_late_gain = (
        best_late["macro_f1"] - best_early["macro_f1"] >= MEANINGFUL_MACRO_F1_GAIN
        and min(item["validation_loss"] for item in late) <= min(item["validation_loss"] for item in early) - MEANINGFUL_LOSS_DROP
        and best_late["substitution_origin_top1"] >= best_early["substitution_origin_top1"] - SUBSTITUTION_REGRESSION_TOLERANCE
    )
    overfitting = (
        selected_epoch <= 30
        and last3_mean_f1 <= all_best["macro_f1"] - OVERFIT_MACRO_F1_DROP
        and last3_mean_loss >= best_loss_item["validation_loss"] + OVERFIT_LOSS_INCREASE
        and compact[-1]["train_loss"] < all_best["train_loss"]
    )
    if overfitting:
        curve = "OVERFITTING"
        flag = "OVERFITTING_DETECTED"
    elif meaningful_late_gain:
        curve = "CONTINUED_IMPROVEMENT"
        flag = "TRAINING_BUDGET_STILL_LIMITING" if selected_epoch in NEAR_BUDGET_EPOCHS else "TRAINING_PLATEAU_REACHED"
    else:
        curve = "PLATEAU"
        flag = "TRAINING_PLATEAU_REACHED"
    return {
        "preregistered_thresholds": {
            "meaningful_macro_f1_gain": MEANINGFUL_MACRO_F1_GAIN, "meaningful_loss_drop": MEANINGFUL_LOSS_DROP,
            "substitution_regression_tolerance": SUBSTITUTION_REGRESSION_TOLERANCE,
            "overfit_macro_f1_drop": OVERFIT_MACRO_F1_DROP, "overfit_loss_increase": OVERFIT_LOSS_INCREASE,
            "near_budget_epochs": list(NEAR_BUDGET_EPOCHS),
        },
        "epochs_24_through_36": compact[23:],
        "best_epochs_1_24": best_early,
        "best_epochs_25_36": best_late,
        "best_overall": all_best,
        "best_validation_loss": best_loss_item,
        "epoch_24_to_36_delta": {
            field: compact[-1][field] - compact[23][field]
            for field in ("train_loss", "validation_loss", "top1_accuracy", "macro_f1", "balanced_accuracy", "substitution_origin_top1")
        },
        "last3_mean_macro_f1": last3_mean_f1,
        "last3_mean_validation_loss": last3_mean_loss,
        "meaningful_late_gain": meaningful_late_gain,
        "overfitting_rule_triggered": overfitting,
        "curve_classification": curve,
        "trend_flag": flag,
        "selected_epoch_near_36": selected_epoch in NEAR_BUDGET_EPOCHS,
    }


def comparison_report(selected: dict[str, Any]) -> dict[str, Any]:
    old = json.loads((R3B_EXPERIMENT_DIR / "selected_validation_report.json").read_text(encoding="utf-8"))["metrics"]
    old_per_class = json.loads((R3B_EXPERIMENT_DIR / "per_class_metrics.json").read_text(encoding="utf-8"))
    per_class = {
        phone: {
            "r3_1b_recall": old_per_class[phone]["recall"], "r3_1c_recall": selected["per_class"][phone]["recall"],
            "recall_delta": selected["per_class"][phone]["recall"] - old_per_class[phone]["recall"],
            "r3_1b_f1": old_per_class[phone]["f1"], "r3_1c_f1": selected["per_class"][phone]["f1"],
            "f1_delta": selected["per_class"][phone]["f1"] - old_per_class[phone]["f1"],
        }
        for phone in r3a.PHONE_VOCAB
    }
    return {
        "overall": {
            field: {"r3_1b": old[field], "r3_1c": selected[field], "delta": selected[field] - old[field]}
            for field in ("loss", "accuracy", "macro_f1", "balanced_accuracy", "macro_precision", "macro_recall", "top3_accuracy")
        },
        "per_class": per_class,
        "focus_phones": {phone: per_class[phone] for phone in ("AX", "D", "TH", "G", "OY", "ZH")},
        "zero_recall_r3_1b": [phone for phone, item in old_per_class.items() if item["recall"] == 0.0],
        "zero_recall_r3_1c": [phone for phone, item in selected["per_class"].items() if item["recall"] == 0.0],
        "top_confusion_pairs_r3_1b": r3b.confusion_pairs(old["confusion_matrix"]),
        "top_confusion_pairs_r3_1c": r3b.confusion_pairs(selected["confusion_matrix"]),
    }


def main() -> int:
    args = parse_args()
    r3a.set_seed(r3a.SEED)
    audio_root = r3a.require_audio_root()
    split_rows, summary = r3a.load_and_validate_rows(audio_root)
    audio_preflight = r3a.preflight_audio(split_rows["train"] + split_rows["validation"])
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
        print("R3-1C preflight PASS; TEST audio not accessed; no training/artifacts.")
        return 0

    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite existing experiment directory: {EXPERIMENT_DIR}")
    EXPERIMENT_DIR.mkdir(parents=True)
    r3a.EXPERIMENT_DIR = EXPERIMENT_DIR
    config = config_payload(summary, class_weights)
    r3a.write_json(EXPERIMENT_DIR / "config.json", config)
    environment = r3a.environment_payload(audio_root, device, summary["dataset_sha256"])
    environment.update({"experiment": "R3-1C", "test_audio_accessed": False})
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
        record = {"epoch": epoch, "train_loss": train_loss_sum / train_seen, "validation": validation_metrics, "epoch_seconds": time.perf_counter() - epoch_started}
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
                "config": config, "fresh_random_initialization": True, "loaded_r3_1b_weights": False,
            }, checkpoint_path)
        print(
            f"epoch={epoch} val_loss={validation_metrics['loss']:.6f} top1={validation_metrics['accuracy']:.6f} "
            f"macro_f1={macro_f1:.6f} balanced={validation_metrics['balanced_accuracy']:.6f} "
            f"top3={validation_metrics['top3_accuracy']:.6f} sub_top1={substitution_accuracy:.6f} "
            f"sub_macro_f1={validation_metrics['relation_source']['substitution']['macro_f1']:.6f}"
        )
        if epoch == 24:
            reproduction = reproduction_report(records)
            r3a.write_json(EXPERIMENT_DIR / "reproduction_epochs_1_24.json", reproduction)
            if reproduction["status"] != "PASS":
                elapsed = time.perf_counter() - run_started
                r3a.write_json(EXPERIMENT_DIR / "final_status.json", {
                    "status": "R3_1C_REPRODUCTION_FAIL", "trend_flag": "TRAINING_PLATEAU_REACHED",
                    "test_opened": False, "test_eligible": False, "completed_epochs": 24,
                    "failures": reproduction["failures"], "elapsed_seconds": elapsed,
                })
                print("R3_1C_REPRODUCTION_FAIL; stopped before epoch 25; TEST not accessed")
                return 0
            print("R3-1C epochs 1-24 reproduction PASS; continuing to epoch 25")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_metrics, selected_details = r3a.evaluate(
        model, validation_dataset, split_rows["validation"], criterion, device, substitution_supported, collect_details=True
    )
    elapsed = time.perf_counter() - run_started
    outputs = r3a.save_selected_outputs(selected_metrics, selected_details, split_rows["validation"], summary, best_epoch, elapsed, checkpoint_path)
    comparison = comparison_report(selected_metrics)
    trend = trend_report(records, best_epoch)
    r3a.write_json(EXPERIMENT_DIR / "comparison_with_r3_1b.json", comparison)
    r3a.write_json(EXPERIMENT_DIR / "trend_epochs_24_36.json", trend)
    status = "R3_1C_PASS_VALIDATION" if outputs["gates"]["all_pass"] else "R3_1C_VALIDATION_FAIL"
    final = {
        "status": status, "trend_flag": trend["trend_flag"], "curve_classification": trend["curve_classification"],
        "test_opened": False, "test_eligible": bool(outputs["gates"]["all_pass"]),
        "selected_epoch": best_epoch, "selected_epoch_near_36": best_epoch in NEAR_BUDGET_EPOCHS,
        "training_and_validation_seconds": elapsed,
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "reproduction_epochs_1_24": "PASS", "gates": outputs["gates"],
    }
    r3a.write_json(EXPERIMENT_DIR / "final_status.json", final)
    r3a.write_json(EXPERIMENT_DIR / "model_metadata.json", {
        "name": "SmallPronunciationCNNAttention", "task": "R3-1C observed-phone 40-class extended training",
        "markers": ["RESEARCH_ONLY", "NOT_PRODUCTION", "NOT_RUNTIME_CONNECTED"],
        "fresh_random_initialization": True, "loaded_r3_1b_weights": False,
        "only_changed_variable": "max_epochs 24 -> 36", "selected_epoch": best_epoch,
        "checkpoint": checkpoint_path.name, "validation_macro_f1": selected_metrics["macro_f1"],
        "class_to_index": r3a.PHONE_TO_ID, "test_opened": False, "test_eligible": final["test_eligible"],
    })
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
