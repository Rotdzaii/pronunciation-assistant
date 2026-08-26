from __future__ import annotations

import argparse
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
import run_l2_arctic_observed_phone_r3_1c_36epochs as r3c  # noqa: E402


REPO_ROOT = r3a.REPO_ROOT
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r3_1d_observed_phone_seed42_48epochs"
R3C_EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r3_1c_observed_phone_seed42_36epochs"
EPOCHS = 48
CHECKPOINT_NAME = "R3_1D_observed_phone_40class_seed42_best_validation_macro_f1.pt"

REPRODUCTION_TOLERANCES = dict(r3c.REPRODUCTION_TOLERANCES)
PLATEAU_MACRO_F1_GAIN_MAX = 0.010
PLATEAU_TOP1_GAIN_MAX = 0.005
PLATEAU_SUBSTITUTION_MACRO_F1_GAIN_MAX = 0.010
MEANINGFUL_LOSS_DROP = 0.010
OVERFIT_MACRO_F1_DROP = 0.020
OVERFIT_LOSS_INCREASE = 0.030
NEAR_BUDGET_EPOCHS = (46, 47, 48)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked R3-1D 48-epoch observed-phone baseline.")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def config_payload(summary: dict[str, Any], class_weights: list[float]) -> dict[str, Any]:
    payload = r3c.config_payload(summary, class_weights)
    payload.pop("epoch_24_reproduction_gate", None)
    payload.pop("trend_rules_preregistered", None)
    payload.update({
        "experiment": "R3-1D 40-class observed-phone final training-budget extension to 48 epochs",
        "parent_experiment": "R3-1C",
        "hypothesis": "The unchanged observed-phone model still benefits from training beyond epoch 36",
        "only_changed_variable": {"max_epochs": {"from": 36, "to": 48}},
        "epochs": EPOCHS,
        "fresh_random_initialization": True,
        "loaded_r3_1c_weights": False,
        "epoch_36_reproduction_gate": {
            "reference": "r3_1c_observed_phone_seed42_36epochs/epoch_metrics.csv",
            "absolute_tolerances": REPRODUCTION_TOLERANCES,
            "policy": "stop before epoch 37 if any epoch 1-36 field exceeds tolerance",
        },
        "trend_rules_preregistered": {
            "practical_plateau_exact": {
                "selected_macro_f1_improvement_lt": PLATEAU_MACRO_F1_GAIN_MAX,
                "selected_top1_improvement_lt": PLATEAU_TOP1_GAIN_MAX,
                "selected_substitution_macro_f1_improvement_lt": PLATEAU_SUBSTITUTION_MACRO_F1_GAIN_MAX,
                "all_required": True,
            },
            "budget_still_limiting": {
                "selected_epoch_in": list(NEAR_BUDGET_EPOCHS),
                "selected_macro_f1_improvement_min": PLATEAU_MACRO_F1_GAIN_MAX,
                "late_validation_loss_drop_min": MEANINGFUL_LOSS_DROP,
                "clear_validation_deterioration": False,
                "all_required": True,
            },
            "overfitting": {
                "selected_epoch_at_most": 42,
                "last4_mean_macro_f1_below_best_by": OVERFIT_MACRO_F1_DROP,
                "last4_mean_validation_loss_above_best_by": OVERFIT_LOSS_INCREASE,
                "training_loss_must_continue_decreasing": True,
                "all_required": True,
            },
            "conservative_fallback": (
                "PRACTICAL_PLATEAU_REACHED when neither registered evidence for sustained overfitting "
                "nor every TRAINING_BUDGET_STILL_LIMITING condition is present"
            ),
        },
    })
    return payload


def reproduction_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    reference_path = R3C_EXPERIMENT_DIR / "epoch_metrics.csv"
    if not reference_path.is_file():
        raise FileNotFoundError(f"R3-1C trajectory is required: {reference_path}")
    reference = r3b.read_epoch_csv(reference_path)
    if len(reference) != 36 or len(records) != 36:
        raise RuntimeError(f"Reproduction comparison requires 36+36 epochs, got {len(reference)}+{len(records)}")
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


def comparison_report(selected: dict[str, Any]) -> dict[str, Any]:
    old = json.loads((R3C_EXPERIMENT_DIR / "selected_validation_report.json").read_text(encoding="utf-8"))["metrics"]
    old_per_class = json.loads((R3C_EXPERIMENT_DIR / "per_class_metrics.json").read_text(encoding="utf-8"))
    per_class = {
        phone: {
            "r3_1c_recall": old_per_class[phone]["recall"],
            "r3_1d_recall": selected["per_class"][phone]["recall"],
            "recall_delta": selected["per_class"][phone]["recall"] - old_per_class[phone]["recall"],
            "r3_1c_f1": old_per_class[phone]["f1"],
            "r3_1d_f1": selected["per_class"][phone]["f1"],
            "f1_delta": selected["per_class"][phone]["f1"] - old_per_class[phone]["f1"],
        }
        for phone in r3a.PHONE_VOCAB
    }
    return {
        "overall": {
            field: {"r3_1c": old[field], "r3_1d": selected[field], "delta": selected[field] - old[field]}
            for field in ("loss", "accuracy", "macro_f1", "balanced_accuracy", "macro_precision", "macro_recall", "top3_accuracy")
        },
        "substitution_origin_macro_f1": {
            "r3_1c": old["relation_source"]["substitution"]["macro_f1"],
            "r3_1d": selected["relation_source"]["substitution"]["macro_f1"],
            "delta": (
                selected["relation_source"]["substitution"]["macro_f1"]
                - old["relation_source"]["substitution"]["macro_f1"]
            ),
        },
        "per_class": per_class,
        "focus_phones": {phone: per_class[phone] for phone in ("AX", "D", "TH", "G", "OY", "ZH")},
        "zero_recall_r3_1c": [phone for phone, item in old_per_class.items() if item["recall"] == 0.0],
        "zero_recall_r3_1d": [phone for phone, item in selected["per_class"].items() if item["recall"] == 0.0],
        "top_confusion_pairs_r3_1c": r3b.confusion_pairs(old["confusion_matrix"]),
        "top_confusion_pairs_r3_1d": r3b.confusion_pairs(selected["confusion_matrix"]),
    }


def trend_report(records: list[dict[str, Any]], selected_epoch: int, comparison: dict[str, Any]) -> dict[str, Any]:
    compact = [r3c.compact_epoch(record) for record in records]
    early, late = compact[:36], compact[36:]
    best_early = max(early, key=lambda item: (item["macro_f1"], item["substitution_origin_top1"], -item["epoch"]))
    best_late = max(late, key=lambda item: (item["macro_f1"], item["substitution_origin_top1"], -item["epoch"]))
    best_overall = max(compact, key=lambda item: (item["macro_f1"], item["substitution_origin_top1"], -item["epoch"]))
    best_loss = min(compact, key=lambda item: item["validation_loss"])
    last4 = compact[-4:]
    last4_mean_f1 = float(np.mean([item["macro_f1"] for item in last4]))
    last4_mean_loss = float(np.mean([item["validation_loss"] for item in last4]))
    deltas = {
        "macro_f1": comparison["overall"]["macro_f1"]["delta"],
        "top1_accuracy": comparison["overall"]["accuracy"]["delta"],
        "substitution_origin_macro_f1": comparison["substitution_origin_macro_f1"]["delta"],
        "validation_loss": comparison["overall"]["loss"]["delta"],
    }
    plateau_exact = (
        deltas["macro_f1"] < PLATEAU_MACRO_F1_GAIN_MAX
        and deltas["top1_accuracy"] < PLATEAU_TOP1_GAIN_MAX
        and deltas["substitution_origin_macro_f1"] < PLATEAU_SUBSTITUTION_MACRO_F1_GAIN_MAX
    )
    loss_meaningfully_decreases = (
        min(item["validation_loss"] for item in late)
        <= comparison["overall"]["loss"]["r3_1c"] - MEANINGFUL_LOSS_DROP
    )
    overfitting = (
        selected_epoch <= 42
        and last4_mean_f1 <= best_overall["macro_f1"] - OVERFIT_MACRO_F1_DROP
        and last4_mean_loss >= best_loss["validation_loss"] + OVERFIT_LOSS_INCREASE
        and compact[-1]["train_loss"] < best_overall["train_loss"]
    )
    still_limiting = (
        selected_epoch in NEAR_BUDGET_EPOCHS
        and deltas["macro_f1"] >= PLATEAU_MACRO_F1_GAIN_MAX
        and loss_meaningfully_decreases
        and not overfitting
    )
    if overfitting:
        decision = "OVERFITTING_DETECTED"
        decision_basis = "registered sustained-degradation rule"
    elif still_limiting:
        decision = "TRAINING_BUDGET_STILL_LIMITING"
        decision_basis = "all registered still-limiting conditions"
    elif plateau_exact:
        decision = "PRACTICAL_PLATEAU_REACHED"
        decision_basis = "all three registered practical-gain ceilings"
    else:
        decision = "PRACTICAL_PLATEAU_REACHED"
        decision_basis = "conservative fallback: insufficient joint evidence for continued budget limitation"
    return {
        "preregistered_thresholds": {
            "plateau_macro_f1_gain_lt": PLATEAU_MACRO_F1_GAIN_MAX,
            "plateau_top1_gain_lt": PLATEAU_TOP1_GAIN_MAX,
            "plateau_substitution_macro_f1_gain_lt": PLATEAU_SUBSTITUTION_MACRO_F1_GAIN_MAX,
            "meaningful_validation_loss_drop": MEANINGFUL_LOSS_DROP,
            "overfit_macro_f1_drop": OVERFIT_MACRO_F1_DROP,
            "overfit_loss_increase": OVERFIT_LOSS_INCREASE,
            "near_budget_epochs": list(NEAR_BUDGET_EPOCHS),
        },
        "epochs_36_through_48": compact[35:],
        "best_epochs_1_36": best_early,
        "best_epochs_37_48": best_late,
        "best_overall": best_overall,
        "best_validation_loss": best_loss,
        "selected_delta_vs_r3_1c": deltas,
        "last4_mean_macro_f1": last4_mean_f1,
        "last4_mean_validation_loss": last4_mean_loss,
        "plateau_exact_rule_satisfied": plateau_exact,
        "loss_meaningfully_decreases": loss_meaningfully_decreases,
        "overfitting_rule_triggered": overfitting,
        "still_limiting_rule_satisfied": still_limiting,
        "selected_epoch_near_48": selected_epoch in NEAR_BUDGET_EPOCHS,
        "trend_decision": decision,
        "decision_basis": decision_basis,
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
    old_weights = json.loads((R3C_EXPERIMENT_DIR / "class_weights.json").read_text(encoding="utf-8"))["weights"]
    weight_delta = max(abs(class_weights[i] - old_weights[r3a.PHONE_VOCAB[i]]) for i in r3a.ALL_LABELS)
    if weight_delta > 1e-12:
        raise RuntimeError(f"R3-1C class weights did not reproduce: max delta={weight_delta}")
    print(json.dumps({
        "dataset": summary, "audio": audio_preflight, "feature_shape": list(probe_feature.shape),
        "device": str(device), "class_weight_max_delta_vs_r3_1c": weight_delta,
    }, indent=2))
    if args.preflight_only:
        print("R3-1D preflight PASS; TEST audio not accessed; no training/artifacts.")
        return 0

    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite existing experiment directory: {EXPERIMENT_DIR}")
    EXPERIMENT_DIR.mkdir(parents=True)
    r3a.EXPERIMENT_DIR = EXPERIMENT_DIR
    config = config_payload(summary, class_weights)
    r3a.write_json(EXPERIMENT_DIR / "config.json", config)
    environment = r3a.environment_payload(audio_root, device, summary["dataset_sha256"])
    environment.update({"experiment": "R3-1D", "test_audio_accessed": False})
    r3a.write_json(EXPERIMENT_DIR / "environment.json", environment)
    r3a.write_json(EXPERIMENT_DIR / "preflight.json", {
        "dataset": summary, "audio": audio_preflight, "feature_shape": list(probe_feature.shape),
        "class_weight_max_delta_vs_r3_1c": weight_delta, "status": "PASS", "test_audio_accessed": False,
    })
    r3a.write_json(EXPERIMENT_DIR / "phone_vocab.json", {
        "class_to_index": r3a.PHONE_TO_ID, "index_to_class": list(r3a.PHONE_VOCAB),
    })
    r3a.write_json(EXPERIMENT_DIR / "class_weights.json", {
        "formula": "N_train / (40 * train_count[c])", "train_rows": len(split_rows["train"]),
        "weights": {r3a.PHONE_VOCAB[i]: class_weights[i] for i in r3a.ALL_LABELS},
        "train_support": {r3a.PHONE_VOCAB[i]: train_counts[i] for i in r3a.ALL_LABELS},
        "maximum_absolute_delta_vs_r3_1c": weight_delta,
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
        records.append({
            "epoch": epoch, "train_loss": train_loss_sum / train_seen, "validation": validation_metrics,
            "epoch_seconds": time.perf_counter() - epoch_started,
        })
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
                "config": config, "fresh_random_initialization": True, "loaded_r3_1c_weights": False,
            }, checkpoint_path)
        print(
            f"epoch={epoch} val_loss={validation_metrics['loss']:.6f} top1={validation_metrics['accuracy']:.6f} "
            f"macro_f1={macro_f1:.6f} balanced={validation_metrics['balanced_accuracy']:.6f} "
            f"top3={validation_metrics['top3_accuracy']:.6f} sub_top1={substitution_accuracy:.6f} "
            f"sub_macro_f1={validation_metrics['relation_source']['substitution']['macro_f1']:.6f}"
        )
        if epoch == 36:
            reproduction = reproduction_report(records)
            r3a.write_json(EXPERIMENT_DIR / "reproduction_epochs_1_36.json", reproduction)
            if reproduction["status"] != "PASS":
                elapsed = time.perf_counter() - run_started
                r3a.write_json(EXPERIMENT_DIR / "final_status.json", {
                    "status": "R3_1D_REPRODUCTION_FAIL", "trend_decision": "PRACTICAL_PLATEAU_REACHED",
                    "test_opened": False, "test_eligible": False, "completed_epochs": 36,
                    "failures": reproduction["failures"], "elapsed_seconds": elapsed,
                })
                print("R3_1D_REPRODUCTION_FAIL; stopped before epoch 37; TEST not accessed")
                return 0
            print("R3-1D epochs 1-36 reproduction PASS; continuing to epoch 37")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_metrics, selected_details = r3a.evaluate(
        model, validation_dataset, split_rows["validation"], criterion, device, substitution_supported, collect_details=True
    )
    elapsed = time.perf_counter() - run_started
    outputs = r3a.save_selected_outputs(
        selected_metrics, selected_details, split_rows["validation"], summary, best_epoch, elapsed, checkpoint_path
    )
    comparison = comparison_report(selected_metrics)
    trend = trend_report(records, best_epoch, comparison)
    downstream_current = json.loads((EXPERIMENT_DIR / "downstream_binary_diagnostic.json").read_text(encoding="utf-8"))
    downstream_old = json.loads((R3C_EXPERIMENT_DIR / "downstream_binary_diagnostic.json").read_text(encoding="utf-8"))
    downstream_comparison = {
        "r3_1c": downstream_old,
        "r3_1d": downstream_current,
        "delta": {
            "macro_f1": downstream_current["macro_f1"] - downstream_old["macro_f1"],
            "substitution_precision": downstream_current["substitution"]["precision"] - downstream_old["substitution"]["precision"],
            "substitution_recall": downstream_current["substitution"]["recall"] - downstream_old["substitution"]["recall"],
            "substitution_f1": downstream_current["substitution"]["f1"] - downstream_old["substitution"]["f1"],
        },
        "diagnostic_only": True,
        "used_for_checkpoint_selection": False,
    }
    r3a.write_json(EXPERIMENT_DIR / "comparison_with_r3_1c.json", comparison)
    r3a.write_json(EXPERIMENT_DIR / "trend_epochs_36_48.json", trend)
    r3a.write_json(EXPERIMENT_DIR / "downstream_comparison_with_r3_1c.json", downstream_comparison)
    status = "R3_1D_PASS_VALIDATION" if outputs["gates"]["all_pass"] else "R3_1D_VALIDATION_FAIL"
    final = {
        "status": status, "trend_decision": trend["trend_decision"],
        "test_opened": False, "test_eligible": bool(outputs["gates"]["all_pass"]),
        "selected_epoch": best_epoch, "selected_epoch_near_48": best_epoch in NEAR_BUDGET_EPOCHS,
        "training_and_validation_seconds": elapsed,
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "reproduction_epochs_1_36": "PASS", "gates": outputs["gates"],
    }
    r3a.write_json(EXPERIMENT_DIR / "final_status.json", final)
    r3a.write_json(EXPERIMENT_DIR / "model_metadata.json", {
        "name": "SmallPronunciationCNNAttention", "task": "R3-1D observed-phone 40-class final budget extension",
        "markers": ["RESEARCH_ONLY", "NOT_PRODUCTION", "NOT_RUNTIME_CONNECTED"],
        "fresh_random_initialization": True, "loaded_r3_1c_weights": False,
        "only_changed_variable": "max_epochs 36 -> 48", "selected_epoch": best_epoch,
        "checkpoint": checkpoint_path.name, "validation_macro_f1": selected_metrics["macro_f1"],
        "class_to_index": r3a.PHONE_TO_ID, "test_opened": False, "test_eligible": final["test_eligible"],
    })
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
