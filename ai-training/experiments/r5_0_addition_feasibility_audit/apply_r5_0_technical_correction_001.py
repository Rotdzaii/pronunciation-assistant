from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
DRIVER = HERE / "run_r5_0_addition_feasibility_audit.py"
CONTRACT = HERE / "R5_0_TECHNICAL_CORRECTION_001.json"
V4 = HERE.parents[2] / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
EXPECTED_DRIVER = "F7086A6D2AE13881799B696487BFC2C11072423529FCA29E443732D59F821EB7"
EXPECTED_PREREG = "14CBFADAC0BE35D53C01DC966A030A71F24EC4D86EA88384BC449544CE12AEF7"
TRAIN = frozenset(("BWC", "EBVS", "HJK", "NCC", "NJS", "PNV", "RRBI", "TLV", "TNI", "YBAA", "YKWK", "ZHAA"))
TEST = frozenset(("ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_driver():
    spec = importlib.util.spec_from_file_location("r5_original_driver", DRIVER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load original driver")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if sha256(DRIVER) != EXPECTED_DRIVER:
        raise RuntimeError("Original driver identity mismatch")
    if sha256(HERE / "R5_0_PREREGISTRATION.json") != EXPECTED_PREREG:
        raise RuntimeError("Preregistration identity mismatch")
    audio_root = Path(os.environ["L2_ARCTIC_ROOT"]).resolve()
    original = load_driver()
    source_summary, detail_rows, _ = original.scan_source()
    events_by_word, _ = original.map_additions(detail_rows, audio_root)
    words, _ = original.build_train_words(audio_root, events_by_word)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoded, compute = original.infer_train(words, device)
    baseline, reproduced_hallucination = original.run_ctc_audit(words, decoded)
    expected = json.loads(CONTRACT.read_text(encoding="utf-8"))["acceptance"]
    word = baseline["word_level"]
    event = baseline["exact_insertion_event"]
    actual_word = {name: word[name] for name in ("TP", "FP", "FN", "TN")}
    actual_event = {name: event[name] for name in ("tp", "fp", "fn")}
    if actual_word != expected["primary_word_confusion_must_equal"]:
        raise RuntimeError(f"Primary word metrics failed reproduction: {actual_word}")
    if actual_event != expected["exact_event_confusion_must_equal"]:
        raise RuntimeError(f"Event metrics failed reproduction: {actual_event}")
    if reproduced_hallucination["ADDITION_INSERTION_RATE_DELTA"] != expected["addition_insertion_rate_delta_must_equal_float64"]:
        raise RuntimeError("Delta failed exact float64 reproduction")

    correct_only_phone = Counter()
    correct_only_insertions = 0
    for word_record, hypothesis in zip(words, decoded):
        true_events = word_record["true_addition_events"]
        relations = Counter(word_record["relations"])
        is_correct_only = (
            not true_events and word_record["substitution"] == 0 and word_record["deletion"] == 0
            and all(name in {"correct", "non_speech"} or count == 0 for name, count in relations.items())
        )
        if not is_correct_only:
            continue
        predicted = original.insertion_events(word_record["expected_ids"], hypothesis)
        correct_only_insertions += len(predicted)
        correct_only_phone.update(event["phone"] for event in predicted)
    if correct_only_insertions != reproduced_hallucination["insertion_edits"]:
        raise RuntimeError("Correct-only insertion total mismatch")

    train_phone = Counter()
    with V4.open(encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            speaker = source["speaker_id"]
            if speaker in TEST:
                continue
            if speaker in TRAIN and source["relation"] == "addition":
                train_phone[source["observed_phone_canonical"]] += 1
    if sum(train_phone.values()) != expected["clean_train_addition_phone_counts_must_sum"]:
        raise RuntimeError("TRAIN clean addition phone total mismatch")

    initial_hallucination = json.loads((HERE / "r5_0_ctc_hallucination_audit.initial.json").read_text(encoding="utf-8"))
    initial_hallucination["most_common_hallucinated_inserted_phones_all_eligible_train"] = initial_hallucination.pop(
        "most_common_hallucinated_inserted_phones_all_train"
    )
    initial_hallucination["most_common_hallucinated_inserted_phones_correct_only"] = correct_only_phone.most_common(20)
    initial_hallucination["technical_correction"] = "R5_0_TECHNICAL_CORRECTION_001"
    write_json(HERE / "r5_0_ctc_hallucination_audit.json", initial_hallucination)

    distribution = json.loads((HERE / "r5_0_addition_distribution_train.initial.json").read_text(encoding="utf-8"))
    mapped_phone = distribution["by_added_phone"]
    distribution["by_added_phone_manual_word_mapped"] = mapped_phone
    distribution["by_added_phone"] = dict(sorted(train_phone.items()))
    distribution["by_word_manual_word_mapped"] = distribution.pop("by_word")
    ordered = sorted(train_phone.items(), key=lambda item: (item[1], item[0]))
    distribution["bottom_supported_phones"] = ordered[:10]
    distribution["top_supported_phones"] = list(reversed(ordered[-10:]))
    distribution["phone_distribution_event_total"] = sum(train_phone.values())
    distribution["manual_word_mapped_phone_event_total"] = sum(mapped_phone.values())
    distribution["technical_correction"] = "R5_0_TECHNICAL_CORRECTION_001"
    write_json(HERE / "r5_0_addition_distribution_train.json", distribution)

    audit = {
        "status": "PASS",
        "correction_contract_sha256": sha256(CONTRACT),
        "original_driver_sha256": sha256(DRIVER),
        "primary_metrics_reproduced_exactly": True,
        "word_confusion": actual_word,
        "event_confusion": actual_event,
        "addition_insertion_rate_delta": reproduced_hallucination["ADDITION_INSERTION_RATE_DELTA"],
        "correct_only_insertion_edits": correct_only_insertions,
        "correct_only_phone_counts": dict(correct_only_phone),
        "all_clean_train_addition_phone_events": sum(train_phone.values()),
        "train_inference_repeated_for_reporting_correction": True,
        "training": False,
        "validation_inference": False,
        "validation_performance_consumed": False,
        "test_paths_resolved": False,
        "test_audio_accessed": False,
        "test_inference": False,
        "test_performance_consumed": False,
        "compute": compute,
    }
    write_json(HERE / "r5_0_technical_correction_001_audit.json", audit)
    print(json.dumps({"status": "PASS", "correct_only_top": correct_only_phone.most_common(10), "train_phone_total": sum(train_phone.values())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
