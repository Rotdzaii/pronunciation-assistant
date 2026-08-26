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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_l2_arctic_observed_phone_r3_1a as r3  # noqa: E402
import run_r4_3a_word_sequence_design_audit as r43a  # noqa: E402


REPO_ROOT = r3.REPO_ROOT
EXPERIMENT_DIR = REPO_ROOT / "ai-training/experiments/r4_4b_ctc_sequence_seed42"
PREREG_DIR = REPO_ROOT / "ai-training/experiments/r4_4a_ctc_sequence_feasibility"
PREREG_JSON = PREREG_DIR / "r4_4b_preregistered_design.json"
PREREG_MD = PREREG_DIR / "r4_4b_preregistered_design.md"
MATCHED_CSV = PREREG_DIR / "validation_word_eligible_matched_control.csv"
V4_PATH = REPO_ROOT / "ai-training/datasets/l2-arctic/metadata/all_speakers_expected_observed_v4.csv"
CHECKPOINT_NAME = "R4_4B_ctc_phone_sequence_seed42_best_validation_per.pt"
EXPECTED_HASHES = {
    "preregistered_json": "2AB7129149C0A67ED718F5CA087A57E363CA8B034CD8618E60645E207E7A842D",
    "preregistered_markdown": "14ADB9C11F9D055A1C6311841A085CF2BBEC3519112B416DFB695C2473479560",
    "matched_control": "D933F674743DA06CC8FAB425CEBF81D9C78505E1BDB4A90204DDB2E1A15B4798",
    "v4": "160CF1813716CFE598A6C913B38A1A8492E67DE6E3F779BF121B94352DB3F54D",
}
TRAIN_SPEAKERS = frozenset(r3.TRAIN_SPEAKERS)
VALIDATION_SPEAKERS = frozenset(r3.VALIDATION_SPEAKERS)
TEST_SPEAKERS = frozenset(("ASI", "ERMS", "SKA", "THV", "TXHC", "YDCK"))
EXPECTED_WORDS = {"train": 16_259, "validation": 7_728}
PHONE_VOCAB = tuple(r3.PHONE_VOCAB)
PHONE_TO_ID = dict(r3.PHONE_TO_ID)
BLANK = 40
SEED = 42
EPOCHS = 36
BATCH_SIZE = 8
PREPROCESS_BATCH_SIZE = 64
SAMPLE_RATE = 16_000
N_MELS = 64
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
GRAD_CLIP = 5.0
LEARNING_RATE = 1e-4
TIE_TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def set_determinism() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def verify_sources() -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "preregistered_json": PREREG_JSON,
        "preregistered_markdown": PREREG_MD,
        "matched_control": MATCHED_CSV,
        "v4": V4_PATH,
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": EXPECTED_HASHES[name], "actual": digest}
        for name, digest in actual.items() if digest != EXPECTED_HASHES[name]
    }
    if mismatches:
        raise RuntimeError(f"R4_4B_PREREGISTRATION_VERIFICATION_FAIL: {mismatches}")
    prereg = json.loads(PREREG_JSON.read_text(encoding="utf-8"))
    return prereg, {"expected": EXPECTED_HASHES, "actual": actual, "status": "PASS"}


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def target_minimum_steps(target: list[str]) -> int:
    return len(target) + sum(left == right for left, right in zip(target, target[1:]))


def feature_frames(sample_count: int) -> int:
    return sample_count // HOP_LENGTH + 1


def encoder_steps(frame_count: int) -> int:
    return frame_count // 2


class WordWaveStore:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.audio: np.ndarray | None = None

    def load(self, path: Path) -> np.ndarray:
        if self.path != path:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError(f"Unreadable audio: {path}")
            self.path, self.audio = path, audio
        assert self.audio is not None
        return self.audio


def extract_word_waveform(store: WordWaveStore, word: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    audio = store.load(word["audio_path"])
    requested_start = int(round(float(word["mfa_start"]) * SAMPLE_RATE))
    requested_end = int(round(float(word["mfa_end"]) * SAMPLE_RATE))
    if requested_end <= requested_start:
        raise RuntimeError(f"Non-positive word span: {word['word_id']}")
    source_start = max(0, requested_start)
    source_end = min(len(audio), requested_end)
    if source_end <= source_start:
        raise RuntimeError(f"Word span outside audio: {word['word_id']}")
    waveform = audio[source_start:source_end].copy()
    left_missing = source_start - requested_start
    right_missing = requested_end - source_end
    if left_missing or right_missing:
        waveform = np.pad(waveform, (left_missing, right_missing), mode="constant")
    if waveform.size != requested_end - requested_start or not np.isfinite(waveform).all():
        raise RuntimeError(f"Invalid extracted word waveform: {word['word_id']}")
    return waveform, {
        "requested_samples": requested_end - requested_start,
        "left_edge_padding_samples": left_missing,
        "right_edge_padding_samples": right_missing,
    }


class WordLogMel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            window_fn=torch.hann_window,
            power=2.0,
            center=True,
            pad_mode="constant",
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        power = self.mel(waveform)
        log_power = 10.0 * torch.log10(torch.clamp(power, min=1e-10))
        reference = log_power.amax(dim=(-2, -1), keepdim=True)
        return torch.clamp(log_power - reference, min=-80.0)


class WordCTCModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.05),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(96, 41)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features)
        sequence = encoded.mean(dim=2).transpose(1, 2)
        return self.classifier(self.dropout(sequence))


def load_words(audio_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r43a.AUDIT_SPEAKERS = TRAIN_SPEAKERS | VALIDATION_SPEAKERS
    records, reconstruction = r43a.build_word_records(audio_root)
    usable = [word for word in records if word["usable"]]
    counts = {split: sum(word["split"] == split for word in usable) for split in ("train", "validation")}
    if counts != EXPECTED_WORDS:
        raise RuntimeError(f"Frozen word reconstruction mismatch: {counts}")
    if any(word["speaker_id"] in TEST_SPEAKERS for word in usable):
        raise RuntimeError("TEST speaker leaked into R4-4B population")
    for word in usable:
        word["target_ids"] = [PHONE_TO_ID[phone] for phone in word["observed"]]
        word["expected_ids"] = [PHONE_TO_ID[phone] for phone in word["expected"]]
    return usable, {"usable_words": counts, "reconstruction": reconstruction}


def materialize_features(
    words: list[dict[str, Any]], device: torch.device, split_name: str
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    extractor = WordLogMel().to(device).eval()
    store = WordWaveStore()
    features: list[torch.Tensor] = []
    sample_lengths: list[int] = []
    frame_lengths: list[int] = []
    output_lengths: list[int] = []
    edge_padded = 0
    edge_padding_samples = 0
    unalignable: list[str] = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch_start in range(0, len(words), PREPROCESS_BATCH_SIZE):
            batch_words = words[batch_start:batch_start + PREPROCESS_BATCH_SIZE]
            waveforms: list[np.ndarray] = []
            metadata: list[dict[str, Any]] = []
            for word in batch_words:
                waveform, info = extract_word_waveform(store, word)
                waveforms.append(waveform)
                metadata.append(info)
            maximum = max(len(waveform) for waveform in waveforms)
            padded = torch.zeros((len(waveforms), maximum), dtype=torch.float32)
            for index, waveform in enumerate(waveforms):
                padded[index, :len(waveform)] = torch.from_numpy(waveform)
            batch_features = extractor(padded.to(device)).cpu()
            for word, waveform, info, feature in zip(batch_words, waveforms, metadata, batch_features):
                frames = feature_frames(len(waveform))
                if feature.shape[-1] < frames:
                    raise RuntimeError(f"Feature frame underflow: {word['word_id']}")
                item = feature[:, :frames].contiguous()
                if not torch.isfinite(item).all():
                    raise RuntimeError(f"Non-finite feature: {word['word_id']}")
                steps = encoder_steps(frames)
                minimum = target_minimum_steps(word["observed"])
                if steps < minimum:
                    unalignable.append(word["word_id"])
                features.append(item)
                sample_lengths.append(len(waveform))
                frame_lengths.append(frames)
                output_lengths.append(steps)
                padded_samples = info["left_edge_padding_samples"] + info["right_edge_padding_samples"]
                edge_padded += int(padded_samples > 0)
                edge_padding_samples += padded_samples
            print(f"materialize={split_name} words={min(batch_start + len(batch_words), len(words))}/{len(words)}", flush=True)
    return features, {
        "words": len(words),
        "seconds": time.perf_counter() - started,
        "sample_length": distribution(sample_lengths),
        "feature_frames": distribution(frame_lengths),
        "encoder_output_steps": distribution(output_lengths),
        "edge_padded_words": edge_padded,
        "edge_padding_samples": edge_padding_samples,
        "unalignable_words": len(unalignable),
        "unalignable_word_ids": unalignable,
    }


def distribution(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size), "min": float(array.min()), "median": float(np.median(array)),
        "mean": float(array.mean()), "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)), "max": float(array.max()),
    }


def make_training_batches(lengths: list[int], epoch: int) -> list[list[int]]:
    ordered = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    rng = np.random.default_rng(SEED + epoch)
    batches: list[list[int]] = []
    bucket_size = BATCH_SIZE * 32
    for start in range(0, len(ordered), bucket_size):
        bucket = ordered[start:start + bucket_size]
        rng.shuffle(bucket)
        batches.extend([bucket[position:position + BATCH_SIZE] for position in range(0, len(bucket), BATCH_SIZE)])
    rng.shuffle(batches)
    return batches


def make_evaluation_batches(lengths: list[int]) -> list[list[int]]:
    ordered = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    return [ordered[position:position + BATCH_SIZE] for position in range(0, len(ordered), BATCH_SIZE)]


def collate(
    indexes: list[int], features: list[torch.Tensor], words: list[dict[str, Any]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = max(features[index].shape[-1] for index in indexes)
    batch = torch.zeros((len(indexes), 1, N_MELS, maximum), dtype=torch.float32)
    input_lengths: list[int] = []
    target_lengths: list[int] = []
    target_values: list[int] = []
    for position, index in enumerate(indexes):
        feature = features[index]
        batch[position, 0, :, :feature.shape[-1]] = feature
        input_lengths.append(encoder_steps(feature.shape[-1]))
        target = words[index]["target_ids"]
        target_lengths.append(len(target))
        target_values.extend(target)
    targets = torch.tensor(target_values, dtype=torch.long)
    return (
        batch.to(device, non_blocking=True),
        targets.to(device, non_blocking=True),
        torch.tensor(input_lengths, dtype=torch.long, device=device),
        torch.tensor(target_lengths, dtype=torch.long, device=device),
    )


def greedy_decode(logits: torch.Tensor, lengths: torch.Tensor) -> list[list[int]]:
    raw = logits.argmax(dim=-1).detach().cpu()
    output: list[list[int]] = []
    for sequence, length in zip(raw, lengths.detach().cpu().tolist()):
        decoded: list[int] = []
        previous: int | None = None
        for token_value in sequence[:length].tolist():
            token = int(token_value)
            if token != previous and token != BLANK:
                decoded.append(token)
            previous = token
        output.append(decoded)
    return output


def sequence_alignment(reference: list[int], hypothesis: list[int]) -> list[dict[str, Any]]:
    n, m = len(reference), len(hypothesis)
    cost = np.zeros((n + 1, m + 1), dtype=np.int32)
    cost[:, 0] = np.arange(n + 1)
    cost[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = int(cost[i - 1, j - 1]) + int(reference[i - 1] != hypothesis[j - 1])
            deletion = int(cost[i - 1, j]) + 1
            insertion = int(cost[i, j - 1]) + 1
            cost[i, j] = min(diagonal, deletion, insertion)
    operations: list[dict[str, Any]] = []
    i, j = n, m
    while i or j:
        if i and j:
            diagonal_cost = int(cost[i - 1, j - 1]) + int(reference[i - 1] != hypothesis[j - 1])
            if int(cost[i, j]) == diagonal_cost:
                operation = "MATCH" if reference[i - 1] == hypothesis[j - 1] else "SUBSTITUTION"
                operations.append({"operation": operation, "reference_index": i - 1, "hypothesis_index": j - 1})
                i -= 1; j -= 1
                continue
        if i and int(cost[i, j]) == int(cost[i - 1, j]) + 1:
            operations.append({"operation": "DELETE_FROM_EXPECTED", "reference_index": i - 1, "hypothesis_index": None})
            i -= 1
            continue
        if j and int(cost[i, j]) == int(cost[i, j - 1]) + 1:
            operations.append({"operation": "INSERT_IN_DECODED", "reference_index": None, "hypothesis_index": j - 1})
            j -= 1
            continue
        raise RuntimeError("Levenshtein backtrace failed")
    return list(reversed(operations))


def edit_counts(reference: list[int], hypothesis: list[int]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    operations = sequence_alignment(reference, hypothesis)
    counts = Counter(operation["operation"] for operation in operations)
    return {
        "substitution": counts["SUBSTITUTION"],
        "deletion": counts["DELETE_FROM_EXPECTED"],
        "insertion": counts["INSERT_IN_DECODED"],
        "errors": counts["SUBSTITUTION"] + counts["DELETE_FROM_EXPECTED"] + counts["INSERT_IN_DECODED"],
    }, operations


def binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = [int(row["true_relation"] == "deletion") for row in rows]
    predicted = [int(row["predicted_relation"] == "deletion") for row in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(truth, predicted, labels=[0, 1]).astype(int)
    return {
        "rows": len(rows), "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(np.mean(recall)), "macro_f1": float(np.mean(f1)),
        "non_deletion": {"precision": float(precision[0]), "recall": float(recall[0]), "f1": float(f1[0]), "support": int(support[0])},
        "deletion": {"precision": float(precision[1]), "recall": float(recall[1]), "f1": float(f1[1]), "support": int(support[1])},
        "confusion_matrix_labels": ["non_deletion", "deletion"], "confusion_matrix": matrix.tolist(),
    }


def relation_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ["correct", "substitution", "deletion"]
    truth = [row["true_relation"] for row in rows]
    predicted = [row["predicted_relation"] for row in rows]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    return {
        "rows": len(rows), "macro_f1": float(np.mean(f1)),
        "per_class": {
            label: {"precision": float(precision[index]), "recall": float(recall[index]),
                    "f1": float(f1[index]), "support": int(support[index])}
            for index, label in enumerate(labels)
        },
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).astype(int).tolist(),
    }


def false_deletion_rate(rows: list[dict[str, Any]], relation: str) -> float:
    selected = [row for row in rows if row["true_relation"] == relation]
    return sum(row["predicted_relation"] == "deletion" for row in selected) / len(selected)


def acoustic_phone_metrics(word_details: list[dict[str, Any]]) -> dict[str, Any]:
    tp = Counter(); fp = Counter(); fn = Counter(); support = Counter()
    for word in word_details:
        target = word["target_ids"]
        decoded = word["decoded_ids"]
        for operation in word["acoustic_alignment"]:
            name = operation["operation"]
            ref_index, hyp_index = operation["reference_index"], operation["hypothesis_index"]
            if ref_index is not None:
                support[target[ref_index]] += 1
            if name == "MATCH":
                tp[target[ref_index]] += 1
            elif name == "SUBSTITUTION":
                fn[target[ref_index]] += 1; fp[decoded[hyp_index]] += 1
            elif name == "DELETE_FROM_EXPECTED":
                fn[target[ref_index]] += 1
            else:
                fp[decoded[hyp_index]] += 1
    output: dict[str, Any] = {}
    for index, phone in enumerate(PHONE_VOCAB):
        precision = tp[index] / (tp[index] + fp[index]) if tp[index] + fp[index] else 0.0
        recall = tp[index] / (tp[index] + fn[index]) if tp[index] + fn[index] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        output[phone] = {"precision": precision, "recall": recall, "f1": f1, "support": support[index],
                         "tp": tp[index], "fp": fp[index], "fn": fn[index]}
    return output


def evaluate_model(
    model: WordCTCModel,
    words: list[dict[str, Any]],
    features: list[torch.Tensor],
    criterion: nn.CTCLoss,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    decoded_by_index: dict[int, list[int]] = {}
    loss_sum = 0.0
    seen = 0
    lengths = [feature.shape[-1] for feature in features]
    with torch.no_grad():
        for indexes in make_evaluation_batches(lengths):
            batch, targets, input_lengths, target_lengths = collate(indexes, features, words, device)
            logits = model(batch)
            if logits.shape[1] != encoder_steps(batch.shape[-1]):
                raise RuntimeError(f"Empirical temporal length mismatch: input={batch.shape[-1]} output={logits.shape[1]}")
            loss = criterion(logits.log_softmax(-1).transpose(0, 1), targets, input_lengths, target_lengths)
            loss_sum += float(loss.item()) * len(indexes)
            seen += len(indexes)
            for index, decoded in zip(indexes, greedy_decode(logits, input_lengths)):
                decoded_by_index[index] = decoded

    relation_rows: list[dict[str, Any]] = []
    word_details: list[dict[str, Any]] = []
    total_edits = Counter(); total_target_phones = 0; exact = 0; total_insertions = Counter()
    for index, word in enumerate(words):
        decoded = decoded_by_index[index]
        target = word["target_ids"]
        acoustic_counts, acoustic_alignment = edit_counts(target, decoded)
        total_edits.update(acoustic_counts)
        total_target_phones += len(target)
        exact += decoded == target
        relation_alignment = sequence_alignment(word["expected_ids"], decoded)
        predicted_relations: list[str] = ["" for _ in word["expected"]]
        predicted_observed: list[str] = ["" for _ in word["expected"]]
        alignment_operation: list[str] = ["" for _ in word["expected"]]
        inserted: list[str] = []
        for operation in relation_alignment:
            name = operation["operation"]
            expected_index = operation["reference_index"]
            decoded_index = operation["hypothesis_index"]
            if expected_index is None:
                phone = PHONE_VOCAB[decoded[decoded_index]]
                inserted.append(phone); total_insertions[phone] += 1
                continue
            alignment_operation[expected_index] = name
            if name == "MATCH":
                predicted_relations[expected_index] = "correct"
                predicted_observed[expected_index] = word["expected"][expected_index]
            elif name == "SUBSTITUTION":
                predicted_relations[expected_index] = "substitution"
                predicted_observed[expected_index] = PHONE_VOCAB[decoded[decoded_index]]
            elif name == "DELETE_FROM_EXPECTED":
                predicted_relations[expected_index] = "deletion"
                predicted_observed[expected_index] = "<SIL>"
            else:
                raise RuntimeError(f"Unexpected expected-position operation: {name}")
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
                "alignment_operation": alignment_operation[expected_index],
                "decoded_sequence": " ".join(PHONE_VOCAB[token] for token in decoded),
            })
        word_details.append({
            "word_index": index, "word_id": word["word_id"], "speaker_id": word["speaker_id"],
            "utterance_id": word["utterance_id"], "word": word["word"],
            "word_start": float(word["mfa_start"]), "word_end": float(word["mfa_end"]),
            "expected": list(word["expected"]), "target": list(word["observed"]),
            "target_ids": list(target), "decoded": [PHONE_VOCAB[token] for token in decoded],
            "decoded_ids": list(decoded), "acoustic_edit_counts": acoustic_counts,
            "acoustic_alignment": acoustic_alignment, "relation_alignment": relation_alignment,
            "ground_truth_relations": [row["relation"] for row in word["clean_rows"]],
            "predicted_relations": predicted_relations, "predicted_insertions": inserted,
            "contains_deletion": word["deletion"] > 0, "contains_substitution": word["substitution"] > 0,
            "empty_target": len(target) == 0,
            "word_per": (acoustic_counts["errors"] / len(target)) if target else (0.0 if not decoded else None),
        })
    binary = binary_metrics(relation_rows)
    relations = relation_metrics(relation_rows)
    metrics = {
        "loss": loss_sum / seen,
        "phone_error_rate": total_edits["errors"] / total_target_phones,
        "target_phone_denominator": total_target_phones,
        "edit_counts": {name: int(total_edits[name]) for name in ("substitution", "deletion", "insertion", "errors")},
        "exact_decoded_sequence_accuracy": exact / len(words), "exact_decoded_words": exact,
        "binary": binary, "three_relation": relations,
        "correct_false_deletion_rate": false_deletion_rate(relation_rows, "correct"),
        "substitution_false_deletion_rate": false_deletion_rate(relation_rows, "substitution"),
        "predicted_insertions": int(sum(total_insertions.values())),
        "predicted_insertion_phone_counts": dict(total_insertions.most_common()),
        "empty_target_per_rule": "empty target contributes decoded phones as insertions to aggregate PER numerator; denominator remains total nonempty target phones; per-word PER is 0 only for empty->empty and otherwise undefined",
    }
    return metrics, relation_rows, word_details


def train_epoch(
    model: WordCTCModel,
    words: list[dict[str, Any]],
    features: list[torch.Tensor],
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    model.train()
    lengths = [feature.shape[-1] for feature in features]
    batches = make_training_batches(lengths, epoch)
    loss_sum = 0.0; seen = 0; total_errors = 0; total_target = 0; exact = 0
    for batch_number, indexes in enumerate(batches, start=1):
        batch, targets, input_lengths, target_lengths = collate(indexes, features, words, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        log_probs = logits.log_softmax(-1).transpose(0, 1)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}, batch {batch_number}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        loss_sum += float(loss.item()) * len(indexes); seen += len(indexes)
        decoded_batch = greedy_decode(logits, input_lengths)
        for index, decoded in zip(indexes, decoded_batch):
            target = words[index]["target_ids"]
            counts, _ = edit_counts(target, decoded)
            total_errors += counts["errors"]; total_target += len(target); exact += decoded == target
        if batch_number % 250 == 0 or batch_number == len(batches):
            print(f"epoch={epoch} train_batch={batch_number}/{len(batches)} loss={loss_sum/seen:.6f}", flush=True)
    return {
        "ctc_loss": loss_sum / seen,
        "online_phone_error_rate": total_errors / total_target,
        "online_exact_sequence_accuracy": exact / len(words),
        "note": "online TRAIN PER is accumulated from each batch immediately before its optimizer update with training-mode dropout",
    }


def save_history(records: list[dict[str, Any]]) -> None:
    write_json(EXPERIMENT_DIR / "epoch_metrics.json", records)
    fields = [
        "epoch", "train_ctc_loss", "train_online_per", "train_online_exact", "validation_ctc_loss",
        "validation_per", "validation_exact", "binary_macro_f1", "balanced_accuracy", "deletion_precision",
        "deletion_recall", "deletion_f1", "substitution_false_deletion_rate", "three_relation_macro_f1",
        "epoch_seconds",
    ]
    with (EXPERIMENT_DIR / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for record in records:
            validation = record["validation"]
            writer.writerow({
                "epoch": record["epoch"], "train_ctc_loss": record["train"]["ctc_loss"],
                "train_online_per": record["train"]["online_phone_error_rate"],
                "train_online_exact": record["train"]["online_exact_sequence_accuracy"],
                "validation_ctc_loss": validation["loss"], "validation_per": validation["phone_error_rate"],
                "validation_exact": validation["exact_decoded_sequence_accuracy"],
                "binary_macro_f1": validation["binary"]["macro_f1"],
                "balanced_accuracy": validation["binary"]["balanced_accuracy"],
                "deletion_precision": validation["binary"]["deletion"]["precision"],
                "deletion_recall": validation["binary"]["deletion"]["recall"],
                "deletion_f1": validation["binary"]["deletion"]["f1"],
                "substitution_false_deletion_rate": validation["substitution_false_deletion_rate"],
                "three_relation_macro_f1": validation["three_relation"]["macro_f1"],
                "epoch_seconds": record["epoch_seconds"],
            })


def subset_metrics(rows: list[dict[str, Any]], predicate) -> dict[str, Any]:
    selected = [row for row in rows if predicate(row)]
    return binary_metrics(selected) if selected else {"rows": 0}


def selected_diagnostics(
    words: list[dict[str, Any]], relation_rows: list[dict[str, Any]], word_details: list[dict[str, Any]],
    metrics: dict[str, Any], matched_ids: set[int],
) -> dict[str, Any]:
    for row in relation_rows:
        row["is_matched_control_row"] = row["source_csv_row"] in matched_ids
        word = words[row["word_index"]]
        length = len(word["clean_rows"]); position = row["expected_phone_index"]
        row["word_position"] = "single" if length == 1 else "initial" if position == 0 else "final" if position == length - 1 else "medial"
    matched_rows = [row for row in relation_rows if row["is_matched_control_row"]]
    matched = binary_metrics(matched_rows)
    matched["identity_rows"] = len(matched_ids)
    matched["identity_set_reproduced"] = len(matched_rows) == len(matched_ids) == 1434
    if not matched["identity_set_reproduced"]:
        raise RuntimeError(f"Matched control identity reproduction failed: {len(matched_rows)}")
    if matched["deletion"]["support"] != 717 or matched["non_deletion"]["support"] != 717:
        raise RuntimeError("Matched class counts changed")

    word_by_index = {detail["word_index"]: detail for detail in word_details}
    speaker_output: dict[str, Any] = {}
    for speaker in r3.VALIDATION_SPEAKERS:
        speaker_rows = [row for row in relation_rows if row["speaker_id"] == speaker]
        speaker_words = [detail for detail in word_details if detail["speaker_id"] == speaker]
        edits = sum(detail["acoustic_edit_counts"]["errors"] for detail in speaker_words)
        target_count = sum(len(detail["target_ids"]) for detail in speaker_words)
        block = binary_metrics(speaker_rows)
        speaker_output[speaker] = {
            "usable_words": len(speaker_words), "target_phones": target_count,
            "phone_error_rate": edits / target_count, "binary": block,
            "substitution_false_deletion_rate": false_deletion_rate(speaker_rows, "substitution"),
        }

    phone_output = {phone: subset_metrics(relation_rows, lambda row, p=phone: row["expected_phone"] == p) for phone in PHONE_VOCAB}
    group_output = {
        "D_T_R_L": subset_metrics(relation_rows, lambda row: row["expected_phone"] in {"D", "T", "R", "L"}),
        "all_other_phones": subset_metrics(relation_rows, lambda row: row["expected_phone"] not in {"D", "T", "R", "L"}),
    }
    position_output = {
        position: subset_metrics(relation_rows, lambda row, p=position: row["word_position"] == p)
        for position in ("initial", "medial", "final", "single")
    }
    pure_word_indexes = {index for index, word in enumerate(words) if word["deletion"] > 0 and word["substitution"] == 0}
    mixed_word_indexes = {index for index, word in enumerate(words) if word["deletion"] > 0 and word["substitution"] > 0}
    multiple_word_indexes = {index for index, word in enumerate(words) if word["deletion"] > 1}
    multi_error = {
        "pure_deletion_words": subset_metrics(relation_rows, lambda row: row["word_index"] in pure_word_indexes),
        "substitution_plus_deletion_words": subset_metrics(relation_rows, lambda row: row["word_index"] in mixed_word_indexes),
        "multiple_deletion_words": subset_metrics(relation_rows, lambda row: row["word_index"] in multiple_word_indexes),
        "word_counts": {"pure_deletion": len(pure_word_indexes), "substitution_plus_deletion": len(mixed_word_indexes), "multiple_deletion": len(multiple_word_indexes)},
    }
    acoustic_classes = acoustic_phone_metrics(word_details)
    rare = {phone: acoustic_classes[phone] for phone in ("AX", "OY", "ZH")}
    empty_words = [detail for detail in word_details if detail["empty_target"]]
    empty = {
        "words": len(empty_words), "empty_decodes": sum(len(detail["decoded"]) == 0 for detail in empty_words),
        "hallucinated_phone_count": sum(len(detail["decoded"]) for detail in empty_words),
        "decoded_length_distribution": distribution(len(detail["decoded"]) for detail in empty_words),
        "details": [{"word_id": detail["word_id"], "word": detail["word"], "decoded": detail["decoded"],
                     "decoded_length": len(detail["decoded"])} for detail in empty_words],
    }
    insertions = Counter(phone for detail in word_details for phone in detail["predicted_insertions"])
    insertion_diagnostics = {
        "predicted_insertions": sum(insertions.values()), "validation_words": len(word_details),
        "insertions_per_word": sum(insertions.values()) / len(word_details),
        "words_with_insertions": sum(bool(detail["predicted_insertions"]) for detail in word_details),
        "top_inserted_phones": dict(insertions.most_common()), "diagnostic_only": True,
    }
    return {
        "matched": matched, "speakers": speaker_output, "phones": phone_output, "phone_groups": group_output,
        "positions": position_output, "multi_error": multi_error, "acoustic_classes": acoustic_classes,
        "rare": rare, "empty": empty, "insertions": insertion_diagnostics,
    }


def write_prediction_exports(relation_rows: list[dict[str, Any]], word_details: list[dict[str, Any]]) -> None:
    fields = [
        "speaker_id", "utterance_id", "word_id", "word", "word_start", "word_end", "expected_phone",
        "expected_phone_index", "source_csv_row", "true_relation", "true_observed_phone", "predicted_relation",
        "predicted_observed_phone", "decoded_sequence", "alignment_operation", "word_position",
        "is_matched_control_row",
    ]
    with (EXPERIMENT_DIR / "validation_phone_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in relation_rows:
            writer.writerow({field: row[field] for field in fields})
    with (EXPERIMENT_DIR / "validation_word_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for detail in word_details:
            payload = {key: value for key, value in detail.items() if key not in {"target_ids", "decoded_ids"}}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_matched_ids() -> set[int]:
    with MATCHED_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    identities = {int(row["source_csv_row"]) for row in rows}
    if len(rows) != 1434 or len(identities) != 1434:
        raise RuntimeError("Frozen matched control row count/identity uniqueness failed")
    return identities


def gate_report(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    speaker_details = {
        speaker: {
            "eligible": values["binary"]["deletion"]["support"] >= 30,
            "deletion_support": values["binary"]["deletion"]["support"],
            "deletion_recall": values["binary"]["deletion"]["recall"],
            "pass": values["binary"]["deletion"]["support"] < 30 or values["binary"]["deletion"]["recall"] >= 0.25,
        }
        for speaker, values in diagnostics["speakers"].items()
    }
    gates = {
        "A_per_le_0_55": metrics["phone_error_rate"] <= 0.55,
        "B_binary_macro_f1_ge_0_70": metrics["binary"]["macro_f1"] >= 0.70,
        "C_deletion_recall_ge_0_45": metrics["binary"]["deletion"]["recall"] >= 0.45,
        "D_deletion_f1_ge_0_40": metrics["binary"]["deletion"]["f1"] >= 0.40,
        "E_substitution_false_deletion_le_0_25": metrics["substitution_false_deletion_rate"] <= 0.25,
        "F_matched_macro_f1_ge_0_60": diagnostics["matched"]["macro_f1"] >= 0.60,
        "G_matched_deletion_f1_ge_0_55": diagnostics["matched"]["deletion"]["f1"] >= 0.55,
        "H_speaker_recall": all(item["pass"] for item in speaker_details.values()),
        "I_three_relation_macro_f1_ge_0_40": metrics["three_relation"]["macro_f1"] >= 0.40,
    }
    return {"gates": gates, "speaker_details": speaker_details, "all_pass": all(gates.values())}


def classify_result(metrics: dict[str, Any], gates: dict[str, Any]) -> str:
    if gates["all_pass"]:
        return "R4_4B_CTC_DELETION_CONFIRMED"
    if (metrics["phone_error_rate"] <= 0.55 and metrics["binary"]["macro_f1"] >= 0.60
            and metrics["binary"]["deletion"]["f1"] >= 0.25):
        return "R4_4B_CTC_DELETION_PARTIAL"
    if metrics["phone_error_rate"] <= 0.65:
        return "R4_4B_CTC_SEQUENCE_LEARNED_DELETION_WEAK"
    return "R4_4B_CTC_NOT_LEARNED"


def main() -> int:
    if EXPERIMENT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite experiment: {EXPERIMENT_DIR}")
    set_determinism()
    prereg, verification = verify_sources()
    audio_root = r3.require_audio_root()
    words, dataset_summary = load_words(audio_root)
    train_words = [word for word in words if word["split"] == "train"]
    validation_words = [word for word in words if word["split"] == "validation"]
    matched_ids = read_matched_ids()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WordCTCModel().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_empty = sum(not word["target_ids"] for word in train_words)
    validation_empty = sum(not word["target_ids"] for word in validation_words)
    if (train_empty, validation_empty) != (28, 15):
        raise RuntimeError(f"Empty target count mismatch: {(train_empty, validation_empty)}")

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)
    write_json(EXPERIMENT_DIR / "preflight.json", {
        "status": "PREPROCESSING", "verification": verification, "git_commit": git_commit(),
        "python": platform.python_version(), "pytorch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seed": SEED, "training_started": False, "test_paths_resolved": False,
        "test_audio_read": False, "test_target_reconstruction": False, "test_inference": False,
    })
    write_json(EXPERIMENT_DIR / "dataset_summary.json", {
        **dataset_summary, "train_target_phones": sum(len(word["target_ids"]) for word in train_words),
        "validation_target_phones": sum(len(word["target_ids"]) for word in validation_words),
        "empty_targets": {"train": train_empty, "validation": validation_empty},
        "test_speakers_excluded_before_reconstruction": sorted(TEST_SPEAKERS),
    })
    feature_config = dict(prereg["features"])
    feature_config.update({"sample_rate": SAMPLE_RATE, "word_span": "MFA word boundary only", "side_context_seconds": 0.0})
    write_json(EXPERIMENT_DIR / "feature_config.json", feature_config)
    write_json(EXPERIMENT_DIR / "vocabulary.json", prereg["vocabulary"])
    write_json(EXPERIMENT_DIR / "model_config.json", {**prereg["architecture"], "parameter_count": parameter_count,
                                                        "initialization": prereg["initialization"]})
    write_json(EXPERIMENT_DIR / "training_config.json", {
        **prereg["training"], "loss": prereg["loss"], "selection": prereg["training"]["checkpoint_metric"],
        "final_result_classification_frozen_before_training": {
            "confirmed": "all nine frozen gates pass",
            "partial": "PER<=.55 AND binary Macro-F1>=.60 AND deletion F1>=.25",
            "sequence_learned_deletion_weak": "not above, but PER<=.65",
            "not_learned": "PER>.65",
        },
    })

    train_features, train_feature_report = materialize_features(train_words, device, "train")
    validation_features, validation_feature_report = materialize_features(validation_words, device, "validation")
    if train_feature_report["unalignable_words"] or validation_feature_report["unalignable_words"]:
        write_json(EXPERIMENT_DIR / "final_status.json", {
            "status": "R4_4B_TEMPORAL_CONTRACT_FAIL", "train": train_feature_report,
            "validation": validation_feature_report, "training_occurred": False, "r4_test_accessed": False,
        })
        print("R4_4B_TEMPORAL_CONTRACT_FAIL", flush=True)
        return 0
    model.eval()
    with torch.no_grad():
        for frames in sorted({feature.shape[-1] for feature in train_features + validation_features}):
            probe = torch.zeros((1, 1, N_MELS, frames), device=device)
            actual = model(probe).shape[1]
            if actual != encoder_steps(frames):
                raise RuntimeError(f"Empirical temporal contract failed for {frames}: {actual}")
    preflight = json.loads((EXPERIMENT_DIR / "preflight.json").read_text(encoding="utf-8"))
    preflight.update({"status": "PASS", "train_features": train_feature_report,
                      "validation_features": validation_feature_report, "training_started": True})
    write_json(EXPERIMENT_DIR / "preflight.json", preflight)

    criterion = nn.CTCLoss(blank=BLANK, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    checkpoint_path = EXPERIMENT_DIR / CHECKPOINT_NAME
    history: list[dict[str, Any]] = []
    best_epoch = 0; best_per = math.inf; best_deletion_f1 = -1.0
    training_started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_epoch(model, train_words, train_features, criterion, optimizer, device, epoch)
        validation_metrics, _, _ = evaluate_model(model, validation_words, validation_features, criterion, device)
        epoch_record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics,
                        "epoch_seconds": time.perf_counter() - epoch_started}
        history.append(epoch_record); save_history(history)
        current_per = validation_metrics["phone_error_rate"]
        current_deletion_f1 = validation_metrics["binary"]["deletion"]["f1"]
        better = current_per < best_per - TIE_TOLERANCE or (
            abs(current_per - best_per) <= TIE_TOLERANCE and current_deletion_f1 > best_deletion_f1 + TIE_TOLERANCE
        )
        if better:
            best_epoch, best_per, best_deletion_f1 = epoch, current_per, current_deletion_f1
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch, "validation_per": current_per,
                "validation_deletion_f1": current_deletion_f1, "vocabulary": list(PHONE_VOCAB),
                "blank_index": BLANK, "model_config": prereg["architecture"],
                "training_config": prereg["training"], "fresh_random_initialization": True,
                "r3_checkpoint_loaded": False,
            }, checkpoint_path)
        print(
            f"epoch={epoch} train_loss={train_metrics['ctc_loss']:.6f} train_per={train_metrics['online_phone_error_rate']:.6f} "
            f"val_loss={validation_metrics['loss']:.6f} val_per={current_per:.6f} "
            f"exact={validation_metrics['exact_decoded_sequence_accuracy']:.6f} "
            f"binary_mf1={validation_metrics['binary']['macro_f1']:.6f} "
            f"del_p={validation_metrics['binary']['deletion']['precision']:.6f} "
            f"del_r={validation_metrics['binary']['deletion']['recall']:.6f} "
            f"del_f1={current_deletion_f1:.6f} sub_fd={validation_metrics['substitution_false_deletion_rate']:.6f} "
            f"three_mf1={validation_metrics['three_relation']['macro_f1']:.6f} best_epoch={best_epoch}", flush=True
        )

    training_seconds = time.perf_counter() - training_started
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected_metrics, relation_rows, word_details = evaluate_model(
        model, validation_words, validation_features, criterion, device
    )
    if abs(selected_metrics["phone_error_rate"] - best_per) > 1e-12:
        raise RuntimeError("Selected checkpoint PER did not reproduce")
    diagnostics = selected_diagnostics(validation_words, relation_rows, word_details, selected_metrics, matched_ids)
    gates = gate_report(selected_metrics, diagnostics)
    status = classify_result(selected_metrics, gates)
    checkpoint_sha = sha256(checkpoint_path)

    write_json(EXPERIMENT_DIR / "selected_checkpoint.json", {
        "path": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": checkpoint_sha,
        "selected_epoch": best_epoch, "selection_order": ["lowest validation PER", "higher deletion F1", "earlier epoch"],
        "validation_per": selected_metrics["phone_error_rate"], "validation_deletion_f1": selected_metrics["binary"]["deletion"]["f1"],
    })
    write_json(EXPERIMENT_DIR / "validation_per_metrics.json", {
        "phone_error_rate": selected_metrics["phone_error_rate"], "loss": selected_metrics["loss"],
        "exact_decoded_sequence_accuracy": selected_metrics["exact_decoded_sequence_accuracy"],
        "target_phone_denominator": selected_metrics["target_phone_denominator"],
        "edit_counts": selected_metrics["edit_counts"], "per_class": diagnostics["acoustic_classes"],
        "empty_target_rule": selected_metrics["empty_target_per_rule"],
    })
    write_json(EXPERIMENT_DIR / "validation_binary_metrics.json", selected_metrics["binary"])
    write_json(EXPERIMENT_DIR / "validation_3class_metrics.json", selected_metrics["three_relation"])
    write_json(EXPERIMENT_DIR / "matched_control_metrics.json", diagnostics["matched"])
    write_json(EXPERIMENT_DIR / "speaker_metrics.json", diagnostics["speakers"])
    write_json(EXPERIMENT_DIR / "phone_metrics.json", {"phones": diagnostics["phones"], "groups": diagnostics["phone_groups"]})
    write_json(EXPERIMENT_DIR / "position_metrics.json", diagnostics["positions"])
    write_json(EXPERIMENT_DIR / "multi_error_metrics.json", diagnostics["multi_error"])
    write_json(EXPERIMENT_DIR / "rare_phone_metrics.json", diagnostics["rare"])
    write_json(EXPERIMENT_DIR / "empty_target_metrics.json", diagnostics["empty"])
    write_json(EXPERIMENT_DIR / "insertion_diagnostics.json", diagnostics["insertions"])
    write_prediction_exports(relation_rows, word_details)

    peak_deletion_epoch = max(history, key=lambda item: (item["validation"]["binary"]["deletion"]["f1"], -item["epoch"]))["epoch"]
    last_per = [item["validation"]["phone_error_rate"] for item in history[-4:]]
    last_loss = [item["validation"]["loss"] for item in history[-4:]]
    budget_limiting = best_epoch == EPOCHS and last_per[-1] < last_per[0] and last_loss[-1] < last_loss[0]
    trend = {
        "selected_epoch": best_epoch, "minimum_validation_per": best_per,
        "peak_deletion_f1_epoch": peak_deletion_epoch, "same_epoch": best_epoch == peak_deletion_epoch,
        "epoch_36_selected": best_epoch == EPOCHS, "last_four_validation_per": last_per,
        "last_four_validation_loss": last_loss, "training_budget_possibly_limiting": budget_limiting,
        "flag": "TRAINING_BUDGET_POSSIBLY_LIMITING" if budget_limiting else "NO_BUDGET_EXTENSION_AUTHORIZED",
    }
    comparator = {
        "duration_only": {"macro_f1": 0.668146, "deletion_f1": 0.364164},
        "r4_1": {"macro_f1": 0.657336, "deletion_f1": 0.341612},
        "r4_2a": {"macro_f1": 0.566997, "deletion_f1": 0.197525},
        "r4_3b": {"macro_f1": 0.503101, "deletion_f1": 0.025465},
        "r4_4b": {"macro_f1": selected_metrics["binary"]["macro_f1"], "deletion_f1": selected_metrics["binary"]["deletion"]["f1"]},
    }
    write_json(EXPERIMENT_DIR / "training_trend.json", trend)
    write_json(EXPERIMENT_DIR / "comparator_table.json", comparator)
    final = {
        "status": status, "selected_epoch": best_epoch, "gates": gates,
        "training_seconds": training_seconds, "training_occurred": True, "new_neural_runs": 1,
        "r4_test_accessed": False, "test_paths_resolved": False, "test_audio_read": False,
        "test_target_reconstruction": False, "test_inference": False, "test_metrics": False,
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": checkpoint_sha, "trend": trend,
    }
    write_json(EXPERIMENT_DIR / "final_status.json", final)
    report = f"""# R4-4B Locked Self-Trained CTC Sequence Experiment

Research-only. R4 TEST remained closed.

- Final: `{status}`
- Selected epoch: {best_epoch}
- Validation PER: {selected_metrics['phone_error_rate']:.6f}
- Binary Macro-F1: {selected_metrics['binary']['macro_f1']:.6f}
- Deletion P/R/F1: {selected_metrics['binary']['deletion']['precision']:.6f} / {selected_metrics['binary']['deletion']['recall']:.6f} / {selected_metrics['binary']['deletion']['f1']:.6f}
- Three-relation Macro-F1: {selected_metrics['three_relation']['macro_f1']:.6f}
- All frozen gates pass: {gates['all_pass']}
- Checkpoint SHA-256: `{checkpoint_sha}`
- Training: exactly one seed-42 run; R4 TEST access: NO
"""
    (EXPERIMENT_DIR / "r4_4b_report.md").write_text(report, encoding="utf-8")
    key_files = [
        "preflight.json", "dataset_summary.json", "feature_config.json", "vocabulary.json", "model_config.json",
        "training_config.json", "epoch_metrics.json", "training_history.csv", "selected_checkpoint.json",
        "validation_per_metrics.json", "validation_binary_metrics.json", "validation_3class_metrics.json",
        "matched_control_metrics.json", "speaker_metrics.json", "phone_metrics.json", "position_metrics.json",
        "multi_error_metrics.json", "rare_phone_metrics.json", "empty_target_metrics.json",
        "insertion_diagnostics.json", "validation_phone_predictions.csv", "validation_word_predictions.jsonl",
        "training_trend.json", "comparator_table.json", "final_status.json", "r4_4b_report.md", CHECKPOINT_NAME,
    ]
    write_json(EXPERIMENT_DIR / "artifact_hashes.json", {
        "algorithm": "SHA-256", "files": {name: sha256(EXPERIMENT_DIR / name) for name in key_files},
        "note": "artifact_hashes.json intentionally excludes itself",
    })
    print(json.dumps({"final": final, "selected_metrics": selected_metrics, "matched": diagnostics["matched"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
