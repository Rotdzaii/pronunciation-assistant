from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AI_WORKER_ROOT.parent
if str(AI_WORKER_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT))

from app.scorers.cnn_attention_scorer import SmallPronunciationCNNAttention, _index_to_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect local .pt checkpoints for cnn_attention_context compatibility.",
    )
    parser.add_argument(
        "--models-dir",
        default=str(REPO_ROOT / "ai-training" / "models"),
        help="Directory containing local checkpoint files. Default: ai-training/models",
    )
    parser.add_argument("--pattern", default="*.pt", help="Glob pattern for checkpoint files. Default: *.pt")
    parser.add_argument("--top", type=int, default=50, help="Maximum checkpoints to inspect. Default: 50")
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write the inspection result JSON.",
    )
    return parser.parse_args()


def _bool_key_count(keys: list[str], prefix: str) -> int:
    return sum(1 for key in keys if key.startswith(prefix))


def _classify_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(str(key) for key in state_dict.keys())
    return {
        "parameter_key_count": len(keys),
        "attention_key_count": _bool_key_count(keys, "attention.score."),
        "classifier_sequential_key_count": _bool_key_count(keys, "classifier.1."),
        "classifier_simple_key_count": sum(1 for key in keys if key in {"classifier.weight", "classifier.bias"}),
        "feature_key_count": _bool_key_count(keys, "features."),
        "has_attention_score_weight": "attention.score.weight" in keys,
        "has_attention_score_bias": "attention.score.bias" in keys,
        "has_classifier_1_weight": "classifier.1.weight" in keys,
        "has_classifier_1_bias": "classifier.1.bias" in keys,
        "has_classifier_weight": "classifier.weight" in keys,
        "has_classifier_bias": "classifier.bias" in keys,
    }


def _state_dict_from_checkpoint(checkpoint: Any) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(checkpoint, dict):
        return None, "checkpoint is not a dictionary"
    if isinstance(checkpoint.get("model_state_dict"), dict):
        return checkpoint["model_state_dict"], "model_state_dict"
    for key in ("state_dict", "model", "weights"):
        if isinstance(checkpoint.get(key), dict):
            return checkpoint[key], key
    return None, "missing_state_dict"


def _safe_top_level_keys(checkpoint: Any) -> list[str]:
    if not isinstance(checkpoint, dict):
        return []
    return sorted(str(key) for key in checkpoint.keys())


def _compatibility_reason(
    checkpoint: dict[str, Any],
    state_dict: dict[str, Any] | None,
    state_dict_source: str,
) -> tuple[bool, str]:
    if not isinstance(checkpoint, dict):
        return False, "checkpoint is not a dictionary"
    if state_dict is None:
        return False, f"missing compatible state dict key: {state_dict_source}"

    try:
        index_to_label = _index_to_label(checkpoint)
    except Exception as exc:
        return False, f"unable to infer label mapping: {exc}"

    num_classes = len(index_to_label)
    if num_classes <= 0:
        return False, "no label mapping was inferred"

    dropout = float(checkpoint.get("config", {}).get("dropout", 0.2))
    model = SmallPronunciationCNNAttention(num_classes=num_classes, dropout=dropout)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        return False, str(exc)
    return True, "strict load succeeded for SmallPronunciationCNNAttention"


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "compatible": False,
        "reason": "",
        "top_level_keys": [],
        "state_dict_source": None,
        "parameter_key_count": 0,
        "attention_key_count": 0,
        "classifier_sequential_key_count": 0,
        "classifier_simple_key_count": 0,
        "label_count": 0,
        "looks_like_old_simple_cnn": False,
        "suggested_env_command": None,
    }

    try:
        import torch
    except ImportError as exc:
        result["reason"] = f"torch import failed: {exc}"
        return result

    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        result["reason"] = f"torch.load failed: {exc}"
        return result

    result["top_level_keys"] = _safe_top_level_keys(checkpoint)
    state_dict, state_dict_source = _state_dict_from_checkpoint(checkpoint)
    result["state_dict_source"] = state_dict_source

    if isinstance(checkpoint, dict):
        try:
            result["label_count"] = len(_index_to_label(checkpoint))
        except Exception:
            result["label_count"] = 0

    if state_dict is not None:
        key_summary = _classify_state_dict_keys(state_dict)
        result.update(key_summary)
        result["looks_like_old_simple_cnn"] = bool(
            key_summary["classifier_simple_key_count"] > 0 and key_summary["classifier_sequential_key_count"] == 0
        )

    compatible, reason = _compatibility_reason(checkpoint, state_dict, state_dict_source)
    result["compatible"] = compatible
    result["reason"] = reason
    if compatible:
        result["suggested_env_command"] = f'$env:CNN_ATTENTION_CONTEXT_CHECKPOINT_PATH="{path}"'
    return result


def sort_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        results,
        key=lambda item: (
            not bool(item.get("compatible")),
            not str(item.get("filename", "")).lower().startswith("l2_arctic_cnn_attention"),
            str(item.get("filename", "")).lower(),
        ),
    )


def print_text_summary(results: list[dict[str, Any]], inspected_dir: Path, pattern: str, limit: int) -> None:
    compatible = [item for item in results if item.get("compatible")]
    print("CHECKPOINT_INSPECTION")
    print(f"models_dir={inspected_dir}")
    print(f"pattern={pattern}")
    print(f"inspected_count={len(results)}")
    print(f"compatible_count={len(compatible)}")
    print(f"top_limit={limit}")
    print()

    for item in results:
        print(item["filename"])
        print(f"compatible={item['compatible']}")
        print(f"reason={item['reason']}")
        print(f"state_dict_source={item['state_dict_source']}")
        print(f"label_count={item['label_count']}")
        print(f"parameter_key_count={item['parameter_key_count']}")
        print(f"attention_key_count={item['attention_key_count']}")
        print(f"classifier_sequential_key_count={item['classifier_sequential_key_count']}")
        print(f"classifier_simple_key_count={item['classifier_simple_key_count']}")
        print(f"looks_like_old_simple_cnn={item['looks_like_old_simple_cnn']}")
        if item.get("suggested_env_command"):
            print(f"suggested_env_command={item['suggested_env_command']}")
        print()


def main() -> int:
    args = parse_args()
    models_dir = Path(args.models_dir).expanduser()
    if not models_dir.exists():
        print(f"models_dir_missing={models_dir}")
        return 1

    candidates = sorted(models_dir.rglob(args.pattern))[: max(int(args.top), 0)]
    results = sort_results([inspect_checkpoint(path) for path in candidates])

    payload = {
        "models_dir": str(models_dir),
        "pattern": args.pattern,
        "inspected_count": len(results),
        "compatible_count": sum(1 for item in results if item.get("compatible")),
        "results": results,
    }

    if args.json_output:
        output_path = Path(args.json_output).expanduser()
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"json_output_written={output_path}")

    print_text_summary(results, models_dir, args.pattern, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
