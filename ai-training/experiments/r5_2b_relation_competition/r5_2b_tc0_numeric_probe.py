from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_frozen_scorer(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_r5_2b_scorer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load frozen scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def difference(value: float | None, reference: float) -> dict[str, float | None]:
    if value is None:
        return {"absolute": None, "relative_to_scorer": None}
    absolute = abs(value - reference)
    return {
        "absolute": absolute,
        "relative_to_scorer": absolute / abs(reference) if reference else absolute,
    }


def framework_empty(log_probs: torch.Tensor, steps: int) -> dict[str, Any]:
    try:
        loss = torch.nn.CTCLoss(blank=40, reduction="none", zero_infinity=True)(
            log_probs.unsqueeze(1),
            torch.empty(0, dtype=torch.long, device=log_probs.device),
            torch.tensor([steps], dtype=torch.long, device=log_probs.device),
            torch.tensor([0], dtype=torch.long, device=log_probs.device),
        )
        return {"supported": True, "value": -scalar(loss), "error": None}
    except Exception as error:
        return {"supported": False, "value": None, "error": f"{type(error).__name__}: {error}"}


def run_case(scorer, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    steps = 4
    logits = ((torch.arange(steps * 41, dtype=dtype).reshape(steps, 41).remainder(17) - 8.0) / 7.0).to(device)
    shared_log_probs = torch.log_softmax(logits, dim=-1)

    scorer_result = scorer.score_target(logits, [], steps)
    scorer_value = scorer_result.target_score
    shared_same_device_sum = scalar(shared_log_probs[:, 40].sum())
    adapter_cpu_sum = scalar(shared_log_probs.detach().cpu()[:, 40].sum())
    recomputed_same_device_sum = scalar(torch.log_softmax(logits, dim=-1)[:, 40].sum())
    recomputed_cpu_sum = scalar(torch.log_softmax(logits, dim=-1).detach().cpu()[:, 40].sum())
    promoted_same_device_sum = scalar(shared_log_probs[:, 40].to(torch.float64).sum())
    promoted_cpu_sum = scalar(shared_log_probs.detach().cpu()[:, 40].to(torch.float64).sum())
    blank_values_python = [float(value) for value in shared_log_probs.detach().cpu()[:, 40].tolist()]
    python_fsum = float(math.fsum(blank_values_python))
    numpy_values = shared_log_probs.detach().cpu()[:, 40].numpy()
    numpy_source_dtype_sum = float(np.sum(numpy_values, dtype=numpy_values.dtype))
    numpy_float64_sum = float(np.sum(numpy_values, dtype=np.float64))
    framework = framework_empty(shared_log_probs, steps)

    paths = {
        "frozen_scorer": scorer_value,
        "shared_log_probs_same_device_tensor_sum": shared_same_device_sum,
        "historical_adapter_cpu_tensor_sum": adapter_cpu_sum,
        "separate_recomputed_same_device_tensor_sum": recomputed_same_device_sum,
        "separate_recomputed_cpu_tensor_sum": recomputed_cpu_sum,
        "shared_log_probs_promoted_float64_same_device_sum": promoted_same_device_sum,
        "shared_log_probs_promoted_float64_cpu_sum": promoted_cpu_sum,
        "python_math_fsum": python_fsum,
        "numpy_source_dtype_sum": numpy_source_dtype_sum,
        "numpy_float64_sum": numpy_float64_sum,
        "framework_target_length_zero_ctc": framework["value"],
    }
    return {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "shape": list(logits.shape),
        "T": steps,
        "blank_index": 40,
        "logits_dtype": str(logits.dtype).replace("torch.", ""),
        "log_softmax_dtype": str(shared_log_probs.dtype).replace("torch.", ""),
        "logits_device": str(logits.device),
        "log_probs_device": str(shared_log_probs.device),
        "blank_log_prob_values": blank_values_python,
        "paths": paths,
        "differences_from_frozen_scorer": {
            name: difference(value, scorer_value) for name, value in paths.items()
        },
        "framework": framework,
        "recomputation_contributes": recomputed_same_device_sum != shared_same_device_sum,
        "device_transfer_before_sum_contributes": adapter_cpu_sum != shared_same_device_sum,
        "source_dtype_machine_epsilon": float(torch.finfo(dtype).eps),
        "source_dtype_sum_abs_terms": float(math.fsum(abs(value) for value in blank_values_python)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scorer = load_frozen_scorer(args.scorer)

    cases = []
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    for device in devices:
        for dtype in (torch.float32, torch.float64):
            cases.append(run_case(scorer, device, dtype))

    historical = next(case for case in cases if case["device"] == "cuda" and case["dtype"] == "float32")
    observed = historical["differences_from_frozen_scorer"]["historical_adapter_cpu_tensor_sum"]["absolute"]
    payload = {
        "scope": "deterministic synthetic tensors only",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cases": cases,
        "historical_discrepancy_reproduced": observed == 9.5367431640625e-7,
        "historical_absolute_difference": observed,
        "audio_accessed": False,
        "checkpoint_loaded": False,
        "model_inference": False,
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
