#!/usr/bin/env python3
"""
Model evaluation report generator.
Reads existing CSVs/JSONs from datasets/l2-arctic/evaluation/.
No retraining. No new evaluation runs.

Outputs (written to datasets/l2-arctic/evaluation/):
  model_comparison_table.txt
  confusion_matrix_seed42_HQTV.png
  per_error_type_f1_comparison.png
"""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

EVAL_DIR = Path(__file__).parent.parent / "datasets" / "l2-arctic" / "evaluation"
OUT_DIR = EVAL_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _test_f1(rows: list[dict]) -> dict[str, float]:
    return {r["error_type"]: float(r["f1"]) for r in rows if r["split"] == "test"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load all metrics
# ─────────────────────────────────────────────────────────────────────────────

# CNN Baseline
cnn_f1 = _test_f1(_read_csv(EVAL_DIR / "error_type_per_class_metrics.csv"))
cnn_acc, cnn_macro = 0.6966, 0.4657

# CNN + Attention (seed 42) — from stability runs CSV for the overall test metrics
att_runs = {r["seed"]: r for r in _read_csv(EVAL_DIR / "cnn_attention_stability_runs.csv")}
att_s42 = att_runs["42"]
att_f1 = _test_f1(_read_csv(EVAL_DIR / "cnn_attention_per_class_metrics.csv"))
att_acc  = float(att_s42["test_accuracy"])
att_macro = float(att_s42["test_macro_f1"])

# CNN + Attention + Context (seed 42, HQTV held-out)
ctx_rows = _read_csv(EVAL_DIR / "vietnamese_speaker_disjoint_context_stability_per_class.csv")
ctx_hqtv42 = {
    r["error_type"]: float(r["f1"])
    for r in ctx_rows
    if r["seed"] == "42" and r["held_out_speaker"] == "HQTV"
}
ctx_acc, ctx_macro = 0.6848, 0.5489   # from JSON: test accuracy/macro_f1, seed 42 HQTV

# Confusion matrix: CNN+Attention+Context, seed 42, HQTV held-out, test split
with open(EVAL_DIR / "vietnamese_speaker_disjoint_context_stability_results.json", encoding="utf-8") as f:
    ctx_stability = json.load(f)

hqtv42_fold = next(
    fold for fold in ctx_stability["folds"]
    if "seed_42" in fold.get("checkpoint_path", "").replace("\\", "/")
    and fold["held_out_speaker"] == "HQTV"
)
cm_raw = hqtv42_fold["test"]["confusion_matrix"]
LABELS = ["addition", "deletion", "substitution"]
cm = np.array([[cm_raw[a][p] for p in LABELS] for a in LABELS])

# ─────────────────────────────────────────────────────────────────────────────
# 2. Comparison table (console + file)
# ─────────────────────────────────────────────────────────────────────────────

COL = {"model": 38, "acc": 7, "macro": 8, "add": 7, "del": 7, "sub": 7, "note": 30}
H   = "-"

def _row(model, acc, macro, add, d, sub, note):
    return (
        f"  {model:<{COL['model']}} {acc:>{COL['acc']}.4f} {macro:>{COL['macro']}.4f}"
        f" {add:>{COL['add']}.4f} {d:>{COL['del']}.4f} {sub:>{COL['sub']}.4f}  {note}"
    )

header = (
    f"  {'Model':<{COL['model']}} {'Acc':>{COL['acc']}} {'MacroF1':>{COL['macro']}}"
    f" {'AddF1':>{COL['add']}} {'DelF1':>{COL['del']}} {'SubF1':>{COL['sub']}}  Note"
)
sep = "  " + H * (sum(COL.values()) + 10)

rows_text = "\n".join([
    "",
    "=" * (len(sep) - 2),
    "  MODEL EVALUATION COMPARISON  (Test Split)",
    "=" * (len(sep) - 2),
    header,
    sep,
    _row("CNN (Baseline)",
         cnn_acc, cnn_macro,
         cnn_f1["addition"], cnn_f1["deletion"], cnn_f1["substitution"],
         "addition always fails"),
    _row("CNN + Temporal Attention  (seed 42)",
         att_acc, att_macro,
         att_f1["addition"], att_f1["deletion"], att_f1["substitution"],
         "selected model, 4-speaker mixed test"),
    _row("CNN + Attention + Context  (seed 42, HQTV)",
         ctx_acc, ctx_macro,
         ctx_hqtv42["addition"], ctx_hqtv42["deletion"], ctx_hqtv42["substitution"],
         "speaker-disjoint, HQTV held-out"),
    sep,
    textwrap.dedent("""
      Notes
      -----
      • CNN Baseline: test set = 4 Vietnamese speakers (HQTV/PNV/THV/TLV), mixed train/val/test
      • CNN + Attention: same 4-speaker test set, seed 42 run (best epoch 1, val macro F1 0.5388)
      • CNN + Attention + Context: speaker-disjoint — HQTV fully held out from training,
        context window 0.1 s prepended, seed 42 (best epoch 3, val macro F1 0.5660)
      • Macro F1 gain:  CNN → CNN+Att: +0.0256   CNN+Att → CNN+Att+Ctx: +0.0576
      • Addition F1 improvement over baseline:  CNN+Att ×∞ (0→0.175), CNN+Att+Ctx ×∞ (0→0.173)
    """).rstrip(),
    "",
])

print(rows_text)

table_out = OUT_DIR / "model_comparison_table.txt"
table_out.write_text(rows_text, encoding="utf-8")
print(f"Saved: {table_out}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Confusion matrix plot — seed_42_HQTV (CNN + Attention + Context)
# ─────────────────────────────────────────────────────────────────────────────

tick_labels = ["Addition", "Deletion", "Substitution"]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=9)

ax.set(
    xticks=np.arange(3),
    yticks=np.arange(3),
    xticklabels=tick_labels,
    yticklabels=tick_labels,
    xlabel="Predicted label",
    ylabel="True label",
)
ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=10)
ax.set_yticklabels(tick_labels, fontsize=10)
ax.set_title(
    "Confusion Matrix — CNN + Attention + Context\n"
    "seed 42 · HQTV held-out · test split  (n = 1 358)",
    fontsize=11,
    pad=12,
)

# Row-normalised annotation: count (pct)
row_sums = cm.sum(axis=1, keepdims=True)
cm_norm = cm / row_sums

thresh = cm.max() / 2.0
for i in range(3):
    for j in range(3):
        pct = cm_norm[i, j] * 100
        ax.text(
            j, i,
            f"{cm[i, j]}\n({pct:.1f}%)",
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=11,
            fontweight="bold" if i == j else "normal",
        )

fig.tight_layout()
cm_out = OUT_DIR / "confusion_matrix_seed42_HQTV.png"
fig.savefig(cm_out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {cm_out}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-error-type F1 bar chart — 3 models compared
# ─────────────────────────────────────────────────────────────────────────────

error_labels = ["Addition", "Deletion", "Substitution"]
model_names  = ["CNN\n(Baseline)", "CNN +\nAttention\n(seed 42)", "CNN + Attn +\nContext\n(seed 42, HQTV)"]
f1_matrix = np.array([
    [cnn_f1["addition"],      cnn_f1["deletion"],      cnn_f1["substitution"]],
    [att_f1["addition"],      att_f1["deletion"],      att_f1["substitution"]],
    [ctx_hqtv42["addition"],  ctx_hqtv42["deletion"],  ctx_hqtv42["substitution"]],
])

colors = ["#4472C4", "#ED7D31", "#70AD47"]
x = np.arange(len(error_labels))
width = 0.24

fig2, ax2 = plt.subplots(figsize=(9, 5))

for i, (name, vals, color) in enumerate(zip(model_names, f1_matrix, colors)):
    offset = (i - 1) * width
    bars = ax2.bar(x + offset, vals, width, label=name, color=color,
                   edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars, vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=8.5,
        )

ax2.set_xlabel("Error Type", fontsize=11)
ax2.set_ylabel("F1 Score", fontsize=11)
ax2.set_title(
    "Per-Error-Type F1 — CNN vs CNN+Attention vs CNN+Attention+Context\n"
    "(Test split; Context model: speaker-disjoint HQTV fold, seed 42)",
    fontsize=11,
)
ax2.set_xticks(x)
ax2.set_xticklabels(error_labels, fontsize=11)
ax2.set_ylim(0, 0.96)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.85)
ax2.grid(axis="y", linestyle="--", alpha=0.35, color="grey")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

fig2.tight_layout()
f1_out = OUT_DIR / "per_error_type_f1_comparison.png"
fig2.savefig(f1_out, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {f1_out}")

print("\nDone. 3 files written to:", OUT_DIR)
