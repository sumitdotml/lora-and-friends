"""GSM8K accuracy dot/range chart for the phase 13 figure pipeline.

Three x-positions: ``baseline``, ``attention_only``, ``all_layer``. Baseline
is a single gray diamond (N=1, no whiskers). LoRA conditions show mean
(large marker), individual seeds (small jittered markers), and min/max
whiskers. Y-axis restricted to ``[0.80, 0.92]`` to surface seed-level
variation.

Sources baseline accuracy from ``baseline-qwen3-8b-gsm8k-001/summary.json``
and per-condition seed accuracies from the six ``checkpoint-*-gsm8k-*/summary.json``
files. Emits PDF + PNG + grayscale PNG, plus a CSV of plotted points and a
provenance JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from figures.helpers import (
    BASELINE_RUN_DIR,
    MARKERS,
    PALETTE,
    find_main_001_evals,
    read_json,
    rel_to_root,
    save_figure,
    set_paper_style,
    write_provenance,
)


FIG_NAME = "fig_03_gsm8k_accuracy_chart"

ALL_CONDITIONS = ("baseline", "attention_only", "all_layer")
LORA_CONDITIONS = ("attention_only", "all_layer")
SEEDS = (0, 1, 2)
Y_RANGE = (0.80, 0.92)


def build(output_root: Path) -> None:
    fig_dir = Path(output_root) / FIG_NAME
    set_paper_style()

    baseline_summary_path = BASELINE_RUN_DIR / "summary.json"
    baseline = read_json(baseline_summary_path)
    baseline_acc = baseline["primary_metric"]["value"]

    eval_dirs = find_main_001_evals()
    eval_summary_paths = [eval_dirs[k] / "summary.json" for k in sorted(eval_dirs)]
    eval_summaries = {
        (s["condition"], s["seed"]): s
        for s in (read_json(p) for p in eval_summary_paths)
    }

    seed_accs: dict[str, list[float]] = {
        cond: [eval_summaries[(cond, seed)]["primary_metric"]["value"] for seed in SEEDS]
        for cond in LORA_CONDITIONS
    }

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    seed_x_offsets = np.linspace(-0.07, 0.07, len(SEEDS))
    labels = {
        "baseline":       "baseline",
        "attention_only": "attention-only",
        "all_layer":      "all-layer",
    }

    for x, cond in enumerate(ALL_CONDITIONS):
        color = PALETTE[cond]
        marker = MARKERS[cond]
        if cond == "baseline":
            ax.scatter(
                [x], [baseline_acc],
                color=color, s=140, marker=marker,
                edgecolors="white", linewidths=1.6, zorder=4,
            )
            continue

        accs = seed_accs[cond]
        mean_acc = sum(accs) / len(accs)

        ax.scatter(
            x + seed_x_offsets, accs,
            color=color, s=32, marker=marker, alpha=0.55,
            edgecolors="none", zorder=3,
        )
        ax.scatter(
            [x], [mean_acc],
            color=color, s=140, marker=marker,
            edgecolors="white", linewidths=1.6, zorder=4,
        )

    ax.set_xticks(range(len(ALL_CONDITIONS)))
    ax.set_xticklabels([labels[c] for c in ALL_CONDITIONS])
    ax.set_xlim(-0.5, len(ALL_CONDITIONS) - 0.5)
    ax.set_ylim(*Y_RANGE)
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("GSM8K accuracy by condition")
    ax.grid(False)
    ax.yaxis.grid(True, color="#e7e7e7", linewidth=0.7)
    ax.spines["left"].set_color("#777777")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("#777777")
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, color="#777777")
    fig.tight_layout()

    plotted_data = _build_plotted_data(baseline_acc, seed_accs)

    pdf_path, csv_path = save_figure(fig_dir, FIG_NAME, fig, plotted_data)
    plt.close(fig)

    caption_path = fig_dir / f"{FIG_NAME}.md"
    caption_path.write_text(_caption_markdown(baseline_acc, seed_accs), encoding="utf-8")

    write_provenance(
        fig_dir, FIG_NAME,
        script=rel_to_root(__file__),
        inputs=[baseline_summary_path, *eval_summary_paths],
        derived_data_path=csv_path,
        figure_path=pdf_path,
    )


def _caption_markdown(
    baseline_acc: float,
    seed_accs: dict[str, list[float]],
) -> str:
    att_mean = sum(seed_accs["attention_only"]) / len(seed_accs["attention_only"])
    al_mean = sum(seed_accs["all_layer"]) / len(seed_accs["all_layer"])
    return f"""# Figure 3: GSM8K accuracy by condition (Qwen3-8B)

## Caption

GSM8K accuracy by condition on the 1,319-example test set. Untouched Qwen3-8B baseline (gray diamond, N=1) and two LoRA conditions at their step-3169 checkpoints (large marker = mean across 3 seeds, small markers = individual seeds). Mean accuracies: baseline {baseline_acc:.4f}, attention-only {att_mean:.4f}, all-layer {al_mean:.4f}. Y-axis truncated to [0.80, 0.92] to surface seed-level variation; no statistical confidence interval is shown.

## Marker encoding

| condition | color | shape | what is plotted |
| --- | --- | --- | --- |
| `baseline` | `#666666` | diamond | single point at the untouched-model accuracy |
| `attention_only` | `#1f77b4` | circle | mean (large marker) plus three jittered seed accuracies (small markers) |
| `all_layer` | `#ff7f0e` | square | mean (large marker) plus three jittered seed accuracies (small markers) |

## Source files

- Plotted points: `{FIG_NAME}.data.csv` (nine rows: 1 baseline + 6 individual seed accuracies + 2 condition means).
- Provenance and input SHA-256 hashes: `{FIG_NAME}.provenance.json`.
- Rendered: `{FIG_NAME}.pdf`, `{FIG_NAME}.png`, `{FIG_NAME}.grayscale.png`.
"""


def _build_plotted_data(
    baseline_acc: float,
    seed_accs: dict[str, list[float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{
        "kind":      "baseline",
        "condition": "baseline",
        "seed":      "",
        "accuracy":  baseline_acc,
    }]
    for cond in LORA_CONDITIONS:
        accs = seed_accs[cond]
        for seed, acc in zip(SEEDS, accs):
            rows.append({
                "kind":      "seed",
                "condition": cond,
                "seed":      seed,
                "accuracy":  acc,
            })
        rows.append({
            "kind":      "mean",
            "condition": cond,
            "seed":      "",
            "accuracy":  sum(accs) / len(accs),
        })
    return rows
