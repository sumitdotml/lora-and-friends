"""Validation NLL checkpoint-selection diagnostic for the phase 13 figure pipeline.

Plots validation mean NLL over training optimizer steps for both LoRA
conditions. Faint per-seed lines plus bold per-condition mean line; vertical
marker at the selected step (3169).

Sources from the six ``main-001-<condition>-seed-<seed>/metrics.jsonl``
files. Each file is filtered to rows with ``split == "main_val"``; their
``eval_metric.value`` is the validation_mean_nll plotted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from figures.helpers import (
    MARKERS,
    PALETTE,
    find_main_001_train_dirs,
    read_jsonl,
    rel_to_root,
    save_figure,
    set_paper_style,
    write_provenance,
)


FIG_NAME = "fig_05_validation_nll_diagnostic"

CONDITIONS = ("attention_only", "all_layer")
SEEDS = (0, 1, 2)
SELECTED_STEP = 3169


def build(output_root: Path) -> None:
    fig_dir = Path(output_root) / FIG_NAME
    set_paper_style()

    train_dirs = find_main_001_train_dirs()
    metrics_paths = [train_dirs[k] / "metrics.jsonl" for k in sorted(train_dirs)]

    val_curves: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for (cond, seed), d in train_dirs.items():
        rows = read_jsonl(d / "metrics.jsonl")
        val_rows = [r for r in rows if r.get("split") == "main_val"]
        val_curves[(cond, seed)] = sorted(
            (int(r["step"]), float(r["eval_metric"]["value"])) for r in val_rows
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # faint per-seed lines first so they sit beneath the bold means
    selected_means: dict[str, float] = {}
    for cond in CONDITIONS:
        color = PALETTE[cond]
        for seed in SEEDS:
            steps, nlls = zip(*val_curves[(cond, seed)])
            ax.plot(steps, nlls, color=color, alpha=0.24, linewidth=1.0, zorder=2)

    for cond in CONDITIONS:
        color = PALETTE[cond]
        marker = MARKERS[cond]
        steps_sorted = sorted({s for (cnd, _), curve in val_curves.items()
                               if cnd == cond for (s, _) in curve})
        means = []
        for step in steps_sorted:
            seed_values = [
                v for seed in SEEDS
                for (s, v) in val_curves[(cond, seed)]
                if s == step
            ]
            means.append(sum(seed_values) / len(seed_values))
        selected_means[cond] = means[steps_sorted.index(SELECTED_STEP)]
        ax.plot(
            steps_sorted, means,
            color=color, linewidth=2.1, zorder=3,
        )
        normal_points = [
            (step, mean) for step, mean in zip(steps_sorted, means)
            if step != SELECTED_STEP
        ]
        normal_steps, normal_means = zip(*normal_points)
        ax.scatter(
            normal_steps, normal_means,
            s=32, marker=marker, facecolors=color, edgecolors="white",
            linewidths=0.7, zorder=4,
        )

        label = {"attention_only": "attention-only", "all_layer": "all-layer"}[cond]
        ax.text(
            steps_sorted[-1] + 120, means[-1],
            label, color=color, fontsize=9.5,
            va="center", ha="left",
        )

    ax.scatter(
        [SELECTED_STEP], [selected_means["all_layer"]],
        s=42, marker=MARKERS["all_layer"], facecolors=PALETTE["all_layer"],
        edgecolors="white", linewidths=0.9, zorder=6,
    )
    ax.scatter(
        [SELECTED_STEP], [selected_means["attention_only"]],
        s=30, marker=MARKERS["attention_only"], facecolors=PALETTE["attention_only"],
        edgecolors="white", linewidths=0.9, zorder=7,
    )

    ax.set_xlim(850, 6650)
    ax.set_ylim(0.3351, 0.3474)
    ax.set_xticks([1000, 2000, SELECTED_STEP, 4000, 5000, 6000])
    ax.set_xticklabels(["1000", "2000", "3169", "4000", "5000", "6000"])
    ax.set_yticks(np.arange(0.336, 0.347, 0.002))
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Validation mean NLL")
    ax.set_title("Validation NLL over training")
    ax.grid(False)
    ax.yaxis.grid(True, color="#e7e7e7", linewidth=0.7)
    ax.spines["left"].set_color("#777777")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_color("#777777")
    ax.spines["bottom"].set_linewidth(0.8)

    fig.tight_layout()

    plotted_data = _build_plotted_data(val_curves)
    pdf_path, csv_path = save_figure(fig_dir, FIG_NAME, fig, plotted_data)
    plt.close(fig)

    caption_path = fig_dir / f"{FIG_NAME}.md"
    caption_path.write_text(_caption_markdown(val_curves), encoding="utf-8")

    write_provenance(
        fig_dir, FIG_NAME,
        script=rel_to_root(__file__),
        inputs=metrics_paths,
        derived_data_path=csv_path,
        figure_path=pdf_path,
    )


def _build_plotted_data(
    val_curves: dict[tuple[str, int], list[tuple[int, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            for step, nll in val_curves[(cond, seed)]:
                rows.append({
                    "kind":      "seed",
                    "condition": cond,
                    "seed":      seed,
                    "step":      step,
                    "validation_mean_nll": nll,
                })
        steps_sorted = sorted({s for s, _ in val_curves[(cond, 0)]})
        for step in steps_sorted:
            seed_values = [
                v for seed in SEEDS
                for (s, v) in val_curves[(cond, seed)]
                if s == step
            ]
            rows.append({
                "kind":      "mean",
                "condition": cond,
                "seed":      "",
                "step":      step,
                "validation_mean_nll": sum(seed_values) / len(seed_values),
            })
    return rows


def _caption_markdown(
    val_curves: dict[tuple[str, int], list[tuple[int, float]]],
) -> str:
    selected_means = {}
    for cond in CONDITIONS:
        selected_values = [
            v for seed in SEEDS
            for (s, v) in val_curves[(cond, seed)]
            if s == SELECTED_STEP
        ]
        selected_means[cond] = sum(selected_values) / len(selected_values)

    # per-condition NLL delta after the minimum, for caption numbers
    att_min  = min(v for seed in (0, 1, 2) for (s, v) in val_curves[("attention_only", seed)] if s == SELECTED_STEP)
    att_post = max(v for seed in (0, 1, 2) for (s, v) in val_curves[("attention_only", seed)] if s > SELECTED_STEP)
    all_step4k = {s: [] for s in {s for seed in (0, 1, 2) for (s, _) in val_curves[("all_layer", seed)] if s > SELECTED_STEP}}
    for seed in (0, 1, 2):
        for s, v in val_curves[("all_layer", seed)]:
            if s in all_step4k:
                all_step4k[s].append(v)
    first_post_step = min(all_step4k)
    all_first_post_mean = sum(all_step4k[first_post_step]) / len(all_step4k[first_post_step])
    all_min = min(v for seed in (0, 1, 2) for (s, v) in val_curves[("all_layer", seed)] if s == SELECTED_STEP)
    all_post_delta = all_first_post_mean - all_min

    return f"""# Figure 5: Validation NLL by training step (main-001, both LoRA conditions)

## Caption

Validation negative log-likelihood over training for attention-only LoRA (blue, circles) and all-layer LoRA (orange, squares). Bold lines show per-condition means across three seeds; faint lines show individual seed trajectories. Both conditions reach minimum NLL at step {SELECTED_STEP}, the checkpoint selected by the frozen validation-loss rule; the plotted means are close but not identical ({selected_means["attention_only"]:.6f} for attention-only vs. {selected_means["all_layer"]:.6f} for all-layer). After the minimum, attention-only NLL remains nearly flat — never exceeding its minimum by more than {att_post - att_min:.3f} — while all-layer NLL rises sharply, gaining ~{all_post_delta:.3f} by step {first_post_step} and returning to near its step-1000 level, consistent with earlier overfitting under the higher-capacity adapter.

## Marker encoding

| element | color | style | what is plotted |
| --- | --- | --- | --- |
| `attention_only` mean | `#1f77b4` | solid bold | per-step mean of validation NLL across 3 seeds |
| `all_layer` mean | `#ff7f0e` | solid bold | per-step mean of validation NLL across 3 seeds |
| selected step | condition colors | compact overlapping markers | shared minimum checkpoint at step {SELECTED_STEP} |
| individual seeds | condition color, alpha 0.24 | thin lines | per-seed validation NLL trajectory |

## Source files

- Plotted points: `{FIG_NAME}.data.csv` (per-seed and per-condition mean rows for each retained validation step).
- Provenance and input SHA-256 hashes: `{FIG_NAME}.provenance.json`.
- Rendered: `{FIG_NAME}.pdf`, `{FIG_NAME}.png`, `{FIG_NAME}.grayscale.png`.
"""
