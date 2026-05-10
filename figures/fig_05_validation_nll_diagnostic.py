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
from matplotlib.transforms import blended_transform_factory

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

    max_step = max(s for curve in val_curves.values() for s, _ in curve)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # post-minimum background band (behind everything)
    ax.axvspan(SELECTED_STEP, max_step + 500, color="#f0f0f0", linewidth=0, zorder=0)

    # dashed selection line — annotated directly, not via legend
    ax.axvline(SELECTED_STEP, color="#999999", linestyle="--", linewidth=1.0, zorder=2)

    # faint per-seed lines first so they sit beneath the bold means
    for cond in CONDITIONS:
        color = PALETTE[cond]
        for seed in SEEDS:
            steps, nlls = zip(*val_curves[(cond, seed)])
            ax.plot(steps, nlls, color=color, alpha=0.30, linewidth=1.0, zorder=2)

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
        ax.plot(
            steps_sorted, means,
            color=color, linewidth=2.2, marker=marker, markersize=7,
            markeredgecolor="white", markeredgewidth=1.0, zorder=3, label=cond,
        )

    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Validation mean NLL")
    ax.set_title("Validation NLL by training step (main-001, both LoRA conditions)")
    ax.legend(loc="lower left")

    # direct annotation on the dashed line (x = data coords, y = axes fraction)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        SELECTED_STEP + 80, 0.99,
        f"selected checkpoint\n(step {SELECTED_STEP})",
        transform=trans, color="#888888", fontsize=8.5,
        va="top", ha="left", linespacing=1.4,
    )

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
    selected_values = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            for step, nll in val_curves[(cond, seed)]:
                if step == SELECTED_STEP:
                    selected_values.append(nll)
    selected_min = min(selected_values)
    selected_max = max(selected_values)

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

Validation negative log-likelihood over training for attention-only LoRA (blue, circles) and all-layer LoRA (orange, squares). Bold lines show per-condition means across three seeds; faint lines show individual seed trajectories. Both conditions reach minimum NLL at step {SELECTED_STEP} (dashed line), the checkpoint selected by the frozen validation-loss rule. After the minimum, attention-only NLL remains nearly flat — never exceeding its minimum by more than {att_post - att_min:.3f} — while all-layer NLL rises sharply, gaining ~{all_post_delta:.3f} by step {first_post_step} and returning to near its step-1000 level, consistent with earlier overfitting under the higher-capacity adapter.

## Marker encoding

| element | color | style | what is plotted |
| --- | --- | --- | --- |
| `attention_only` mean | `#1f77b4` | solid bold | per-step mean of validation NLL across 3 seeds |
| `all_layer` mean | `#ff7f0e` | solid bold | per-step mean of validation NLL across 3 seeds |
| individual seeds | condition color, alpha 0.30 | thin lines | per-seed validation NLL trajectory |
| selected step | `#999999` | vertical dashed | step {SELECTED_STEP}, the frozen checkpoint-selection rule's min-validation-NLL choice |

## Source files

- Plotted points: `{FIG_NAME}.data.csv` (per-seed and per-condition mean rows for each retained validation step).
- Provenance and input SHA-256 hashes: `{FIG_NAME}.provenance.json`.
- Rendered: `{FIG_NAME}.pdf`, `{FIG_NAME}.png`, `{FIG_NAME}.grayscale.png`.
"""
