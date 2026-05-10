"""LR-selection line plot for the phase 13 figure pipeline.

Shows validation mean NLL across the small-slice learning-rate grid for
both LoRA conditions. Selected peak LR (3e-4 for both) is marked by a
dashed vertical line with a direct annotation.

Sources from the six ``lr-select-001-<condition>-lr-<lr>/summary.json``
files. Each summary's ``primary_metric.value`` is the run's
validation_mean_nll on the small-slice eval (128 validation rows, seed 7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from figures.helpers import (
    MARKERS,
    PALETTE,
    ROOT,
    read_json,
    rel_to_root,
    save_figure,
    set_paper_style,
    write_provenance,
)


FIG_NAME = "fig_06_lr_selection"

CONDITIONS = ("attention_only", "all_layer")
LR_LABELS = ("1e-4", "3e-4", "1e-3")
LR_VALUES = (1e-4, 3e-4, 1e-3)
SELECTED_LR_LABEL = "3e-4"


def build(output_root: Path) -> None:
    fig_dir = Path(output_root) / FIG_NAME
    set_paper_style()

    summary_paths: list[Path] = []
    nll_grid: dict[tuple[str, str], float] = {}
    for cond in CONDITIONS:
        for lr in LR_LABELS:
            p = ROOT / "artifacts" / "results" / f"lr-select-001-{cond}-lr-{lr}" / "summary.json"
            summary_paths.append(p)
            nll_grid[(cond, lr)] = float(read_json(p)["primary_metric"]["value"])

    selected_lr = LR_VALUES[LR_LABELS.index(SELECTED_LR_LABEL)]
    cond_display = {"attention_only": "attention-only", "all_layer": "all-layer"}
    callout_y_offset = {"attention_only": -0.0006, "all_layer": 0.0}

    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    ax.axvline(selected_lr, color="#999999", linestyle="--", linewidth=1.0, zorder=2)

    for cond in CONDITIONS:
        color = PALETTE[cond]
        marker = MARKERS[cond]
        nlls = [nll_grid[(cond, lr)] for lr in LR_LABELS]
        ax.plot(
            LR_VALUES, nlls,
            color=color, linewidth=2.0, marker=marker, markersize=8,
            markeredgecolor="white", markeredgewidth=1.0, zorder=3,
        )

    label_va = {"attention_only": "center", "all_layer": "top"}
    for cond in CONDITIONS:
        ax.text(
            LR_VALUES[0] * 1.18, nll_grid[(cond, "1e-4")],
            cond_display[cond],
            color=PALETTE[cond], fontsize=9,
            va=label_va[cond], ha="left",
        )

    for cond in CONDITIONS:
        nll = nll_grid[(cond, SELECTED_LR_LABEL)]
        ax.text(
            selected_lr * 1.05, nll + callout_y_offset[cond],
            f"{nll:.4f}",
            color=PALETTE[cond], fontsize=8.5,
            va="top", ha="left",
        )

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        selected_lr * 1.05, 0.99, "selected LR",
        transform=trans, color="#888888", fontsize=8.5,
        va="top", ha="left",
    )

    ax.set_xscale("log")
    ax.set_xticks(LR_VALUES)
    ax.set_xticklabels(LR_LABELS)
    ax.set_xlabel("Peak learning rate")
    ax.set_ylabel("Validation mean NLL")
    ax.set_title("LR selection: validation NLL across the LR grid")
    fig.tight_layout()

    plotted_data = [
        {"condition": cond, "lr": lr, "validation_mean_nll": nll_grid[(cond, lr)]}
        for cond in CONDITIONS for lr in LR_LABELS
    ]
    pdf_path, csv_path = save_figure(fig_dir, FIG_NAME, fig, plotted_data)
    plt.close(fig)

    caption_path = fig_dir / f"{FIG_NAME}.md"
    caption_path.write_text(_caption_markdown(nll_grid), encoding="utf-8")

    write_provenance(
        fig_dir, FIG_NAME,
        script=rel_to_root(__file__),
        inputs=summary_paths,
        derived_data_path=csv_path,
        figure_path=pdf_path,
    )


def _caption_markdown(nll_grid: dict[tuple[str, str], float]) -> str:
    att_sel = nll_grid[("attention_only", SELECTED_LR_LABEL)]
    al_sel  = nll_grid[("all_layer",      SELECTED_LR_LABEL)]
    att_1e3 = nll_grid[("attention_only", "1e-3")]
    al_1e3  = nll_grid[("all_layer",      "1e-3")]
    return f"""# Figure 6: LR selection: validation NLL across the LR grid

## Caption

Validation NLL on the small-slice learning-rate sweep used to select peak LR before the main training runs (512 train rows, 128 validation rows, seed 7). Both attention-only LoRA (blue, circles) and all-layer LoRA (orange, squares) minimize at 3e-4 — the selected peak LR for both conditions, marked by the dashed line. The valley shapes differ sharply: attention-only NLL is nearly flat after the minimum ({att_sel:.4f} at 3e-4, {att_1e3:.4f} at 1e-3, Δ +{att_1e3 - att_sel:.3f}), while all-layer rises steeply ({al_sel:.4f} at 3e-4, {al_1e3:.4f} at 1e-3, Δ +{al_1e3 - al_sel:.3f}), indicating greater LR sensitivity under the higher-capacity adapter.

## Marker encoding

| element | color | shape | what is plotted |
| --- | --- | --- | --- |
| `attention_only` | `#1f77b4` | circle (filled) | per-LR validation NLL on the small-slice sweep |
| `all_layer` | `#ff7f0e` | square (filled) | per-LR validation NLL on the small-slice sweep |
| dashed vertical line | `#999999` | dashed | selected peak LR (3e-4); NLL values annotated beside each condition's minimum marker |

## Source files

- Plotted points: `{FIG_NAME}.data.csv` (one row per condition × LR pair).
- Provenance and input SHA-256 hashes: `{FIG_NAME}.provenance.json`.
- Rendered: `{FIG_NAME}.pdf`, `{FIG_NAME}.png`, `{FIG_NAME}.grayscale.png`.
"""
