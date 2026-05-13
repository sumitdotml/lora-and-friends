"""Throughput-probe appendix plot for the phase 13 figure pipeline.

Compares wall-clock seconds per optimizer step across three Tinker
training-request shapes at effective batch size 8 on attention-only LoRA:
single-datum sequential, batched, and batched-pipelined.

Sources from ``throughput-probe-001/summary.json`` (single_datum_calls and
batched_datums) and ``throughput-probe-batch8-pipelined-001/summary.json``
(batched_datums_pipelined).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from figures.helpers import (
    ROOT,
    read_json,
    rel_to_root,
    save_figure,
    set_paper_style,
    write_provenance,
)


FIG_NAME = "fig_08_throughput_probe"

REQUEST_SHAPES = ("single_datum_calls", "batched_datums", "batched_datums_pipelined")
PROBE_SOURCES = (
    ("throughput-probe-001", ("single_datum_calls", "batched_datums")),
    ("throughput-probe-batch8-pipelined-001", ("batched_datums_pipelined",)),
)


def build(output_root: Path) -> None:
    fig_dir = Path(output_root) / FIG_NAME
    set_paper_style()

    summary_paths: list[Path] = []
    secs_per_step: dict[str, float] = {}
    for run_id, shapes in PROBE_SOURCES:
        p = ROOT / "artifacts" / "results" / run_id / "summary.json"
        summary_paths.append(p)
        s = read_json(p)
        for entry in s["request_shapes"]:
            shape = entry["request_shape"]
            if shape in shapes:
                secs_per_step[shape] = float(entry["seconds_per_optimizer_step"])

    missing = [shape for shape in REQUEST_SHAPES if shape not in secs_per_step]
    if missing:
        raise RuntimeError(f"missing seconds_per_optimizer_step for shapes: {missing}")

    values = [secs_per_step[shape] for shape in REQUEST_SHAPES]
    baseline = values[0]
    multipliers = [
        "1×" if (baseline / v) < 1.05 else f"{baseline / v:.1f}×"
        for v in values
    ]
    labels = ["Sequential", "Batched", "Pipelined"]
    y = list(range(len(REQUEST_SHAPES)))

    fig, ax = plt.subplots(figsize=(6.3, 3.8))

    bar_colors = ["#d95f02", "#1b9e77", "#7570b3"]
    bars = ax.barh(
        y, values,
        height=0.52, color=bar_colors, edgecolor="white", linewidth=1.0, zorder=3,
    )
    for rect, value, mult in zip(bars, values, multipliers):
        cy = rect.get_y() + rect.get_height() / 2
        label = f"{value:.1f} s"
        if mult != "1×":
            label = f"{label}  ({mult} faster)"
        ax.annotate(
            label,
            xy=(value, cy), xytext=(7, 0), textcoords="offset points",
            ha="left", va="center", fontsize=9.5, color="#333333",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_xlabel("Seconds per optimizer step (lower = faster)")
    ax.set_title("Throughput by request shape")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#777777")
    ax.spines["bottom"].set_linewidth(0.8)
    fig.tight_layout()

    plotted_data = [
        {"request_shape": shape, "seconds_per_optimizer_step": secs_per_step[shape]}
        for shape in REQUEST_SHAPES
    ]
    pdf_path, csv_path = save_figure(fig_dir, FIG_NAME, fig, plotted_data)
    plt.close(fig)

    caption_path = fig_dir / f"{FIG_NAME}.md"
    caption_path.write_text(_caption_markdown(secs_per_step), encoding="utf-8")

    write_provenance(
        fig_dir, FIG_NAME,
        script=rel_to_root(__file__),
        inputs=summary_paths,
        derived_data_path=csv_path,
        figure_path=pdf_path,
    )


def _caption_markdown(secs_per_step: dict[str, float]) -> str:
    single = secs_per_step["single_datum_calls"]
    batched = secs_per_step["batched_datums"]
    pipelined = secs_per_step["batched_datums_pipelined"]
    return f"""# Figure 8: Throughput by request shape

## Caption

Wall-clock seconds per optimizer step for Qwen3-8B attention-only LoRA at effective batch size 8. Batching cuts the step time from {single:.1f} s to {batched:.1f} s, and pipelined batching lowers it to {pipelined:.1f} s ({single / pipelined:.1f}× faster than sequential).

## Marker encoding

| element | color | shape | what is plotted |
| --- | --- | --- | --- |
| sequential | `#d95f02` | horizontal rectangle | mean wall-clock seconds per optimizer step over 16 probe steps |
| batched | `#1b9e77` | horizontal rectangle | mean wall-clock seconds per optimizer step over 16 probe steps |
| pipelined | `#7570b3` | horizontal rectangle | mean wall-clock seconds per optimizer step over 16 probe steps |
| value labels | `#333333` | text at bar end | seconds per optimizer step and speedup versus sequential |

## Source files

- Plotted points: `{FIG_NAME}.data.csv` (one row per request shape).
- Provenance and input SHA-256 hashes: `{FIG_NAME}.provenance.json`.
- Rendered: `{FIG_NAME}.pdf`, `{FIG_NAME}.png`, `{FIG_NAME}.grayscale.png`.
"""
