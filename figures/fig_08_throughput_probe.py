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
from matplotlib.transforms import blended_transform_factory

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

    x = list(range(len(REQUEST_SHAPES)))
    values = [secs_per_step[shape] for shape in REQUEST_SHAPES]
    baseline = values[0]
    multipliers = [
        "1×" if (baseline / v) < 1.05 else f"{baseline / v:.1f}×"
        for v in values
    ]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.axhline(baseline, color="#bbbbbb", linestyle="--", linewidth=0.8, zorder=1)

    bar_colors = ["#74c0dc", "#2d8fbf", "#1a5e8a"]
    bars = ax.bar(
        x, values,
        width=0.40, color=bar_colors, edgecolor="white", linewidth=1.0, zorder=3,
    )
    for rect, value, mult in zip(bars, values, multipliers):
        cx = rect.get_x() + rect.get_width() / 2
        ax.annotate(
            f"{value:.1f} s",
            xy=(cx, value), xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=8.5, color="#888888",
        )
        ax.annotate(
            mult,
            xy=(cx, value), xytext=(0, 17), textcoords="offset points",
            ha="center", va="bottom", fontsize=10, color="#333333",
        )

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.99, baseline, "baseline", transform=trans,
            ha="right", va="bottom", color="#bbbbbb", fontsize=8)

    ax.text(0.02, 0.97, "lower = faster", transform=ax.transAxes,
            fontsize=8, color="#aaaaaa", va="top", ha="left", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(["Sequential", "Batched", "Pipelined"])
    ax.set_xlim(-0.6, len(REQUEST_SHAPES) - 0.4)
    ax.set_ylim(0, max(values) * 1.35)
    ax.set_ylabel("Seconds per optimizer step")
    ax.set_xlabel("Request shape")
    ax.set_title("Throughput probe: seconds per optimizer step by request shape")
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
    return f"""# Figure 8: Throughput probe: seconds per optimizer step by request shape

## Caption

Pipelined batched training calls deliver a {single / pipelined:.1f}× throughput improvement over sequential single-datum calls on Qwen3-8B with attention-only LoRA (effective batch size 8). Batching alone — submitting all 8 examples in one API call — accounts for {single / batched:.1f}× of the speedup; pipelining, which submits the next batch before awaiting the prior optimizer step, contributes an additional {batched / pipelined:.1f}× by overlapping communication with computation. Measured wall-clock seconds per optimizer step: sequential {single:.1f} s, batched {batched:.1f} s, pipelined {pipelined:.1f} s. The pipelined shape was adopted for all main training runs.

## Marker encoding

| element | color | shape | what is plotted |
| --- | --- | --- | --- |
| bars | `#1f77b4` | filled rectangle | mean wall-clock seconds per optimizer step over 16 probe steps |
| value labels | `#333333` | text above bar | seconds per optimizer step rounded to two decimals |

## Source files

- Plotted points: `{FIG_NAME}.data.csv` (one row per request shape).
- Provenance and input SHA-256 hashes: `{FIG_NAME}.provenance.json`.
- Rendered: `{FIG_NAME}.pdf`, `{FIG_NAME}.png`, `{FIG_NAME}.grayscale.png`.
"""
