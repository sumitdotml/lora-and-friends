"""Per-seed prediction-disagreement small-multiples for the phase 13 figure pipeline.

For each of the three training seeds, two bars are plotted:

- ``attention_only_only`` (blue): examples where attention-only is correct
  AND all-layer is wrong.
- ``all_layer_only`` (orange, hatched): examples where all-layer is
  correct AND attention-only is wrong.

The blue bar is taller than the orange bar in every panel, which is the
headline: attention-only wins not by a uniform shift in net accuracy but
because it is correct on more examples that all-layer misses than the
reverse. ``both_correct`` and ``both_wrong`` agreement counts are shown
as text under each panel; the full contingency is in fig_07.

Sources predictions from the six ``checkpoint-*-gsm8k-*/predictions.jsonl``
files; pairs by ``benchmark_index`` within a seed. Emits PDF + PNG +
grayscale PNG, plus a CSV of plotted points, provenance JSON, and a
paper-style caption.

The module/file name (``paired_seed_slope``) is retained from an earlier
iteration so existing artifact paths and provenance hashes do not move;
the chart itself is the disagreement small-multiples described above.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from figures.helpers import (
    PALETTE,
    find_main_001_evals,
    read_jsonl,
    rel_to_root,
    save_figure,
    set_paper_style,
    write_provenance,
)


FIG_NAME = "fig_04_paired_seed_slope"

CONDITIONS = ("attention_only", "all_layer")
SEEDS = (0, 1, 2)
TOTAL_EXAMPLES = 1319


def build(output_root: Path) -> None:
    fig_dir = Path(output_root) / FIG_NAME
    set_paper_style()

    eval_dirs = find_main_001_evals()
    predictions_paths = [eval_dirs[k] / "predictions.jsonl" for k in sorted(eval_dirs)]
    predictions = {k: read_jsonl(eval_dirs[k] / "predictions.jsonl") for k in eval_dirs}

    paired = _pair_predictions(predictions)

    fig, axes = plt.subplots(1, len(SEEDS), figsize=(8.5, 4.4), sharey=True)

    ymax = max(max(paired[s]["attention_only_only"], paired[s]["all_layer_only"]) for s in SEEDS)
    ymax_padded = int(ymax * 1.30)

    for ax, seed in zip(axes, SEEDS):
        p = paired[seed]
        values = [p["attention_only_only"], p["all_layer_only"]]
        bars = ax.bar(
            [0, 1], values,
            width=0.55,
            color=[PALETTE["attention_only"], PALETTE["all_layer"]],
            edgecolor="white", linewidth=1.0, zorder=3,
        )
        bars[1].set_hatch("///")

        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=10, color="#333333",
            )

        ax.set_title(f"Seed {seed}")
        ax.set_xticks([])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(0, ymax_padded)
        ax.grid(axis="x", visible=False)

        agreement = (
            f"both right: {p['both_correct']:,}\n"
            f"both wrong: {p['both_wrong']:,}"
        )
        ax.text(
            0.5, -0.10, agreement,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9, color="#777777",
        )

    axes[0].set_ylabel(f"Test examples (of {TOTAL_EXAMPLES:,})")

    legend_handles = [
        Patch(
            facecolor=PALETTE["attention_only"], edgecolor="white",
            label="attention-only correct, all-layer wrong",
        ),
        Patch(
            facecolor=PALETTE["all_layer"], edgecolor="white", hatch="///",
            label="all-layer correct, attention-only wrong",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02),
        frameon=False, fontsize=10,
    )

    fig.suptitle("Per-seed prediction disagreement on GSM8K", y=0.99)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))

    plotted_data = _build_plotted_data(paired)
    pdf_path, csv_path = save_figure(fig_dir, FIG_NAME, fig, plotted_data)
    plt.close(fig)

    caption_path = fig_dir / f"{FIG_NAME}.md"
    caption_path.write_text(_caption_markdown(paired), encoding="utf-8")

    write_provenance(
        fig_dir, FIG_NAME,
        script=rel_to_root(__file__),
        inputs=predictions_paths,
        derived_data_path=csv_path,
        figure_path=pdf_path,
    )


def _pair_predictions(
    predictions: dict[tuple[str, int], list[dict[str, Any]]],
) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = {}
    for seed in SEEDS:
        att_rows = {r["benchmark_index"]: bool(r["correct"]) for r in predictions[("attention_only", seed)]}
        all_rows = {r["benchmark_index"]: bool(r["correct"]) for r in predictions[("all_layer", seed)]}
        if set(att_rows) != set(all_rows):
            raise RuntimeError(
                f"benchmark_index sets differ for seed {seed}: "
                f"attention_only has {len(att_rows)}, all_layer has {len(all_rows)}",
            )
        att_only = sum(1 for i in att_rows if att_rows[i] and not all_rows[i])
        all_only = sum(1 for i in att_rows if all_rows[i] and not att_rows[i])
        both_correct = sum(1 for i in att_rows if att_rows[i] and all_rows[i])
        both_wrong = sum(1 for i in att_rows if not att_rows[i] and not all_rows[i])
        total = att_only + all_only + both_correct + both_wrong
        if total != TOTAL_EXAMPLES:
            raise RuntimeError(
                f"seed {seed}: paired counts sum to {total}, expected {TOTAL_EXAMPLES}",
            )
        out[seed] = {
            "attention_only_only": att_only,
            "all_layer_only":      all_only,
            "both_correct":        both_correct,
            "both_wrong":          both_wrong,
        }
    return out


def _build_plotted_data(paired: dict[int, dict[str, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        p = paired[seed]
        rows.append({
            "seed":                seed,
            "attention_only_only": p["attention_only_only"],
            "all_layer_only":      p["all_layer_only"],
            "both_correct":        p["both_correct"],
            "both_wrong":          p["both_wrong"],
            "disagreement_delta":  p["attention_only_only"] - p["all_layer_only"],
        })
    return rows


def _caption_markdown(paired: dict[int, dict[str, int]]) -> str:
    deltas = ", ".join(
        f"+{paired[s]['attention_only_only'] - paired[s]['all_layer_only']}"
        for s in SEEDS
    )
    return f"""# Figure 4: Per-seed prediction disagreement on GSM8K

## Caption

Per-seed prediction disagreement between attention-only and all-layer LoRA on the GSM8K test set ({TOTAL_EXAMPLES:,} examples). Each panel covers one training seed; the blue bar counts examples where attention-only is correct and all-layer is wrong, and the orange hatched bar counts the reverse. The blue bar is taller in every panel, with disagreement deltas of {deltas} examples across seeds 0, 1, 2 — so attention-only wins not by a uniform shift in net accuracy but because it is correct on more questions that all-layer misses than the reverse. Agreement counts (both correct, both wrong) are annotated under each panel.

## Marker encoding

| element | color | fill | what is plotted |
| --- | --- | --- | --- |
| left bar | `{PALETTE['attention_only']}` | solid | examples where attention-only is correct AND all-layer is wrong |
| right bar | `{PALETTE['all_layer']}` | diagonal hatch (grayscale-fallback marker) | examples where all-layer is correct AND attention-only is wrong |
| sub-panel annotation | `#777777` | text | both-correct and both-wrong counts (the agreement region not shown as bars) |

## Source files

- Plotted points: `{FIG_NAME}.data.csv` (one row per seed; disagreement counts, agreement counts, and the disagreement delta).
- Provenance and input SHA-256 hashes: `{FIG_NAME}.provenance.json`.
- Rendered: `{FIG_NAME}.pdf`, `{FIG_NAME}.png`, `{FIG_NAME}.grayscale.png`.

## Appendix

For the full per-seed contingency table and one concrete disagreement example per direction per seed (complete question text and both models' full generations), see [Figure 7](../fig_07_prediction_disagreement_table/fig_07_prediction_disagreement_table.md).
"""
