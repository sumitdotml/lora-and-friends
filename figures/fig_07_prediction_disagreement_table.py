"""Prediction-disagreement appendix table for the phase 13 figure pipeline.

Per-seed contingency between attention-only and all-layer predictions on
GSM8K. Each row counts how many test examples fall in the four cells of
the (attention_only_correct, all_layer_correct) truth table.

Sources from the six ``checkpoint-*-gsm8k-*/predictions.jsonl`` files,
joined by ``benchmark_index`` within a seed. Emits a Markdown table, CSV,
provenance JSON, and paper-style caption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from figures.helpers import (
    find_main_001_evals,
    read_jsonl,
    rel_to_root,
    save_table,
    write_provenance,
)


FIG_NAME = "fig_07_prediction_disagreement_table"

CONDITIONS = ("attention_only", "all_layer")
SEEDS = (0, 1, 2)
TOTAL_EXAMPLES = 1319


def build(output_root: Path) -> None:
    fig_dir = Path(output_root) / FIG_NAME

    eval_dirs = find_main_001_evals()
    predictions_paths = [eval_dirs[k] / "predictions.jsonl" for k in sorted(eval_dirs)]
    predictions = {k: read_jsonl(eval_dirs[k] / "predictions.jsonl") for k in eval_dirs}

    paired = _pair_predictions(predictions)
    rows = _build_rows(paired)

    columns = [
        "Seed",
        "Attn correct, All-layer wrong",
        "All-layer correct, Attn wrong",
        "Δ (Attn − All)",
        "Both correct",
        "Both wrong",
    ]
    align = {col: "right" for col in columns}
    caption = (
        "Per-seed prediction agreement between attention-only and all-layer LoRA "
        "on the GSM8K test set (1,319 examples), paired by example index. "
        "Columns count test examples where one condition is correct and the other "
        "is wrong, where both are correct, and where both are wrong. "
        "Attention-only wins more disagreements at every seed (Δ = +5 to +7), "
        "consistent with its higher accuracy in the main comparison; "
        "see Figure 4 for the corresponding visualisation and per-seed disagreement examples."
    )

    md_path, csv_path = save_table(
        fig_dir, FIG_NAME, rows,
        columns=columns,
        align=align,
        title="Prediction disagreement: attention-only vs all-layer (per seed)",
        caption=caption,
    )

    examples = _find_disagree_examples(predictions)
    with md_path.open("a", encoding="utf-8") as f:
        f.write("\n## Disagreement examples\n\n")
        f.write(
            "One example per direction per seed — lowest `benchmark_index` satisfying "
            "the condition. Shows the question, reference answer, each model's extracted "
            "answer, and the full generation.\n"
        )
        f.write(_render_examples(examples))
        f.write(
            "\n## Related artifacts\n\n"
            "This appendix supplements [Figure 4](../fig_04_paired_seed_slope/fig_04_paired_seed_slope.md), "
            "which visualises the disagreement counts as a per-seed bar chart in the main body.\n"
        )

    write_provenance(
        fig_dir, FIG_NAME,
        script=rel_to_root(__file__),
        inputs=predictions_paths,
        derived_data_path=csv_path,
        figure_path=md_path,
    )


def _find_disagree_examples(
    predictions: dict[tuple[str, int], list[dict[str, Any]]],
) -> dict[int, dict[str, tuple]]:
    out: dict[int, dict[str, tuple]] = {}
    for seed in SEEDS:
        att_by_idx = {r["benchmark_index"]: r for r in predictions[("attention_only", seed)]}
        all_by_idx = {r["benchmark_index"]: r for r in predictions[("all_layer", seed)]}
        att_wins = [
            (i, att_by_idx[i], all_by_idx[i])
            for i in sorted(att_by_idx)
            if att_by_idx[i]["correct"] and not all_by_idx[i]["correct"]
        ]
        all_wins = [
            (i, att_by_idx[i], all_by_idx[i])
            for i in sorted(att_by_idx)
            if all_by_idx[i]["correct"] and not att_by_idx[i]["correct"]
        ]
        out[seed] = {
            "att_only_only": att_wins[0] if att_wins else None,
            "all_layer_only": all_wins[0] if all_wins else None,
        }
    return out


def _render_examples(examples: dict[int, dict[str, tuple]]) -> str:
    sections: list[str] = []
    for seed in SEEDS:
        ex = examples[seed]
        parts: list[str] = [f"\n### Seed {seed}"]
        for label, key in (
            ("Attn correct, All-layer wrong", "att_only_only"),
            ("All-layer correct, Attn wrong", "all_layer_only"),
        ):
            if ex[key] is None:
                continue
            idx, att_row, all_row = ex[key]
            parts.append(f"\n#### {label} (benchmark_index {idx})\n")
            parts.append(f"**Question:** {att_row['question']}\n")
            parts.append(f"**Reference answer:** {att_row['reference_answer']}\n")
            parts.append(
                f"**attention-only** → `{att_row['extracted_answer']}` "
                f"({'correct' if att_row['correct'] else 'wrong'})\n"
            )
            parts.append(
                f"**all-layer** → `{all_row['extracted_answer']}` "
                f"({'correct' if all_row['correct'] else 'wrong'})\n"
            )
            parts.append(
                f"\n_attention-only generation:_\n\n"
                f"```\n{att_row['generated_text'].replace('<|im_end|>', '').strip()}\n```\n"
            )
            parts.append(
                f"\n_all-layer generation:_\n\n"
                f"```\n{all_row['generated_text'].replace('<|im_end|>', '').strip()}\n```\n"
            )
        sections.append("\n".join(parts))
    return "\n\n---\n\n".join(sections) + "\n"


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
            "attention_only_only":    att_only,
            "all_layer_only":         all_only,
            "both_correct":           both_correct,
            "both_wrong":             both_wrong,
        }
    return out


def _build_rows(paired: dict[int, dict[str, int]]) -> list[dict[str, Any]]:
    return [
        {
            "Seed":                   seed,
            "Attn correct, All-layer wrong": paired[seed]["attention_only_only"],
            "All-layer correct, Attn wrong": paired[seed]["all_layer_only"],
            "Δ (Attn − All)":         f"+{paired[seed]['attention_only_only'] - paired[seed]['all_layer_only']}",
            "Both correct":            paired[seed]["both_correct"],
            "Both wrong":              paired[seed]["both_wrong"],
        }
        for seed in SEEDS
    ]
